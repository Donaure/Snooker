"""Timed player_0/player_1 Heyball rack with an angle/speed action interface."""
from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import asdict
import math
import numbers
import threading
import time
from typing import Protocol, Mapping

import numpy as np

from .heyball_physics import (
    BREAK_POSITION, FOOT_SPOT, HEAD_STRING_Y, HeyballPhysics, rack_positions,
)
from .heyball_rules import GROUPS, PLAYERS, ShotEvents, adjudicate, other_player
from .table_geometry import BALL_RADIUS, TABLE_LENGTH, TABLE_SURFACE_Z, TABLE_WIDTH


class MatchPhysics(Protocol):
    pockets: Mapping[str, tuple[float, float, float]]

    def reset(self, positions: Mapping[int, tuple[float, float, float]]) -> None: ...
    def positions(self) -> dict[int, tuple[float, float, float]]: ...
    def execute(self, angle: float, speed: float) -> ShotEvents: ...
    def placement_error(self, x: float, y: float) -> str | None: ...
    def place_cue_ball(self, x: float, y: float) -> None: ...


def _scalar(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be one real scalar")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class HeyballMatchEnv:
    """One rack, beginning with player_0's break.

    Angles are radians counterclockwise from world +X; speed is cue speed in
    m/s. Call get_game_state, then submit_shot with its turn_id. The monotonic
    turn clock starts when the turn is exposed, includes ball-in-hand UI time,
    and excludes internal physics/planning. Every continuing shot gets 90 s.
    Use as a context manager, or call close to cancel the deadline watchdog.
    """

    def __init__(self, *, seed: int | None = None, turn_seconds: float = 90.0,
                 physics: MatchPhysics | None = None) -> None:
        self.turn_seconds = _scalar(turn_seconds, "turn_seconds")
        if self.turn_seconds <= 0:
            raise ValueError("turn_seconds must be positive")
        self.physics = physics if physics is not None else HeyballPhysics()
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._deadline: float | None = None
        self._paused_seconds: float | None = None
        self._generation = 0
        self._closed = False
        self.turn_id = 0
        self.reset(seed=seed)

    def __enter__(self) -> HeyballMatchEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _cancel_timer(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._deadline = None

    def _start_clock(self, seconds: float | None = None) -> None:
        self._cancel_timer()
        duration = self.turn_seconds if seconds is None else max(0., seconds)
        self._deadline = time.monotonic() + duration
        generation = self._generation

        def expire() -> None:
            with self._lock:
                if generation == self._generation and not self._closed:
                    self._check_timeout()

        self._timer = threading.Timer(duration, expire)
        self._timer.daemon = True
        self._timer.start()

    def _check_timeout(self) -> None:
        if self.winner is None and self._deadline is not None and time.monotonic() >= self._deadline:
            self.winner = other_player(self.current_player)
            self.reason = "timeout"
            self._cancel_timer()

    def _new_turn(self) -> None:
        self.turn_id += 1
        self._start_clock()

    def reset(self, *, seed: int | None = None) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("Match is closed")
            self._cancel_timer()
            self._paused_seconds = None
            self.rng = np.random.default_rng(seed)
            self.current_player = PLAYERS[0]
            self.groups: dict[str, str | None] = dict.fromkeys(PLAYERS)
            self.is_break = True
            self.ball_in_hand = False
            self.winner: str | None = None
            self.reason = "break"
            self.last_shot: dict | None = None
            self.shot_count = 0
            self.rerack_count = 0
            self.physics.reset(rack_positions(self.rng))
            self._new_turn()
            return self.get_game_state()

    def get_game_state(self, player: str | None = None) -> dict:
        """Return a detached JSON-compatible snapshot; reading never resets time."""
        with self._lock:
            if player is not None and player not in PLAYERS:
                raise ValueError("Unknown player")
            self._check_timeout()
            positions = self.physics.positions()
            group = self.groups[self.current_player]
            targets = ((GROUPS[group] & positions.keys()) if group else
                       ((GROUPS["solids"] | GROUPS["stripes"]) & positions.keys()))
            if group and not targets:
                targets = {8}
            if self.is_break:
                targets = set(positions) - {0}
            xmin, xmax, ymin, ymax = getattr(self.physics, "playing_bounds",
                                            (-TABLE_WIDTH / 2, TABLE_WIDTH / 2,
                                             -TABLE_LENGTH / 2, TABLE_LENGTH / 2))
            return {
                "current_player": self.current_player, "players": list(PLAYERS),
                "turn_id": self.turn_id, "shot_count": self.shot_count,
                "rerack_count": self.rerack_count, "is_break": self.is_break,
                "open_table": group is None, "groups": self.groups.copy(),
                "legal_targets": sorted(targets), "ball_in_hand": self.ball_in_hand,
                "phase": "finished" if self.winner else "ball_in_hand" if self.ball_in_hand else "shot",
                "winner": self.winner, "terminated": self.winner is not None,
                "reason": self.reason,
                "seconds_remaining": (self._paused_seconds if self._paused_seconds is not None else
                                      max(0., self._deadline - time.monotonic()) if self._deadline else 0.),
                "clock_paused": self._paused_seconds is not None,
                "turn_seconds": self.turn_seconds,
                "balls": [{"number": n, "active": n in positions,
                           "position": list(positions[n]) if n in positions else None}
                          for n in range(16)],
                "table": {"width": xmax - xmin, "length": ymax - ymin,
                          "playing_bounds": [xmin, xmax, ymin, ymax],
                          "surface_z": TABLE_SURFACE_Z, "ball_radius": BALL_RADIUS,
                          "foot_spot": list(FOOT_SPOT), "head_string_y": HEAD_STRING_Y,
                          "break_position": list(BREAK_POSITION),
                          "pockets": {k: list(v) for k, v in self.physics.pockets.items()}},
                "action": {"angle_units": "radians", "angle_zero": "+X",
                           "angle_positive": "counterclockwise toward +Y",
                           "speed_units": "m/s", "speed_kind": "cue_speed",
                           "minimum_break_speed": 1.5},
                "last_shot": copy.deepcopy(self.last_shot),
            }

    def _check_actor(self, player: str, turn_id: int) -> None:
        self._check_timeout()
        if self._closed or self.winner is not None:
            raise RuntimeError("Match is closed or finished")
        if player != self.current_player:
            raise ValueError("It is not this player's turn")
        if turn_id != self.turn_id:
            raise ValueError("Stale turn_id")

    def preview_cue_ball(self, player: str, x: float, y: float, *, turn_id: int) -> str | None:
        """Validate a preview without committing it or restarting the clock."""
        with self._lock:
            self._check_actor(player, turn_id)
            if not self.ball_in_hand:
                raise ValueError("Ball in hand is not available")
            return self.physics.placement_error(_scalar(x, "x"), _scalar(y, "y"))

    def confirm_cue_ball(self, player: str, x: float, y: float, *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            if not self.ball_in_hand:
                raise ValueError("Ball in hand is not available")
            self.physics.place_cue_ball(_scalar(x, "x"), _scalar(y, "y"))
            self.ball_in_hand = False
            return self.get_game_state()

    def submit_shot(self, player: str, angle: float, speed: float, *, turn_id: int) -> dict:
        """Execute exactly two scalar action values, then expose the next turn.

        Invalid requests leave the clock running. An infeasible stroke or
        simulation error raises, preserving the table and unused decision time.
        """
        with self._lock:
            self._check_actor(player, turn_id)
            if self._paused_seconds is not None:
                raise ValueError("Resume the turn clock before shooting")
            if self.ball_in_hand:
                raise ValueError("Confirm the cue-ball position before shooting")
            angle, speed = _scalar(angle, "angle"), _scalar(speed, "speed")
            if speed < 0 or (speed == 0 and not self.is_break):
                raise ValueError("Cue speed must be positive")
            self._check_actor(player, turn_id)
            remaining = frozenset(self.physics.positions()) - {0}
            unused = max(0., self._deadline - time.monotonic())
            self._cancel_timer()
            try:
                events = (ShotEvents() if self.is_break and speed < 1.5
                          else self.physics.execute(angle, speed))
            except Exception:
                self._start_clock(unused)
                raise
            decision = adjudicate(player=player, group=self.groups[player], remaining=remaining,
                                  is_break=self.is_break, speed=speed, events=events)
            self.shot_count += 1
            self.last_shot = {"player": player, "angle": angle, "speed": speed,
                              "events": asdict(events), "decision": asdict(decision)}
            if decision.assigned_group:
                self.groups[player] = decision.assigned_group
                self.groups[other_player(player)] = ("stripes" if decision.assigned_group == "solids"
                                                     else "solids")
            self.current_player = decision.next_player
            self.winner = decision.winner
            self.reason = decision.reason
            self.ball_in_hand = decision.ball_in_hand
            self.is_break = decision.rerack
            if decision.rerack:
                self.rerack_count += 1
                self.groups = dict.fromkeys(PLAYERS)
                self.physics.reset(rack_positions(self.rng))
            if self.winner is None:
                self._new_turn()
            return self.get_game_state()

    def forfeit(self, player: str, *, turn_id: int, reason: str = "forfeit") -> dict:
        with self._lock:
            self._check_timeout()
            if self.winner is not None:
                return self.get_game_state()
            self._check_actor(player, turn_id)
            self.winner = other_player(player)
            self.reason = reason
            self._cancel_timer()
            return self.get_game_state()

    @contextmanager
    def pause_for_placement(self):
        """Opt-in operator pause for interactive recordings, never competition.

        Confirming placement retains the unused turn budget. This context does
        not permit submitting a shot while the clock is suspended.
        """
        with self._lock:
            self._check_actor(self.current_player, self.turn_id)
            if not self.ball_in_hand or self._paused_seconds is not None:
                raise ValueError("Only an unpaused ball-in-hand turn can be suspended")
            token = self.turn_id
            self._paused_seconds = max(0., self._deadline - time.monotonic())
            self._cancel_timer()
        try:
            yield
        finally:
            with self._lock:
                remaining = self._paused_seconds
                self._paused_seconds = None
                if not self._closed and self.winner is None and token == self.turn_id:
                    self._start_clock(remaining)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_timer()
