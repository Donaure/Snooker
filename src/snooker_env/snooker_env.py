"""Timed snooker frame with heading, speed and optional Cartesian contact offsets."""
from __future__ import annotations

import copy
from dataclasses import asdict
import math
import threading
import time
from typing import Callable, Protocol

from .cue_stroke import validate_cue_offset
from .heyball_env import MatchPhysics, _scalar
from .snooker_geometry import (
    BALL_RADIUS, BAULK_Y, BREAK_POSITION, COLOURS, D_RADIUS, NAMES, REDS,
    SPOTS, SURFACE_Z, TABLE_LENGTH, TABLE_WIDTH, VALUES, Positions,
    d_positions, feasible_route, in_d, rack_positions, respot_position,
    snookered, snookered_in_hand,
)
from .snooker_physics import SnookerPhysics
from .snooker_rules import PLAYERS, ShotEvents, adjudicate, balls_on, other_player


class SnookerMatchPhysics(MatchPhysics, Protocol):
    def execute(self, angle: float, speed: float, u: float = 0., v: float = 0.) -> ShotEvents: ...


class SnookerMatchEnv:
    """One frame; every shot has 120 seconds, including pre-shot choices.

    get_match_state returns a detached JSON snapshot. submit_shot takes just
    angle (radians CCW from +X), cue speed (m/s) and optional unit-disk u/v
    contact coordinates, plus actor/turn credentials. The contact point uses
    the horizontal heading frame and stays fixed when the cue is elevated.
    Optional nominations, foul choices and D placement use separate methods.
    Reading or modifying pre-shot choices never restarts the clock. Physics and
    automatic refereeing do not consume a player's budget. Use a context manager.
    """

    def __init__(self, *, physics: SnookerMatchPhysics | None = None,
                 turn_seconds: float = 120.,
                 route_oracle: Callable[[Positions, frozenset[int]], bool] = feasible_route) -> None:
        self.turn_seconds = _scalar(turn_seconds, "turn_seconds")
        if self.turn_seconds <= 0:
            raise ValueError("turn_seconds must be positive")
        self.physics = physics if physics is not None else SnookerPhysics()
        self.route_oracle = route_oracle
        self._lock = threading.RLock()
        self._timer: threading.Timer | None = None
        self._deadline: float | None = None
        self._generation = 0
        self._closed = False
        self.turn_id = 0
        self.reset()

    def __enter__(self) -> SnookerMatchEnv:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _cancel_clock(self) -> None:
        self._generation += 1
        if self._timer is not None:
            self._timer.cancel()
        self._timer = None
        self._deadline = None

    def _start_clock(self, seconds: float | None = None) -> None:
        self._cancel_clock()
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
        if not self.terminated and self._deadline is not None and time.monotonic() >= self._deadline:
            self.terminated = True
            self.winner = other_player(self.current_player)
            self.reason = "timeout"
            self._cancel_clock()

    def _new_turn(self) -> None:
        self.turn_id += 1
        self.nomination = None
        self.free_ball = None
        self._start_clock()

    def reset(self) -> dict:
        with self._lock:
            if self._closed:
                raise RuntimeError("Match is closed")
            self._cancel_clock()
            self.current_player = PLAYERS[0]
            self.scores = dict.fromkeys(PLAYERS, 0)
            self.highest_breaks = dict.fromkeys(PLAYERS, 0)
            self.current_break = 0
            self.stage = "red"
            self.clearance_ball = 16
            self.ball_in_hand = True
            self.free_ball_available = False
            self.pending_foul: dict | None = None
            self.nomination: int | None = None
            self.free_ball: int | None = None
            self.terminated = False
            self.winner: str | None = None
            self.reason = "break"
            self.last_shot: dict | None = None
            self.shot_count = 0
            self.physics.reset(rack_positions())
            self._new_turn()
            return self.get_match_state()

    def _targets(self, *, nominated: bool = True) -> frozenset[int]:
        return balls_on(self.stage, frozenset(self.physics.positions()), self.clearance_ball,
                        self.nomination if nominated else None)

    def get_match_state(self, player: str | None = None) -> dict:
        """Read-only observation, including both scores, targets, choices and clock."""
        with self._lock:
            if player is not None and player not in PLAYERS:
                raise ValueError("Unknown player")
            self._check_timeout()
            positions = self.physics.positions()
            options = []
            if self.pending_foul and not self.terminated:
                options = ["accept", "play_again"]
                if self.pending_foul["miss"]:
                    options.append("restore")
            return {
                "players": list(PLAYERS), "current_player": self.current_player,
                "turn_id": self.turn_id, "shot_count": self.shot_count,
                "scores": self.scores.copy(), "current_break": self.current_break,
                "highest_breaks": self.highest_breaks.copy(),
                "stage": self.stage, "clearance_ball": self.clearance_ball if self.stage == "clearance" else None,
                "legal_targets": sorted(self._targets()), "nominated_colour": self.nomination,
                "snookered_at_current_position": (snookered(positions, self._targets())
                                                   if 0 in positions and self._targets() else False),
                "free_ball_available": self.free_ball_available, "nominated_free_ball": self.free_ball,
                "ball_in_hand": self.ball_in_hand, "foul_options": options,
                "phase": "finished" if self.terminated else "foul_choice" if options else "shot",
                "terminated": self.terminated, "winner": self.winner,
                "draw": self.terminated and self.winner is None, "reason": self.reason,
                "turn_seconds": self.turn_seconds,
                "seconds_remaining": max(0., self._deadline - time.monotonic()) if self._deadline else 0.,
                "balls": [{"number": n, "name": NAMES[n], "value": VALUES[n],
                           "active": n in positions, "position": list(positions[n]) if n in positions else None}
                          for n in range(22)],
                "table": {"width": TABLE_WIDTH, "length": TABLE_LENGTH, "surface_z": SURFACE_Z,
                          "ball_radius": BALL_RADIUS, "baulk_y": BAULK_Y, "d_radius": D_RADIUS,
                          "colour_spots": {str(n): list(p) for n, p in SPOTS.items()},
                          "pockets": {name: list(p) for name, p in self.physics.pockets.items()}},
                "action": {"angle_units": "radians", "angle_zero": "+X",
                           "angle_positive": "counterclockwise toward +Y", "speed_units": "m/s",
                           "speed_kind": "cue_speed", "cue_offset": [0., 0.],
                           "fields": ["angle", "speed", "u", "v"],
                           "cue_offset_units": "ball_radius_fraction",
                           "cue_offset_frame": "horizontal_heading_right_world_up",
                           "cue_offset_constraint": "u*u + v*v < 1",
                           "cue_offset_max_m": BALL_RADIUS,
                           "contact_point_fixed_during_elevation": True,
                           "elevation": "automatic"},
                "last_shot": copy.deepcopy(self.last_shot),
            }

    get_game_state = get_match_state

    def _check_actor(self, player: str, turn_id: int) -> None:
        self._check_timeout()
        if self._closed or self.terminated:
            raise RuntimeError("Match is closed or finished")
        if player != self.current_player:
            raise ValueError("It is not this player's turn")
        if turn_id != self.turn_id:
            raise ValueError("Stale turn_id")

    def nominate_colour(self, player: str, ball: int, *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            if self.stage != "colour" or isinstance(ball, bool) or ball not in COLOURS:
                raise ValueError("Nominate a colour only after scoring a red")
            self.nomination = int(ball)
            self.pending_foul = None  # A pre-shot choice accepts the position.
            return self.get_match_state()

    def nominate_free_ball(self, player: str, ball: int | None, *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            if not self.free_ball_available:
                raise ValueError("No free ball is available")
            if ball is not None and (isinstance(ball, bool) or ball == 0
                                     or ball not in self.physics.positions() or ball in self._targets()):
                raise ValueError("A free ball must be an active object ball that is not Ball On")
            self.free_ball = None if ball is None else int(ball)
            self.pending_foul = None
            return self.get_match_state()

    def place_cue_ball(self, player: str, x: float, y: float, *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            if not self.ball_in_hand:
                raise ValueError("Cue ball is not in hand")
            x, y = _scalar(x, "x"), _scalar(y, "y")
            if not in_d(x, y):
                raise ValueError("Place the cue-ball centre within the D")
            self.physics.place_cue_ball(x, y)
            self.pending_foul = None
            # Remains in hand until the stroke; repositioning uses the same clock.
            return self.get_match_state()

    confirm_cue_ball = place_cue_ball

    def choose_foul(self, player: str, choice: str, *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            pending = self.pending_foul
            if pending is None or choice not in ("accept", "play_again", "restore"):
                raise ValueError("No such foul choice is available")
            if choice == "restore" and not pending["miss"]:
                raise ValueError("Restoration requires Foul and a Miss")
            if choice == "accept":
                self.pending_foul = None
                return self.get_match_state()
            if choice == "restore":
                before = pending["before"]
                self.physics.reset(before["positions"])
                self.stage = before["stage"]
                self.clearance_ball = before["clearance_ball"]
                self.ball_in_hand = before["ball_in_hand"]
            self.current_player = pending["offender"]
            self.pending_foul = None
            self.free_ball_available = False
            self.reason = "replay_restored" if choice == "restore" else "play_again"
            self._new_turn()
            return self.get_match_state()

    def _infer_colour(self, angle: float, positions: Positions) -> int:
        """An unobstructed direct aim makes a colour obvious before the stroke."""
        cue = positions[0]
        dx, dy = math.cos(angle), math.sin(angle)
        hits = []
        for n, p in positions.items():
            if n == 0:
                continue
            x, y = p[0] - cue[0], p[1] - cue[1]
            along, across = x*dx + y*dy, abs(x*dy - y*dx)
            if along > 0 and across < 2 * BALL_RADIUS:
                hits.append((along - math.sqrt((2 * BALL_RADIUS)**2 - across**2), n))
        hits.sort()
        if hits and hits[0][1] in COLOURS and (len(hits) == 1 or hits[1][0] - hits[0][0] > 1e-6):
            return hits[0][1]
        raise ValueError("Colour target is not obvious; call nominate_colour before submitting")

    def submit_shot(self, player: str, angle: float, speed: float,
                    u: float = 0., v: float = 0., *, turn_id: int) -> dict:
        with self._lock:
            self._check_actor(player, turn_id)
            angle, speed = _scalar(angle, "angle"), _scalar(speed, "speed")
            if speed <= 0:
                raise ValueError("Cue speed must be positive")
            u, v = validate_cue_offset(u, v)
            positions = self.physics.positions()
            nomination = self.nomination
            if self.stage == "colour" and nomination is None:
                nomination = self._infer_colour(angle, positions)
            self._check_actor(player, turn_id)
            unused = max(0., self._deadline - time.monotonic())
            before = {"positions": positions, "stage": self.stage, "clearance_ball": self.clearance_ball,
                      "ball_in_hand": self.ball_in_hand}
            self._cancel_clock()
            try:
                events = self.physics.execute(angle, speed, u, v)
                kwargs = dict(stage=self.stage, clearance_ball=self.clearance_ball,
                              remaining=frozenset(positions) - {0}, nomination=nomination,
                              free_ball=self.free_ball, events=events)
                decision = adjudicate(**kwargs)
                if "no_object_contact" in decision.fouls or "wrong_first_contact" in decision.fouls:
                    targets = (frozenset({self.free_ball}) if self.free_ball is not None else
                               balls_on(self.stage, frozenset(positions), self.clearance_ball, nomination))
                    decision = adjudicate(**kwargs, route_available=self.route_oracle(positions, targets))
                after = self.physics.positions()
                for n in decision.respot:
                    after[n] = respot_position(n, after)
                next_targets = balls_on(decision.stage, frozenset(after), decision.clearance_ball)
                free_available = False
                if decision.penalty and not decision.frame_over:
                    free_available = (snookered_in_hand(after, next_targets) if decision.ball_in_hand
                                      else snookered(after, next_targets))
                if decision.ball_in_hand and not decision.frame_over:
                    try:
                        after[0] = next(d_positions(after))
                    except StopIteration as error:
                        raise RuntimeError("No vacant position in the D") from error
                self.physics.reset(after)
            except Exception:
                self.physics.reset(positions)
                self._start_clock(unused)
                raise
            self.shot_count += 1
            self.last_shot = {"player": player, "angle": angle, "speed": speed,
                              "u": u, "v": v,
                              "nomination": nomination, "free_ball": self.free_ball,
                              "events": asdict(events), "decision": asdict(decision)}
            self.scores[player] += decision.points
            self.scores[other_player(player)] += decision.penalty
            self.current_break += decision.points
            self.highest_breaks[player] = max(self.highest_breaks[player], self.current_break)
            if not decision.continues:
                self.current_break = 0
            self.stage, self.clearance_ball = decision.stage, decision.clearance_ball
            self.ball_in_hand = decision.ball_in_hand
            self.free_ball_available = free_available
            self.pending_foul = ({"offender": player, "before": before, "miss": decision.miss}
                                 if decision.penalty and not decision.frame_over else None)
            self.reason = ("foul_and_miss" if decision.miss else "foul" if decision.penalty else
                           "continue" if decision.continues else "turn_passed")
            if decision.frame_over:
                self.terminated = True
                a, b = (self.scores[p] for p in PLAYERS)
                self.winner = PLAYERS[0] if a > b else PLAYERS[1] if b > a else None
                self.reason = "draw" if a == b else "frame_complete"
            else:
                if not decision.continues:
                    self.current_player = other_player(player)
                self._new_turn()
            return self.get_match_state()

    def forfeit(self, player: str, *, turn_id: int, reason: str = "forfeit") -> dict:
        with self._lock:
            self._check_timeout()
            if not self.terminated:
                self._check_actor(player, turn_id)
                self.terminated = True
                self.winner = other_player(player)
                self.reason = reason
                self._cancel_clock()
            return self.get_match_state()

    def close(self) -> None:
        with self._lock:
            self._closed = True
            self._cancel_clock()
