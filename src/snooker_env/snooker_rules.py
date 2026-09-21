"""Pure adjudication for the requested single-frame snooker match rules."""
from __future__ import annotations

from dataclasses import dataclass

from .heyball_rules import PLAYERS, ShotEvents, other_player
from .snooker_geometry import COLOURS, REDS, VALUES


@dataclass(frozen=True)
class SnookerDecision:
    points: int
    penalty: int
    continues: bool
    stage: str
    clearance_ball: int
    respot: tuple[int, ...]
    ball_in_hand: bool
    fouls: tuple[str, ...]
    miss: bool
    frame_over: bool


def balls_on(stage: str, remaining: set[int] | frozenset[int], clearance_ball: int,
             nomination: int | None = None) -> frozenset[int]:
    if stage == "red":
        return REDS & remaining
    if stage == "colour":
        return frozenset({nomination}) if nomination is not None else frozenset(COLOURS) & remaining
    return frozenset({clearance_ball}) & remaining


def adjudicate(*, stage: str, clearance_ball: int, remaining: frozenset[int],
               nomination: int | None, free_ball: int | None,
               events: ShotEvents, route_available: bool = False) -> SnookerDecision:
    """All inputs describe the pre-shot table; no score is awarded on a foul.

    A free ball acquires the Ball On's value, including for foul penalties.
    A free ball alone in clearance is respotted and leaves the real colour on.
    """
    if stage not in ("red", "colour", "clearance"):
        raise ValueError("Unknown snooker stage")
    if stage == "colour" and nomination not in COLOURS:
        raise ValueError("Nominate one colour before adjudicating")
    targets = balls_on(stage, remaining, clearance_ball, nomination)
    value = 1 if stage == "red" else VALUES[next(iter(targets))]
    pots, off = set(events.pocketed), set(events.off_table)
    first = set(events.first_contacts)
    allowed = targets | ({free_ball} if free_ball is not None else set())
    fouls = list(events.fouls)
    wrong_first = not first or bool(first - allowed) or (free_ball is not None and free_ball not in first)
    if wrong_first:
        fouls.append("no_object_contact" if not first else "wrong_first_contact")
    if 0 in pots:
        fouls.append("scratch")
    if off:
        fouls.append("ball_off_table")
    if pots - allowed - {0}:
        fouls.append("wrong_ball_potted")
    after = remaining - pots - off
    final_black = stage == "clearance" and clearance_ball == 21
    respot = (pots | off) & set(COLOURS)
    if fouls:
        involved = (first if wrong_first else set()) | (pots - allowed) | off
        penalty = max([4, value] + [value if n == free_ball else VALUES[n] for n in involved])
        next_stage = "red" if after & REDS else "clearance"
        return SnookerDecision(0, penalty, False, next_stage, clearance_ball,
                               tuple(sorted(respot, reverse=True)), 0 in pots | off,
                               tuple(dict.fromkeys(fouls)), wrong_first and route_available,
                               final_black)
    scoring = pots & allowed
    points = len(scoring) if stage == "red" else value if scoring else 0
    if stage == "clearance":
        # Only removal of the real Ball On advances the clearance order.
        respot -= targets
        next_ball = clearance_ball + int(clearance_ball in pots)
        next_stage = "clearance"
    else:
        next_ball = clearance_ball
        next_stage = "colour" if stage == "red" and points else "red" if after & REDS else "clearance"
    if free_ball is not None and free_ball in pots:
        respot.add(free_ball)
    return SnookerDecision(points, 0, bool(points), next_stage, next_ball,
                           tuple(sorted(respot, reverse=True)), False, (), False,
                           final_black and bool(points))


__all__ = ["PLAYERS", "ShotEvents", "SnookerDecision", "adjudicate", "balls_on", "other_player"]
