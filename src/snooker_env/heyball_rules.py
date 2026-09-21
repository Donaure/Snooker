"""Pure adjudication for the project's specified Heyball single-rack rules."""
from __future__ import annotations

from dataclasses import dataclass

PLAYERS = ("player_0", "player_1")
GROUPS = {"solids": frozenset(range(1, 8)), "stripes": frozenset(range(9, 16))}


def other_player(player: str) -> str:
    return PLAYERS[1 - PLAYERS.index(player)]


def ball_group(ball: int) -> str | None:
    return next((name for name, balls in GROUPS.items() if ball in balls), None)


@dataclass(frozen=True)
class ShotEvents:
    """Facts from a completed shot; 0 identifies the cue ball.

    Contacts in the same integration step are adjudicated together. A rail
    contact must begin after the first object contact, rather than before it.
    """

    first_contacts: tuple[int, ...] = ()
    pocketed: tuple[int, ...] = ()
    off_table: tuple[int, ...] = ()
    rail_after_contact: bool = False
    fouls: tuple[str, ...] = ()
    elevation_degrees: float = 0.0
    simulation_seconds: float = 0.0


@dataclass(frozen=True)
class ShotDecision:
    next_player: str
    assigned_group: str | None = None
    ball_in_hand: bool = False
    rerack: bool = False
    winner: str | None = None
    reason: str = "turn_passed"
    fouls: tuple[str, ...] = ()


def adjudicate(*, player: str, group: str | None, remaining: frozenset[int],
               is_break: bool, speed: float, events: ShotEvents) -> ShotDecision:
    """Use the balls remaining BEFORE the shot for all 8-ball decisions."""
    opponent = other_player(player)
    pots = set(events.pocketed)
    off = set(events.off_table)
    fouls = list(events.fouls)
    if is_break and (speed < 1.5 or not events.first_contacts):
        return ShotDecision(opponent, winner=opponent, reason="illegal_break")
    if 8 in off:
        return ShotDecision(opponent, winner=opponent, reason="eight_off_table")
    if is_break and 8 in pots:
        return ShotDecision(player, rerack=True, reason="eight_on_break_rerack")
    if 0 in pots:
        fouls.append("scratch")
    if off:
        fouls.append("ball_off_table")
    if not is_break:
        targets = (GROUPS[group] & remaining if group is not None
                   else (GROUPS["solids"] | GROUPS["stripes"]) & remaining)
        if group is not None and not targets:
            targets = frozenset({8})
        if not events.first_contacts:
            fouls.append("no_object_contact")
        elif any(ball not in targets for ball in events.first_contacts):
            fouls.append("wrong_first_contact")
        if not pots - {0} and not events.rail_after_contact:
            fouls.append("no_rail")
        if 8 in pots:
            ready = group is not None and not (GROUPS[group] & remaining)
            win = ready and not fouls
            return ShotDecision(player if win else opponent,
                                winner=player if win else opponent,
                                reason="legal_eight" if win else "illegal_eight",
                                fouls=tuple(dict.fromkeys(fouls)))
    if fouls:
        return ShotDecision(opponent, ball_in_hand=True, reason="foul",
                            fouls=tuple(dict.fromkeys(fouls)))
    if is_break:
        return ShotDecision(player if pots else opponent,
                            reason="continue" if pots else "turn_passed")
    assigned = None
    if group is None and events.first_contacts:
        first_groups = {ball_group(ball) for ball in events.first_contacts}
        # A simultaneous mixed-group contact cannot unambiguously assign a group.
        if len(first_groups) == 1:
            candidate = first_groups.pop()
            if candidate is not None and pots & GROUPS[candidate]:
                assigned = candidate
        group = assigned
    continues = group is not None and bool(pots & GROUPS[group])
    return ShotDecision(player if continues else opponent, assigned_group=assigned,
                        reason="continue" if continues else "turn_passed")
