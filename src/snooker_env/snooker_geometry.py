"""Regulation snooker dimensions, layout and deterministic geometric refereeing.

SI units; +Y points from baulk towards black. Keep the repository cloth origin
at 1.05 m and lower the displayed floor so the rail is 0.864 m above it.
"""
from __future__ import annotations

import math
from typing import Mapping

import numpy as np

TABLE_WIDTH = 1.778
TABLE_LENGTH = 3.569
BALL_RADIUS = .02625
SURFACE_Z = 1.05
BALL_Z = SURFACE_Z + BALL_RADIUS
BAULK_Y = -TABLE_LENGTH / 2 + .737
D_RADIUS = .292
RAIL_TOP_Z = SURFACE_Z + .043
FLOOR_Z = RAIL_TOP_Z - .864
REDS = frozenset(range(1, 16))
COLOURS = tuple(range(16, 22))
NAMES = {0: "white", **{n: f"red_{n}" for n in REDS},
         **dict(zip(COLOURS, ("yellow", "green", "brown", "blue", "pink", "black")))}
VALUES = {0: 0, **{n: 1 for n in REDS}, **dict(zip(COLOURS, range(2, 8)))}
SPOTS = {16: (D_RADIUS, BAULK_Y, BALL_Z), 17: (-D_RADIUS, BAULK_Y, BALL_Z),
         18: (0., BAULK_Y, BALL_Z), 19: (0., 0., BALL_Z),
         20: (0., TABLE_LENGTH / 4, BALL_Z),
         21: (0., TABLE_LENGTH / 2 - .324, BALL_Z)}
BREAK_POSITION = (0., BAULK_Y - D_RADIUS / 2, BALL_Z)
Positions = Mapping[int, tuple[float, float, float]]


def rack_positions() -> dict[int, tuple[float, float, float]]:
    positions = {0: BREAK_POSITION, **SPOTS}
    diameter = 2 * BALL_RADIUS + .00001
    number = 1
    for row in range(5):
        for column in range(row + 1):
            positions[number] = ((column - row / 2) * diameter,
                                 SPOTS[20][1] + diameter + row * math.sqrt(3) / 2 * diameter,
                                 BALL_Z)
            number += 1
    return positions


def in_d(x: float, y: float) -> bool:
    """The centre may be on the D boundary, as in the snooker in-hand rule."""
    return y <= BAULK_Y + 1e-10 and x*x + (y - BAULK_Y)**2 <= D_RADIUS**2 + 1e-10


def vacant(point: tuple, positions: Positions, *, ignore: int | None = None) -> bool:
    return all(n == ignore or math.dist(point[:2], p[:2]) >= 2 * BALL_RADIUS + 1e-7
               for n, p in positions.items())


def respot_position(ball: int, positions: Positions) -> tuple[float, float, float]:
    """Own spot, highest vacant spot, then nearest free point towards top cushion.

    Interval union gives the nearest non-touching position without a grid. If
    pink/black cannot fit towards the top, search towards baulk on the same line.
    """
    for candidate in (ball, *reversed(COLOURS)):
        if vacant(SPOTS[candidate], positions, ignore=ball):
            return SPOTS[candidate]
    x, origin, z = SPOTS[ball]
    clearance = 2 * BALL_RADIUS + 2e-7
    intervals = []
    for n, p in positions.items():
        dx = abs(x - p[0])
        if n != ball and dx < clearance:
            dy = math.sqrt(clearance**2 - dx**2)
            intervals.append((p[1] - dy, p[1] + dy))
    for direction in (1, -1):
        y = origin
        for low, high in sorted(intervals, reverse=direction < 0):
            if low <= y <= high:
                y = high + 1e-9 if direction > 0 else low - 1e-9
        if abs(y) <= TABLE_LENGTH / 2 - BALL_RADIUS and vacant((x, y, z), positions, ignore=ball):
            return (x, y, z)
    raise RuntimeError("No legal colour respot on the longitudinal line")


def _segment_clear(a: np.ndarray, b: np.ndarray, obstacles: list[np.ndarray]) -> bool:
    delta = b - a
    length2 = float(delta @ delta)
    if length2 < 1e-16:
        return True
    for point in obstacles:
        t = float(np.clip((point - a) @ delta / length2, 0., 1.))
        if np.linalg.norm(point - (a + t * delta)) < 2 * BALL_RADIUS - 1e-8:
            return False
    return True


def snookered(positions: Positions, targets: set[int] | frozenset[int]) -> bool:
    """Both extreme cue-centre tangent paths to at least one Ball On must be clear.

    Other Balls On do not constitute snookering obstructions. Cushions are not
    snookering balls. In-hand callers must consider legal positions throughout D.
    """
    if 0 not in positions:
        raise ValueError("Supply a cue-ball position for visibility")
    cue = np.array(positions[0][:2])
    blockers = [np.array(p[:2]) for n, p in positions.items() if n and n not in targets]
    for n in targets & positions.keys():
        target = np.array(positions[n][:2])
        delta = target - cue
        distance = float(np.linalg.norm(delta))
        if distance <= 2 * BALL_RADIUS + 1e-8:
            return False
        heading = math.atan2(delta[1], delta[0])
        spread = math.asin(2 * BALL_RADIUS / distance)
        tangent_length = math.sqrt(distance**2 - (2 * BALL_RADIUS)**2)
        endpoints = [cue + tangent_length * np.array([math.cos(heading + sign * spread),
                                                      math.sin(heading + sign * spread)])
                     for sign in (-1, 1)]
        if all(_segment_clear(cue, end, blockers) for end in endpoints):
            return False
    return True


def d_positions(positions: Positions):
    """Deterministic 5 mm in-hand referee grid, including the default position."""
    if vacant(BREAK_POSITION, positions, ignore=0):
        yield BREAK_POSITION
    for y in np.arange(BAULK_Y - D_RADIUS, BAULK_Y + 1e-10, .005):
        for x in np.arange(-D_RADIUS, D_RADIUS + 1e-10, .005):
            p = (float(x), float(y), BALL_Z)
            if in_d(x, y) and vacant(p, positions, ignore=0):
                yield p


def snookered_in_hand(positions: Positions, targets: set[int] | frozenset[int]) -> bool:
    return all(snookered({**positions, 0: p}, targets) for p in d_positions(positions))


def feasible_route(positions: Positions, targets: set[int] | frozenset[int],
                   max_cushions: int = 4) -> bool:
    """Search direct and reflected cushion routes for the automatic miss referee.

    Deterministic geometric approximation: 17 impact offsets per target and up
    to four ideal cushion rebounds. A found clear route proves feasibility;
    absence is not a mathematical proof of impossibility. Hosts may inject a
    stronger route oracle into SnookerMatchEnv.
    """
    cue = np.array(positions[0][:2], dtype=np.float64)
    bounds = np.array([TABLE_WIDTH / 2 - BALL_RADIUS, TABLE_LENGTH / 2 - BALL_RADIUS])
    for n in sorted(targets & positions.keys()):
        target = np.array(positions[n][:2])
        blockers = [np.array(p[:2]) for k, p in positions.items() if k and k not in targets]
        distance = np.linalg.norm(target - cue)
        if distance <= 2 * BALL_RADIUS + 1e-8:
            return True
        # Unfold the rectangle; reflected targets correspond to cushion paths.
        for ix in range(-max_cushions, max_cushions + 1):
            for iy in range(-max_cushions, max_cushions + 1):
                if abs(ix) + abs(iy) > max_cushions:
                    continue
                mirror = np.array([2*ix*bounds[0] + (-1)**ix * target[0],
                                   2*iy*bounds[1] + (-1)**iy * target[1]])
                delta = mirror - cue
                heading = math.atan2(delta[1], delta[0])
                spread = math.asin(min(1., 2 * BALL_RADIUS / np.linalg.norm(delta)))
                for offset in np.linspace(-.98 * spread, .98 * spread, 17):
                    direction = np.array([math.cos(heading + offset), math.sin(heading + offset)])
                    start = cue.copy()
                    for bounce in range(max_cushions + 1):
                        times = np.divide(np.sign(direction) * bounds - start, direction,
                                          out=np.full(2, np.inf), where=np.abs(direction) > 1e-12)
                        axis = int(np.argmin(times))
                        travel = times[axis]
                        # Find the first Ball On along this segment.
                        hit = float('inf')
                        for t in targets & positions.keys():
                            rel = np.array(positions[t][:2]) - start
                            along = float(rel @ direction)
                            perpendicular2 = float(rel @ rel) - along**2
                            if perpendicular2 <= (2 * BALL_RADIUS)**2 and along > 0:
                                hit = min(hit, max(0., along - math.sqrt(max(0., (2 * BALL_RADIUS)**2 - perpendicular2))))
                        end = start + min(travel, hit) * direction
                        if not _segment_clear(start, end, blockers):
                            break
                        if hit <= travel:
                            return True
                        # Gaps in the actual cushions cannot provide a rebound.
                        if ((axis == 0 and abs(end[1]) < .09)
                                or abs(end[1 - axis]) > bounds[1 - axis] - .07):
                            break
                        start = end
                        direction[axis] *= -1
    return False
