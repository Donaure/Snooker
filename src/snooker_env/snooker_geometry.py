"""Regulation snooker dimensions, layout and deterministic geometric refereeing.

SI units; +Y points from baulk towards black. Keep the repository cloth origin
at 1.05 m and lower the displayed floor so the rail is 0.864 m above it.
"""
from __future__ import annotations

import math
from functools import lru_cache
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


def _route_cushions() -> tuple[tuple[float, float], tuple[float, float]]:
    """Straight nose runs from the asset builder, in the cue-centre rectangle.

    Curved jaws are excluded: they do not obey axis-aligned mirror reflection.
    Along a straight nose, the contact point and cue centre share the same
    tangential coordinate, so run endpoints need no extra radius subtraction.
    Import lazily because table_assets also imports the table dimensions here.
    """
    from .snooker_table_assets import CORNER_RUNOUT, MIDDLE_RUNOUT
    return ((MIDDLE_RUNOUT, TABLE_LENGTH / 2 - CORNER_RUNOUT),
            (-TABLE_WIDTH / 2 + CORNER_RUNOUT, TABLE_WIDTH / 2 - CORNER_RUNOUT))


def _route_ray(cue, balls, targets, angle, max_cushions, bounds, runs) -> bool:
    """First-contact ray test; no sampling or physics integration inside."""
    x, y = cue
    dx, dy = math.cos(angle), math.sin(angle)
    diameter2 = (2 * BALL_RADIUS)**2
    for bounce in range(max_cushions + 1):
        tx = (math.copysign(bounds[0], dx) - x) / dx if abs(dx) > 1e-15 else math.inf
        ty = (math.copysign(bounds[1], dy) - y) / dy if abs(dy) > 1e-15 else math.inf
        travel = min(tx, ty)
        if travel < -1e-10:
            return False
        travel = max(0., travel)
        target_hit = blocker_hit = math.inf
        for number, bx, by in balls:
            rx, ry = bx - x, by - y
            along = rx*dx + ry*dy
            cross = rx*dy - ry*dx
            discriminant = diameter2 - cross*cross
            if discriminant < -1e-14:
                continue
            half = math.sqrt(max(0., discriminant))
            if along + half < 0:
                continue
            # Leaving an existing contact is not a new collision. Without this
            # case, a cue touching a non-target would be trapped in every direction.
            if along <= 0 and along + half <= 1e-12:
                continue
            hit = max(0., along - half)
            if number in targets:
                target_hit = min(target_hit, hit)
            else:
                blocker_hit = min(blocker_hit, hit)
        # Simultaneous contact with a non-target is not a legal first contact.
        if target_hit <= travel + 1e-12 and target_hit < blocker_hit - 1e-12:
            return True
        if blocker_hit <= travel + 1e-12 or bounce == max_cushions:
            return False
        x, y = x + travel*dx, y + travel*dy
        if tx <= ty:
            if not runs[0][0] + 1e-12 < abs(y) < runs[0][1] - 1e-12:
                return False
            dx = -dx
        else:
            if not runs[1][0] + 1e-12 < x < runs[1][1] - 1e-12:
                return False
            dy = -dy
    return False


def _route_events(cue, balls, targets, max_cushions, bounds, runs):
    """Critical bearings of the unfolded, clipped circle arrangement.

    Tangencies change ray/circle intersections. Target/blocker circle crossings
    change first-contact order. Circle/rectangle crossings change whether a hit
    precedes a rebound. Run endpoints/corners change valid rebound sequences.
    Between successive events the legal-first-contact answer is constant in
    this ideal model. Full circles can add harmless extra events outside tiles;
    the folded ray test always clips each segment at its actual cushion.
    """
    radius = 2 * BALL_RADIUS
    points = [(sx*bounds[0], sy*bounds[1]) for sx in (-1, 1) for sy in (-1, 1)]
    for sign in (-1, 1):
        points.extend((sign*bounds[0], y) for y in (-runs[0][1], -runs[0][0], *runs[0]))
        points.extend((x, sign*bounds[1]) for x in runs[1])
    for _, x, y in balls:
        for axis in (0, 1):
            center = (x, y)
            for sign in (-1, 1):
                wall = sign*bounds[axis]
                square = radius**2 - (wall - center[axis])**2
                if square >= 0:
                    offset = math.sqrt(square)
                    for value in (center[1-axis] - offset, center[1-axis] + offset):
                        points.append((wall, value) if axis == 0 else (value, wall))
    target_balls = [ball for ball in balls if ball[0] in targets]
    blockers = [ball for ball in balls if ball[0] not in targets]
    for _, ax, ay in target_balls:
        for _, bx, by in blockers:
            dx, dy = bx - ax, by - ay
            distance = math.hypot(dx, dy)
            if 0 < distance <= 2*radius:
                offset = math.sqrt(max(0., radius**2 - distance**2 / 4)) / distance
                for sign in (-1, 1):
                    points.append(((ax+bx)/2 - sign*dy*offset, (ay+by)/2 + sign*dx*offset))
    events = {0., 2*math.pi}
    for ix in range(-max_cushions, max_cushions + 1):
        for iy in range(-max_cushions, max_cushions + 1):
            if abs(ix) + abs(iy) > max_cushions:
                continue
            sx, sy = (-1)**ix, (-1)**iy
            ox, oy = 2*ix*bounds[0] - cue[0], 2*iy*bounds[1] - cue[1]
            for x, y in points:
                events.add(math.atan2(oy + sy*y, ox + sx*x) % (2*math.pi))
            for _, x, y in balls:
                rx, ry = ox + sx*x, oy + sy*y
                distance = math.hypot(rx, ry)
                heading = math.atan2(ry, rx)
                if distance >= radius:
                    spread = math.asin(radius / distance)
                    events.add((heading - spread) % (2*math.pi))
                    events.add((heading + spread) % (2*math.pi))
    return sorted(events)


@lru_cache(maxsize=1)
def _feasible_route_cached(snapshot, targets, max_cushions) -> bool:
    cue = next((x, y) for n, x, y in snapshot if n == 0)
    balls = tuple(ball for ball in snapshot if ball[0] != 0)
    if not targets:
        return False
    bounds = (TABLE_WIDTH / 2 - BALL_RADIUS, TABLE_LENGTH / 2 - BALL_RADIUS)
    if any(abs(cue[axis]) > bounds[axis] + 1e-10 for axis in (0, 1)):
        return False
    runs = _route_cushions()
    # Common unobstructed shots need neither unfolding nor arrangement events.
    for number, x, y in balls:
        if number in targets and _route_ray(cue, balls, targets, math.atan2(y-cue[1], x-cue[0]),
                                           0, bounds, runs):
            return True
    # Resolve direct visibility first, then search the full rebound budget.
    for limit in dict.fromkeys((0, max_cushions)):
        events = _route_events(cue, balls, targets, limit, bounds, runs)
        for low, high in zip(events, events[1:]):
            if _route_ray(cue, balls, targets, (low + high)/2, limit, bounds, runs):
                return True
        # Include isolated tangent solutions; simultaneous illegal contacts and
        # exact jaw transitions are rejected by the same first-contact test.
        for angle in events[:-1]:
            if _route_ray(cue, balls, targets, angle, limit, bounds, runs):
                return True
    return False


def feasible_route(positions: Positions, targets: set[int] | frozenset[int],
                   max_cushions: int = 4) -> bool:
    """Geometric first-contact search, at most four ideal rebounds by default.

    Sweep critical angular intervals instead of a fixed angular sample grid.
    Rebounds use straight nose runs shared with table assets; pockets and curved
    jaws cannot supply mirror rebounds. Speed, spin, cue access and 3D cushion
    response are outside this model. Cache only the latest exact XY snapshot,
    target set and rebound budget; consecutive restores reuse that result.
    """
    if isinstance(max_cushions, bool) or not isinstance(max_cushions, int) or max_cushions < 0:
        raise ValueError("max_cushions must be a nonnegative integer")
    if 0 not in positions:
        raise ValueError("Supply a cue-ball position for route search")
    snapshot = tuple(sorted((n, float(p[0]), float(p[1])) for n, p in positions.items()))
    if not all(math.isfinite(x) and math.isfinite(y) for _, x, y in snapshot):
        raise ValueError("Route positions must be finite")
    return _feasible_route_cached(snapshot, frozenset(targets & positions.keys()) - {0}, max_cushions)
