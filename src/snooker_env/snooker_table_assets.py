"""Shared, analytic snooker table geometry in metres.

The pocket openings are illustrative rounded snooker templates, not WPBSA
certified templates. Each collision mesh is convex; no convex hull spans a hole.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .snooker_geometry import SURFACE_Z, TABLE_LENGTH, TABLE_WIDTH

HALF_WIDTH = TABLE_WIDTH / 2
HALF_LENGTH = TABLE_LENGTH / 2
SLATE_MARGIN = .125
SLATE_THICKNESS = .045
CUSHION_NOSE_HEIGHT = .03675
# Rounded leading edge, broad upper cloth shoulder and sloping lower face.
# Coordinates are (distance outward from the nose, height above cloth).
CUSHION_PROFILE = ((0., .03675), (.0003, .0378), (.0012, .03865),
                   (.018, .043), (.055, .043), (.055, .003),
                   (.008, .011), (.0003, .0357))
CORNER_THROAT = .090
MIDDLE_THROAT = .100
CORNER_JAW_RADIUS = .025
MIDDLE_JAW_RADIUS = .045
CORNER_RUNOUT = CORNER_THROAT / math.sqrt(2)
MIDDLE_RUNOUT = MIDDLE_THROAT / 2 + MIDDLE_JAW_RADIUS


@dataclass(frozen=True)
class Pocket:
    name: str
    center: tuple[float, float]
    drop_radius: float
    kind: str


POCKETS = tuple(
    [Pocket(f'pocket_{i}', (sx * (HALF_WIDTH + .050), sy * (HALF_LENGTH + .050)),
            .065, 'corner')
     for i, (sx, sy) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1)))]
    + [Pocket(f'pocket_{4+i}', (sx * (HALF_WIDTH + .075), 0.), .060, 'middle')
       for i, sx in enumerate((-1, 1))]
)


def cloth_supports(x: float, y: float) -> bool:
    """Slate support with curved mouths opening through the outer slate edge."""
    if abs(x) > HALF_WIDTH+SLATE_MARGIN or abs(y) > HALF_LENGTH+SLATE_MARGIN:
        return False
    for pocket in POCKETS:
        px, py = pocket.center
        outward_x = math.copysign(1., px)*x
        dx = max(0., abs(px)-outward_x)
        if dx >= pocket.drop_radius:
            continue
        half = math.sqrt(pocket.drop_radius**2-dx**2)
        if pocket.kind == 'middle':
            if abs(y) < half:
                return False
        elif math.copysign(1., py)*y > abs(py)-half:
            return False
    return True


def profile_vertices(start, end, outward) -> list[tuple[float, float, float]]:
    """Exact convex straight-cushion fixture vertices, in world coordinates."""
    return [(point[0] + outward[0]*u, point[1] + outward[1]*u, SURFACE_Z + z)
            for point in (start, end) for u, z in CUSHION_PROFILE]


def extrusion_faces(count: int) -> list[tuple[int, int, int]]:
    faces = [(0, i+1, i) for i in range(1, count-1)]
    faces += [(count, count+i, count+i+1) for i in range(1, count-1)]
    for i in range(count):
        j = (i+1) % count
        faces += [(i, j, count+j), (i, count+j, count+i)]
    return faces


def cloth_polygons(arc_segments: int = 24) -> list[list[tuple[float, float]]]:
    """Convex slate slabs around semicircular side and corner slatefall cutouts.

    The circles open outwards through the edge of the slate. Pocket irons and
    leather support the net; no artificial cloth ring floats behind the hole.
    Arc chord error is below 0.15 mm with the default 24 quarter-circle steps.
    """
    core = HALF_WIDTH - .018
    xmax, ymax = HALF_WIDTH + SLATE_MARGIN, HALF_LENGTH + SLATE_MARGIN
    polygons = [[(-core, -ymax), (core, -ymax), (core, ymax), (-core, ymax)]]
    for sign in (-1, 1):
        pockets = sorted((p for p in POCKETS if p.center[0]*sign > 0),
                         key=lambda p: p.center[1])
        breaks = {core, xmax}
        for p in pockets:
            cx = abs(p.center[0])
            breaks.update(cx-p.drop_radius*math.cos(math.pi*i/(2*arc_segments))
                          for i in range(arc_segments+1)
                          if core < cx-p.drop_radius*math.cos(math.pi*i/(2*arc_segments)) < xmax)
        xs = sorted(breaks)
        for x0, x1 in zip(xs, xs[1:]):
            active = [p for p in pockets if (x0+x1)/2 > abs(p.center[0])-p.drop_radius]
            lower0 = lower1 = -ymax
            for p in active:
                ys = [math.sqrt(max(0., p.drop_radius**2-max(0.,abs(p.center[0])-x)**2))
                      for x in (x0,x1)]
                upper0,upper1 = p.center[1]-ys[0],p.center[1]-ys[1]
                if p.kind == 'corner' and p.center[1] < 0:
                    upper0=upper1=-ymax
                if upper0 > lower0+1e-10 or upper1 > lower1+1e-10:
                    poly=[(sign*x0,lower0),(sign*x1,lower1),(sign*x1,upper1),(sign*x0,upper0)]
                    polygons.append(poly if sign > 0 else poly[::-1])
                lower0,lower1=p.center[1]+ys[0],p.center[1]+ys[1]
                if p.kind == 'corner' and p.center[1] > 0:
                    lower0=lower1=ymax
            if lower0 < ymax-1e-10 or lower1 < ymax-1e-10:
                poly=[(sign*x0,lower0),(sign*x1,lower1),(sign*x1,ymax),(sign*x0,ymax)]
                polygons.append(poly if sign > 0 else poly[::-1])
    return polygons


def jaw_sections() -> list[tuple[str, list[list[tuple[float, float, float]]]]]:
    """Swept quarter-round rubber profiles, one convex wedge per 5 degrees."""
    sections = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            # Each end has one side-rail and one end-rail jaw.
            for axis in ('side', 'end'):
                rings = []
                for theta in np.linspace(math.pi, math.pi/2, 19):
                    ring = []
                    for u, z in CUSHION_PROFILE:
                        a = HALF_WIDTH + CORNER_JAW_RADIUS + (CORNER_JAW_RADIUS-min(u,.92*CORNER_JAW_RADIUS))*math.cos(theta)
                        b = HALF_LENGTH - CORNER_RUNOUT + (CORNER_JAW_RADIUS-min(u,.92*CORNER_JAW_RADIUS))*math.sin(theta)
                        if axis == 'end':
                            a, b = (HALF_WIDTH - CORNER_RUNOUT
                                    + (CORNER_JAW_RADIUS-min(u,.92*CORNER_JAW_RADIUS))*math.sin(theta),
                                    HALF_LENGTH + CORNER_JAW_RADIUS
                                    + (CORNER_JAW_RADIUS-min(u,.92*CORNER_JAW_RADIUS))*math.cos(theta))
                        ring.append((sx*a, sy*b, SURFACE_Z+z))
                    rings.append(ring)
                sections.append((f'corner_{sx}_{sy}_{axis}', rings))
        for sy in (-1, 1):
            rings = []
            for theta in np.linspace(math.pi, 1.5*math.pi, 19):
                rings.append([(sx*(HALF_WIDTH+MIDDLE_JAW_RADIUS+(MIDDLE_JAW_RADIUS-min(u,.92*MIDDLE_JAW_RADIUS))*math.cos(theta)),
                               sy*(MIDDLE_RUNOUT+(MIDDLE_JAW_RADIUS-min(u,.92*MIDDLE_JAW_RADIUS))*math.sin(theta)),
                               SURFACE_Z+z) for u, z in CUSHION_PROFILE])
            sections.append((f'middle_{sx}_{sy}', rings))
    return sections
