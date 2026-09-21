"""Plan table-clearing strokes with continuous clearance from other balls."""
from __future__ import annotations

import mujoco
import numpy as np

from snooker_env.cue_stroke import (
    CueClearanceProbe, CueStroke, cue_shaft_geom_names, plan_center_ball_stroke,
    prepare_cue_clearance_probe, projected_ball_contact,
)
from snooker_env.table_geometry import BALL_RADIUS

MULTIBALL_STROKE_VERSION = 'multiball-swept-cue-v2'


class StrokeInfeasibleError(ValueError):
    """No stroke satisfies the configured contact and clearance limits."""


def swept_ball_clearance(model: mujoco.MjModel, stroke: CueStroke,
                         other_ball_centers: np.ndarray, *,
                         ball_radius: float = BALL_RADIUS,
                         geom_names: tuple[str, ...] | None = None) -> float:
    """Exact capsule/sphere clearance over axial translation, including endpoints.

    Other balls are stationary obstacles at planning time. The cue ball is
    deliberately absent; contact with it is the intended strike.
    """
    centers = np.asarray(other_ball_centers, dtype=np.float64).reshape(-1, 3)
    if geom_names is None:
        geom_names = ('cue_tip', *cue_shaft_geom_names(model))
    minimum = float('inf')
    for name in geom_names:
        gid = model.geom(name).id
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_CAPSULE:
            raise ValueError('Swept cue clearance requires capsule cue geometry')
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[gid])
        if not np.allclose(np.abs(rotation.reshape(3, 3)[:, 2]), [1, 0, 0], atol=1e-8):
            raise ValueError('Cue capsule must align with local X')
        if not np.allclose(model.geom_pos[gid, 1:], 0):
            raise ValueError('Cue capsule must be centered on local X')
        half = float(model.geom_size[gid, 1])
        start = stroke.initial_body_position + stroke.direction * (model.geom_pos[gid, 0] - half)
        length = 2 * half + stroke.travel
        delta = centers - start
        projection = np.clip(delta @ stroke.direction, 0, length)
        distances = np.linalg.norm(delta - projection[:, None] * stroke.direction, axis=1)
        if len(distances):
            minimum = min(minimum, float(np.min(distances)) - ball_radius - float(model.geom_size[gid, 0]))
    return minimum


def plan_multiball_stroke(model: mujoco.MjModel, data: mujoco.MjData,
                         ball_center: np.ndarray, heading: np.ndarray, *,
                         other_ball_centers: np.ndarray,
                         backoff: float = .10, follow_through: float = .05,
                         max_elevation_degrees: float = 75.,
                         clearance_probe: CueClearanceProbe | None = None,
                         ball_radius: float = BALL_RADIUS,
                         cue_offset: tuple[float, float] | None = None) -> CueStroke:
    """Prefer lowest elevation, then longest feasible backoff and follow-through.

    Reject when no stroke is found on the 0.1-degree refinement / 1-degree
    coarse grid, with backoffs down to 5 mm and at least 1 mm follow-through.
    Table sweeps retain the shared planner's 2 mm / 1 mm sampling and margin;
    ball sweeps are continuous analytic capsule tests, not point samples.
    A supplied offset fixes the contact point in the horizontal heading's
    projected disk throughout elevation search. None preserves the legacy
    through-ball-center axis used by other pool tasks.
    """
    obstacles = np.asarray(other_ball_centers, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(obstacles).all():
        raise ValueError('Other ball positions must be finite')
    workspace = clearance_probe or CueClearanceProbe(model)
    baseline = None
    contact = normal = None
    if cue_offset is None:
        try:
            baseline = plan_center_ball_stroke(
                model, data, ball_center, heading, backoff=backoff,
                follow_through=follow_through, max_elevation_degrees=max_elevation_degrees,
                clearance_probe=workspace, ball_radius=ball_radius)
        except ValueError as error:
            if 'No collision-free' not in str(error):
                raise
    else:
        contact, normal = projected_ball_contact(ball_center, heading, cue_offset,
                                                 ball_radius=ball_radius)
        if (not np.isfinite([backoff, follow_through, max_elevation_degrees]).all()
                or backoff <= 0 or follow_through < 0 or not 0 <= max_elevation_degrees <= 85):
            raise ValueError('Invalid stroke travel or elevation limit')
        prepare_cue_clearance_probe(model, data, workspace)
    minimum_follow = min(.001, follow_through)
    # A clear horizontal baseline already meets the preferred ordering.
    if (baseline is not None and baseline.elevation_degrees == 0
            and baseline.follow_through >= minimum_follow
            and swept_ball_clearance(model, baseline, obstacles, ball_radius=ball_radius) >= .0005):
        return baseline
    probe_model, probe = workspace.model, workspace.data
    center = np.asarray(ball_center, dtype=np.float64)
    xy = np.asarray(heading, dtype=np.float64)[:2]
    xy = xy / np.linalg.norm(xy)
    yaw = float(np.arctan2(xy[1], xy[0]))
    tip = model.geom('cue_tip').id
    shaft_names = cue_shaft_geom_names(model)
    cue_ids = {tip, *(model.geom(name).id for name in shaft_names)}
    adr = int(model.jnt_qposadr[model.joint('cue_free').id])
    contact_offset = ball_radius + model.geom_size[tip, 0] + model.geom_pos[tip, 0] + model.geom_size[tip, 1]
    tip_radius = float(model.geom_size[tip, 0])
    tip_end = float(model.geom_pos[tip, 0] + model.geom_size[tip, 1])
    backoffs = sorted({backoff, *[b for b in (.075, .05, .025, .01, .005) if b <= backoff]}, reverse=True)

    def candidate(degrees: float) -> CueStroke | None:
        pitch = np.deg2rad(degrees)
        direction = np.array([xy[0]*np.cos(pitch), xy[1]*np.cos(pitch), -np.sin(pitch)])
        cy, sy, cp, sp = np.cos(yaw/2), np.sin(yaw/2), np.cos(pitch/2), np.sin(pitch/2)
        quat = np.array([cy*cp, -sy*sp, cy*sp, sy*cp])
        if contact is None:
            contact_body = center - direction*contact_offset
        else:
            # The front spherical cap touches p when its centre is p + r*n.
            # n·d < 0 also makes the front endpoint the closest capsule point
            # and ensures the cue approaches this point from outside the ball.
            if float(normal @ direction) >= -1e-8:
                return None
            contact_body = contact + tip_radius*normal - tip_end*direction

        def table_clear(offset: float) -> bool:
            probe.qpos[adr:adr+3] = contact_body - direction*offset
            probe.qpos[adr+3:adr+7] = quat
            mujoco.mj_forward(probe_model, probe)
            return not any((int(c.geom1) in cue_ids or int(c.geom2) in cue_ids)
                           and c.dist < .0005 for c in probe.contact)

        for distance in backoffs:
            initial = contact_body - direction*distance
            if contact is not None:
                approach = CueStroke(direction, quat, initial, degrees, distance, 0.)
                if swept_ball_clearance(model, approach, center[None, :], ball_radius=ball_radius,
                                        geom_names=shaft_names) < .0005:
                    continue
            stroke = CueStroke(direction, quat, initial, degrees, distance, minimum_follow)
            if swept_ball_clearance(model, stroke, obstacles, ball_radius=ball_radius) < .0005:
                continue
            if not all(table_clear(float(offset)) for offset in np.linspace(distance, -minimum_follow, max(2, int(np.ceil((distance+minimum_follow)/.002))+1))):
                continue
            safe_follow = minimum_follow
            for follow in np.linspace(minimum_follow, follow_through, max(2, int(np.ceil((follow_through-minimum_follow)/.001))+1)):
                trial = CueStroke(direction, quat, initial, degrees, distance, float(follow))
                if swept_ball_clearance(model, trial, obstacles, ball_radius=ball_radius) < .0005 or not table_clear(-float(follow)):
                    break
                safe_follow = float(follow)
            return CueStroke(direction, quat, initial, degrees, distance, safe_follow)
        return None

    for degrees in sorted({*np.arange(0., max_elevation_degrees + 1e-9, 1.), max_elevation_degrees}):
        stroke = candidate(float(degrees))
        if stroke is not None:
            for refined in np.arange(max(0., degrees-1.), degrees, .1):
                refined_stroke = candidate(float(refined))
                if refined_stroke is not None:
                    return refined_stroke
            return stroke
    raise StrokeInfeasibleError('No collision-free cue path within elevation/backoff limits')
