"""Cue geometry, projected contact points and static-table clearance."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import math
import numbers

import mujoco
import numpy as np

from snooker_env.table_geometry import BALL_RADIUS


@dataclass(frozen=True)
class CueStroke:
    direction: np.ndarray
    quaternion: np.ndarray
    initial_body_position: np.ndarray
    elevation_degrees: float
    backoff: float
    follow_through: float

    @property
    def travel(self) -> float:
        return self.backoff + self.follow_through


def cue_shaft_geom_names(model: mujoco.MjModel) -> tuple[str, ...]:
    """Named shaft capsules, including optional taper segments on the cue body."""
    body = model.body('cue_body').id
    first = int(model.body_geomadr[body])
    names = (model.geom(g).name for g in range(first, first + int(model.body_geomnum[body])))
    return tuple(name for name in names if name and (name == 'cue_shaft' or name.startswith('cue_shaft_')))


class CueClearanceProbe:
    """Reusable private collision workspace for a model with fixed geometry.

    Own one per simulator; do not share it between simultaneous planning calls.
    Recreate it after changing the source model's geometry or collision settings.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.source_model = model
        self.model = copy.copy(model)
        # The compiled BVH omits the normally non-colliding shaft. Changing
        # geom masks cannot add it back. Enumerate geometry pairs in this
        # private probe so the enabled shaft is actually tested against the
        # table, without altering the simulation model or its contact masks.
        self.model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        self.data = mujoco.MjData(self.model)


def validate_cue_offset(u: float, v: float) -> tuple[float, float]:
    """Normalized Cartesian contact coordinates in the open ball-radius disk."""
    for value in (u, v):
        if (isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))):
            raise ValueError('Cue offsets must be finite real scalars')
    u, v = float(u), float(v)
    if math.hypot(u, v) >= 1:
        raise ValueError('Cue offsets must satisfy u*u + v*v < 1')
    return u, v


def projected_ball_contact(ball_center: np.ndarray, heading: np.ndarray,
                           cue_offset: tuple[float, float], *,
                           ball_radius: float = BALL_RADIUS) -> tuple[np.ndarray, np.ndarray]:
    """Return a fixed surface point and outward normal, independent of elevation.

    Heading defines a horizontal view. Positive u is right in that view;
    positive v is world up. Both offsets are fractions of the ball radius.
    """
    center = np.asarray(ball_center, dtype=np.float64)
    xy = np.asarray(heading, dtype=np.float64)[:2]
    if (center.shape != (3,) or xy.shape != (2,) or not np.isfinite(center).all()
            or not np.isfinite(xy).all() or np.linalg.norm(xy) < 1e-12
            or not np.isfinite(ball_radius) or ball_radius <= 0):
        raise ValueError('Finite ball center, nonzero heading and positive radius required')
    u, v = validate_cue_offset(*cue_offset)
    xy = xy / np.linalg.norm(xy)
    forward = np.array([*xy, 0.], dtype=np.float64)
    right = np.array([xy[1], -xy[0], 0.], dtype=np.float64)
    normal = u*right + np.array([0., 0., v]) - math.sqrt(max(0., 1-u*u-v*v))*forward
    return center + ball_radius*normal, normal


def prepare_cue_clearance_probe(model: mujoco.MjModel, data: mujoco.MjData,
                                workspace: CueClearanceProbe) -> None:
    """Copy state and enable cue-versus-static-table collision checks only."""
    if workspace.source_model is not model:
        raise ValueError('Clearance probe belongs to a different model.')
    probe_model = workspace.model
    workspace.data.qpos[:] = data.qpos
    # Independent collision bits avoid ball/robot interactions during probing.
    for gid in range(model.ngeom):
        static = (int(model.body_weldid[model.geom_bodyid[gid]]) == 0
                  and bool(model.geom_contype[gid] or model.geom_conaffinity[gid]))
        probe_model.geom_contype[gid] = 1 if static else 0
        probe_model.geom_conaffinity[gid] = 2 if static else 0
    for name in ('cue_tip', *cue_shaft_geom_names(model)):
        gid = model.geom(name).id
        probe_model.geom_contype[gid] = 2
        probe_model.geom_conaffinity[gid] = 1
        probe_model.geom_margin[gid] = max(float(model.geom_margin[gid]), 0.001)


def plan_center_ball_stroke(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ball_center: np.ndarray,
    heading: np.ndarray,
    *,
    backoff: float = 0.10,
    follow_through: float = 0.05,
    max_elevation_degrees: float = 75.0,
    clearance_probe: CueClearanceProbe | None = None,
    ball_radius: float = BALL_RADIUS,
) -> CueStroke:
    """Choose the lowest sampled elevation clearing a swept cue capsule.

    The cue axis passes through the ball center (zero intentional spin offset).
    A raised butt means a downward stroke direction. Backoff is measured along
    that axis from first surface contact, including the tip capsule radius.
    Probe copies include the normally non-colliding shaft. Balls are excluded
    from table clearance tests; their shot-path obstruction is a separate task.
    """
    center = np.asarray(ball_center, dtype=np.float64)
    xy = np.asarray(heading, dtype=np.float64)[:2]
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(xy).all() or np.linalg.norm(xy) < 1e-12:
        raise ValueError("Finite ball center and nonzero horizontal heading required.")
    if backoff <= 0 or follow_through < 0 or not 0 <= max_elevation_degrees <= 85:
        raise ValueError("Invalid stroke travel or elevation limit.")
    xy = xy / np.linalg.norm(xy)
    workspace = clearance_probe if clearance_probe is not None else CueClearanceProbe(model)
    prepare_cue_clearance_probe(model, data, workspace)
    probe_model = workspace.model
    probe = workspace.data
    cue_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'cue_free')
    tip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cue_tip')
    shaft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'cue_shaft')
    if min(cue_joint, tip, shaft) < 0:
        raise ValueError('Cue free joint, tip and shaft are required.')
    adr = int(model.jnt_qposadr[cue_joint])
    cue_ids = {tip, *(model.geom(name).id for name in cue_shaft_geom_names(model))}
    tip_end = float(model.geom_pos[tip, 0] + model.geom_size[tip, 1])
    contact_offset = ball_radius + float(model.geom_size[tip, 0]) + tip_end
    yaw = float(np.arctan2(xy[1], xy[0]))

    def geometry(degrees: float) -> tuple[np.ndarray, np.ndarray]:
        pitch = np.deg2rad(degrees)
        direction = np.array([xy[0]*np.cos(pitch), xy[1]*np.cos(pitch), -np.sin(pitch)])
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        cp, sp = np.cos(pitch/2), np.sin(pitch/2)
        return direction, np.array([cy*cp, -sy*sp, cy*sp, sy*cp])

    def clear(direction: np.ndarray, quat: np.ndarray, offset: float) -> bool:
        probe.qpos[adr:adr+3] = center - direction*(contact_offset+offset)
        probe.qpos[adr+3:adr+7] = quat
        mujoco.mj_forward(probe_model, probe)
        return not any(
            (int(c.geom1) in cue_ids or int(c.geom2) in cue_ids) and c.dist < 0.0005
            for c in probe.contact
        )

    approach = np.linspace(backoff, 0.0, max(2, int(np.ceil(backoff/0.002))+1))
    def approach_clear(degrees: float) -> bool:
        direction, quat = geometry(degrees)
        return all(clear(direction, quat, float(offset)) for offset in approach)

    elevation = None
    for degrees in np.arange(0.0, max_elevation_degrees+1e-9, 1.0):
        if approach_clear(float(degrees)):
            elevation = float(degrees)
            break
    if elevation is None:
        raise ValueError('No collision-free center-directed cue approach within elevation limit.')
    if elevation > 0:
        for degrees in np.arange(max(0., elevation-1.), elevation, 0.1):
            if approach_clear(float(degrees)):
                elevation = float(degrees)
                break
    direction, quat = geometry(elevation)
    # A downward follow-through must also stop before entering the cloth/table.
    safe_follow = 0.0
    for distance in np.linspace(0., follow_through, max(2, int(np.ceil(follow_through/0.001))+1)):
        if not clear(direction, quat, -float(distance)):
            break
        safe_follow = float(distance)
    return CueStroke(direction, quat, center-direction*(contact_offset+backoff), elevation, backoff, safe_follow)
