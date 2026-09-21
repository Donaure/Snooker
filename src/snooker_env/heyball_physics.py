"""Full-table MuJoCo execution for Heyball, with per-step rule events."""
from __future__ import annotations

import copy
from typing import Callable, Mapping

import mujoco
import numpy as np

from .cue_stroke import CueClearanceProbe
from .heyball_rules import ShotEvents
from .multiball_cue_stroke import plan_multiball_stroke
from .scene import load_model, project_root
from .table_geometry import (
    BALL_CENTER_Z, BALL_RADIUS, POCKET_ENTRY_Z, TABLE_LENGTH,
    TABLE_SURFACE_Z, cloth_geom_ids, cushion_geom_ids,
)

BREAK_POSITION = (0.0, -0.95, BALL_CENTER_Z)
FOOT_SPOT = (0.0, TABLE_LENGTH / 4, BALL_CENTER_Z)
HEAD_STRING_Y = -TABLE_LENGTH / 4


def rack_positions(rng: np.random.Generator) -> dict[int, tuple[float, float, float]]:
    """Random triangle facing -Y, with 8 at row 3's center."""
    solid = int(rng.integers(1, 8))
    stripe = int(rng.integers(9, 16))
    corners = [solid, stripe]
    rng.shuffle(corners)
    order = [int(n) for n in rng.permutation([n for n in range(1, 16)
                                           if n not in (8, solid, stripe)])]
    fixed = {(2, 1): 8, (4, 0): corners[0], (4, 4): corners[1]}
    positions = {0: BREAK_POSITION}
    diameter = 2 * BALL_RADIUS + 0.00001
    for row in range(5):
        for column in range(row + 1):
            number = fixed[(row, column)] if (row, column) in fixed else order.pop()
            positions[number] = ((column - row / 2) * diameter,
                                 FOOT_SPOT[1] + row * np.sqrt(3) / 2 * diameter,
                                 BALL_CENTER_Z)
    return positions


class SimulationError(RuntimeError):
    """Physics did not settle reliably; the match must not adjudicate it."""


class HeyballPhysics:
    """Preserve source contact parameters and solve elevation internally.

    A failed rollout restores the pre-shot state. Pocketed/off-table balls are
    removed immediately, but every remaining ball rolls to rest after a scratch.
    """

    scene_file = "heyball_scene.xml"
    ball_count = 16
    ball_radius = BALL_RADIUS
    ball_center_z = BALL_CENTER_Z
    surface_z = TABLE_SURFACE_Z
    pocket_entry_z = POCKET_ENTRY_Z
    exit_half_extents = (.90, 1.50)
    nominal_bounds: tuple[float, float, float, float] | None = None
    cushion_name_prefix: str | None = None
    cloth_name_prefix: str | None = None

    def __init__(self, *, max_simulation_seconds: float = 60.0,
                 frame_callback: Callable[[HeyballPhysics], None] | None = None,
                 frame_interval: float = 1 / 30) -> None:
        if not np.isfinite(max_simulation_seconds) or max_simulation_seconds <= 0:
            raise ValueError("Simulation limit must be finite and positive")
        self.max_simulation_seconds = float(max_simulation_seconds)
        if not np.isfinite(frame_interval) or frame_interval <= 0:
            raise ValueError("Frame interval must be finite and positive")
        self.frame_callback = frame_callback
        self.frame_interval = float(frame_interval)
        self.model = load_model(project_root() / "models" / self.scene_file)
        # MuJoCo compiles ngravcomp as a fast-path flag. The parked MJCF balls
        # declare gravcomp=1 so later mask changes really enable compensation;
        # merely editing body_gravcomp on an ngravcomp=0 model is ineffective.
        if self.model.ngravcomp == 0:
            raise ValueError("Parked balls must declare gravity compensation in MJCF")
        self.data = mujoco.MjData(self.model)
        self.names = {n: "cue_ball" if n == 0 else f"object_ball_{n}" for n in range(self.ball_count)}
        self.bodies = np.array([self.model.body(self.names[n]).id for n in range(self.ball_count)])
        self.geoms = np.array([self.model.geom(self.names[n] + "_geom").id for n in range(self.ball_count)])
        joints = [self.model.joint(self.names[n] + "_free").id for n in range(self.ball_count)]
        self.qadr = self.model.jnt_qposadr[joints].copy()
        self.vadr = self.model.jnt_dofadr[joints].copy()
        self.geom_to_ball = {int(g): n for n, g in enumerate(self.geoms)}
        self.rails = set(cushion_geom_ids(self.model))
        if self.cushion_name_prefix is not None:
            self.rails.update(g for g in range(self.model.ngeom)
                              if (self.model.geom(g).name or "").startswith(self.cushion_name_prefix))
        self.cloth = set(cloth_geom_ids(self.model))
        if self.cloth_name_prefix is not None:
            self.cloth.update(g for g in range(self.model.ngeom)
                              if (self.model.geom(g).name or "").startswith(self.cloth_name_prefix))
        joint = self.model.joint("cue_free").id
        self.cue_qadr = int(self.model.jnt_qposadr[joint])
        self.cue_vadr = int(self.model.jnt_dofadr[joint])
        self.tip = self.model.geom("cue_tip").id
        self.model.body_gravcomp[self.model.body("cue_body").id] = 1
        self.probe = CueClearanceProbe(self.model)
        self.placement_data = mujoco.MjData(self.model)
        self.active: set[int] = set()
        mujoco.mj_forward(self.model, self.data)
        noses = [g for g in self.rails if self.model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
        half_width = max((abs(self.data.geom_xpos[g, 0]) - self.model.geom_size[g, 0] for g in noses), default=0.)
        half_length = max((abs(self.data.geom_xpos[g, 1]) - self.model.geom_size[g, 0] for g in noses), default=0.)
        self.playing_bounds = self.nominal_bounds or (-float(half_width), float(half_width),
                               -float(half_length), float(half_length))
        sites = [i for i in range(self.model.nsite)
                 if (self.model.site(i).name or "").startswith("pocket_")]
        self.pockets = {self.model.site(i).name: tuple(self.data.site_xpos[i]) for i in sites}
        self.pocket_xy = np.array([p[:2] for p in self.pockets.values()])
        self.pocket_radii = self.model.site_size[sites, 0] + .03

    def _cue(self, position: np.ndarray | tuple = (0., 0., 6.),
             quaternion: np.ndarray | tuple = (1., 0., 0., 0.),
             velocity: np.ndarray | tuple = (0., 0., 0.)) -> None:
        q, v = self.cue_qadr, self.cue_vadr
        self.data.qpos[q:q + 3] = position
        self.data.qpos[q + 3:q + 7] = quaternion
        self.data.qvel[v:v + 3] = velocity
        self.data.qvel[v + 3:v + 6] = 0

    def _place(self, number: int, xyz: tuple | np.ndarray) -> None:
        q, v = self.qadr[number], self.vadr[number]
        self.data.qpos[q:q + 3] = xyz
        self.data.qpos[q + 3:q + 7] = (1, 0, 0, 0)
        self.data.qvel[v:v + 6] = 0

    def _remove(self, number: int) -> None:
        self.active.discard(number)
        self.model.geom_contype[self.geoms[number]] = 0
        self.model.geom_conaffinity[self.geoms[number]] = 0
        self.model.body_gravcomp[self.bodies[number]] = 1
        self._place(number, (3 + number * .15, 0, 4))

    def reset(self, positions: Mapping[int, tuple[float, float, float]]) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.active = set(positions)
        for number in range(self.ball_count):
            if number in positions:
                self.model.geom_contype[self.geoms[number]] = 1
                self.model.geom_conaffinity[self.geoms[number]] = 1
                self.model.body_gravcomp[self.bodies[number]] = 0
                self._place(number, positions[number])
            else:
                self._remove(number)
        self._cue()
        mujoco.mj_forward(self.model, self.data)

    def positions(self) -> dict[int, tuple[float, float, float]]:
        return {n: tuple(float(x) for x in self.data.qpos[self.qadr[n]:self.qadr[n] + 3])
                for n in sorted(self.active)}

    def _cloth_supports(self, x: float, y: float) -> bool:
        """Whether a ball centre lies over cloth; scenes may supply curved cutouts."""
        point = np.array([x, y, self.surface_z])
        for geom in self.cloth:
            if self.model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            local = self.data.geom_xmat[geom].reshape(3, 3).T @ (point - self.data.geom_xpos[geom])
            if (np.all(np.abs(local[:2]) <= self.model.geom_size[geom, :2])
                    and abs(local[2] - self.model.geom_size[geom, 2]) < .001):
                return True
        return False

    def placement_error(self, x: float, y: float) -> str | None:
        if not np.isfinite([x, y]).all():
            return "Choose finite table coordinates."
        xmin, xmax, ymin, ymax = self.playing_bounds
        if not (xmin + self.ball_radius <= x <= xmax - self.ball_radius
                and ymin + self.ball_radius <= y <= ymax - self.ball_radius):
            return "The whole cue ball must be inside the playing surface."
        for n, p in self.positions().items():
            if n and np.linalg.norm(np.array([x, y]) - np.array(p[:2])) < 2 * self.ball_radius + 1e-6:
                return "The cue ball overlaps another ball."
        if not self._cloth_supports(x, y):
            return "Choose cloth, outside the pocket openings."
        probe = self.placement_data
        probe.qpos[:] = self.data.qpos
        probe.qvel[:] = 0
        q = self.qadr[0]
        probe.qpos[q:q + 3] = (x, y, self.ball_center_z)
        # The cue ball may currently be removed following a scratch.
        gid = self.geoms[0]
        old = (self.model.geom_contype[gid], self.model.geom_conaffinity[gid])
        self.model.geom_contype[gid] = self.model.geom_conaffinity[gid] = 1
        try:
            mujoco.mj_forward(self.model, probe)
            for contact in probe.contact:
                pair = {int(contact.geom1), int(contact.geom2)}
                if gid in pair and not (pair & self.cloth) and contact.dist < -1e-6:
                    return "The cue ball overlaps a cushion or pocket jaw."
        finally:
            self.model.geom_contype[gid], self.model.geom_conaffinity[gid] = old
        return None

    def place_cue_ball(self, x: float, y: float) -> None:
        error = self.placement_error(x, y)
        if error:
            raise ValueError(error)
        self.active.add(0)
        self.model.geom_contype[self.geoms[0]] = 1
        self.model.geom_conaffinity[self.geoms[0]] = 1
        self.model.body_gravcomp[self.bodies[0]] = 0
        self._place(0, (x, y, self.ball_center_z))
        mujoco.mj_forward(self.model, self.data)

    def execute(self, angle: float, speed: float) -> ShotEvents:
        return self._execute_with_offset(angle, speed)

    def _execute_with_offset(self, angle: float, speed: float, *,
                             cue_offset: tuple[float, float] | None = None) -> ShotEvents:
        """Run a shot transaction; subclasses may supply a fixed contact point."""
        if 0 not in self.active:
            raise ValueError("Place the cue ball before executing a shot")
        if not np.isfinite([angle, speed]).all() or speed <= 0:
            raise ValueError("Finite heading and positive cue speed required")
        saved_data = copy.copy(self.data)
        saved_active = self.active.copy()
        try:
            return self._execute(angle, speed, cue_offset=cue_offset)
        except Exception:
            self.reset({n: tuple(saved_data.qpos[self.qadr[n]:self.qadr[n] + 3]) for n in saved_active})
            mujoco.mj_copyData(self.data, self.model, saved_data)
            raise

    def _ball_exit_kind(self, position: np.ndarray, number: int | None = None) -> str | None:
        """Classify a completed drop/exit, leaving pocket-mouth contacts to physics."""
        if position[2] < self.pocket_entry_z:
            if np.any(np.linalg.norm(self.pocket_xy - position[:2], axis=1) <= self.pocket_radii):
                return "pocket"
            return "off_table"
        if (abs(position[0]) > self.exit_half_extents[0] + self.ball_radius
                or abs(position[1]) > self.exit_half_extents[1] + self.ball_radius):
            return "off_table"
        return None

    def _step(self) -> None:
        """Advance native physics; specialised scenes may configure contacts."""
        mujoco.mj_step(self.model, self.data)

    def _execute(self, angle: float, speed: float, *,
                 cue_offset: tuple[float, float] | None = None) -> ShotEvents:
        positions = self.positions()
        stroke = plan_multiball_stroke(
            self.model, self.data, np.array(positions[0]),
            np.array([np.cos(angle), np.sin(angle)]),
            other_ball_centers=np.array([p for n, p in positions.items() if n]),
            clearance_probe=self.probe, ball_radius=self.ball_radius, cue_offset=cue_offset)
        start = float(self.data.time)
        duration = stroke.travel / speed
        first: tuple[int, ...] = ()
        rail_after = False
        pocketed: list[int] = []
        off: list[int] = []
        fouls: set[str] = set()
        previous: set[tuple[int, int]] = set()
        tip_has_hit = False
        tip_done = False
        stopped_since = None
        dt = float(self.model.opt.timestep)
        steps = int(np.ceil(self.max_simulation_seconds / dt))
        velocity_indices = self.vadr[:, None] + np.arange(6)
        frame_stride = max(1, round(self.frame_interval / dt))
        for step in range(steps):
            elapsed = float(self.data.time) - start
            if elapsed < duration and not tip_done and 0 in self.active:
                self._cue(stroke.initial_body_position + stroke.direction * speed * elapsed,
                          stroke.quaternion, stroke.direction * speed)
            else:
                self._cue()
            self._step()
            pairs = {tuple(sorted((int(c.geom1), int(c.geom2)))) for c in self.data.contact
                     if c.dist <= 0}
            new_pairs = pairs - previous
            cue_contacts = []
            tip_touching = False
            for g1, g2 in pairs:
                b1, b2 = self.geom_to_ball.get(g1), self.geom_to_ball.get(g2)
                if self.tip in (g1, g2):
                    ball = b2 if g1 == self.tip else b1
                    if ball == 0:
                        tip_touching = True
                    elif ball is not None:
                        fouls.add("cue_touched_object_ball")
                if b1 is not None and b2 is not None and 0 in (b1, b2):
                    cue_contacts.append(b2 if b1 == 0 else b1)
            # Only a NEW rail contact in a LATER step satisfies the rail rule.
            if first and any((g1 in self.rails and g2 in self.geom_to_ball)
                             or (g2 in self.rails and g1 in self.geom_to_ball)
                             for g1, g2 in new_pairs):
                rail_after = True
            if not first and cue_contacts:
                first = tuple(sorted(set(cue_contacts)))
            if tip_touching and cue_contacts:
                fouls.add("push_shot")
            if tip_has_hit and not tip_touching:
                tip_done = True  # Retract after separation; never strike twice.
            tip_has_hit |= tip_touching
            previous = pairs
            # Scan exits after every integration step; do not stop on a scratch.
            for n in tuple(self.active):
                p = self.data.qpos[self.qadr[n]:self.qadr[n] + 3]
                exit_kind = self._ball_exit_kind(p, n)
                if exit_kind is not None:
                    (pocketed if exit_kind == "pocket" else off).append(n)
                    self._remove(n)
            if self.frame_callback is not None and step % frame_stride == 0:
                self.frame_callback(self)
            if step % 100 == 0:
                if (not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all()
                        or np.max(np.abs(self.data.qvel)) > 150):
                    raise SimulationError(
                        f"Unstable match physics at t={float(self.data.time) - start:.6f}s; "
                        f"max_abs_qvel={float(np.max(np.abs(self.data.qvel))):.3f}; "
                        f"timestep={dt:g}")
                velocities = self.data.qvel[velocity_indices[list(self.active)]]
                stopped = (np.all(np.linalg.norm(velocities[:, :3], axis=1) < .01)
                           and np.all(self.ball_radius * np.linalg.norm(velocities[:, 3:], axis=1) < .01))
                if elapsed >= duration and stopped:
                    if stopped_since is None:
                        stopped_since = elapsed
                    elif elapsed - stopped_since >= .20:
                        for n in tuple(self.active):
                            if self.data.qpos[self.qadr[n] + 2] > self.ball_center_z + .01:
                                off.append(n)
                                self._remove(n)
                        for n in self.active:
                            self.data.qvel[self.vadr[n]:self.vadr[n] + 6] = 0
                        self._cue()
                        mujoco.mj_forward(self.model, self.data)
                        return ShotEvents(first, tuple(pocketed), tuple(off), rail_after,
                                          tuple(sorted(fouls)), stroke.elevation_degrees,
                                          float(self.data.time) - start)
                else:
                    stopped_since = None
        raise SimulationError(f"Balls did not settle in {self.max_simulation_seconds:g} simulated seconds")
