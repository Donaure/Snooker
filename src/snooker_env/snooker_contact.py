"""MuJoCo contact stepping with measured speed-dependent rubber dissipation.

The public split-step/constraint APIs update contact material parameters before
MuJoCo solves impulses. This never substitutes a ball velocity or installs a
process-global callback. The reference experiment measured normal rolling hits;
application to oblique shots and pocket facings is a material-model assumption.
"""
from __future__ import annotations

from collections.abc import Iterable

import mujoco
import numpy as np

from .snooker_parameters import SNOOKER_PHYSICS, SnookerContactProfile


class SnookerContactStepper:
    def __init__(self, model: mujoco.MjModel, ball_geoms: Iterable[int],
                 rubber_geoms: Iterable[int], profile: SnookerContactProfile = SNOOKER_PHYSICS):
        self.model = model
        self.profile = profile
        self.rubber = frozenset(int(g) for g in rubber_geoms)
        self.ball_vadr = {}
        for geom in ball_geoms:
            body = int(model.geom_bodyid[int(geom)])
            joint = int(model.body_jntadr[body])
            if model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_FREE:
                raise ValueError("Snooker balls must have free joints")
            self.ball_vadr[int(geom)] = int(model.jnt_dofadr[joint])
        if model.opt.integrator != mujoco.mjtIntegrator.mjINT_EULER:
            raise ValueError("Calibrated contact stepping requires the Euler integrator")
        self._incident: dict[int, tuple[float, np.ndarray]] = {}
        self._last_time = -float("inf")

    def reset(self) -> None:
        """Call for reset/restore as well as a new shot; no contact state persists."""
        self._incident.clear()
        self._last_time = -float("inf")

    def step(self, data: mujoco.MjData) -> None:
        model = self.model
        if data.time < self._last_time:
            self.reset()
        self._last_time = float(data.time)
        mujoco.mj_step1(model, data)
        changed = False
        seen = set()
        for contact in data.contact:
            if contact.efc_address < 0:
                continue
            g1, g2 = int(contact.geom1), int(contact.geom2)
            ball = g1 if g1 in self.ball_vadr and g2 in self.rubber else (
                g2 if g2 in self.ball_vadr and g1 in self.rubber else None)
            if ball is None:
                continue
            seen.add(ball)
            previous = self._incident.get(ball)
            if previous is None or float(data.time)-previous[0] > .002:
                adr = self.ball_vadr[ball]
                incident = data.qvel[adr:adr+2].copy()
            else:
                incident = previous[1]
            self._incident[ball] = (float(data.time), incident)
            # Latch the incident vector across adjacent jaw facets. Reproject
            # on each local horizontal normal; do not feed decelerating or
            # separating velocities back into a single impact's damping.
            normal = contact.frame[:2]
            length = float(np.linalg.norm(normal))
            if length < .1:
                continue
            speed = abs(float(incident @ normal)) / length
            contact.solref[1] = -self.profile.rubber_damping(speed)
            changed = True
        for ball in self._incident.keys() - seen:
            if float(data.time)-self._incident[ball][0] > .002:
                del self._incident[ball]
        if changed:
            # mj_step1 already computed KBIP/reference terms from the static
            # XML. Rebuild them through public APIs after changing solref.
            mujoco.mj_makeConstraint(model, data)
            mujoco.mj_island(model, data)
            mujoco.mj_projectConstraint(model, data)
            mujoco.mj_referenceConstraint(model, data)
        mujoco.mj_step2(model, data)
