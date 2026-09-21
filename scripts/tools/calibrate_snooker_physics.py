"""Fit and audit snooker contact response against published experiments.

All observations come from mj_step, without analytic velocity replacement.
Run in pool; --fit writes a suggestion JSON, never silently changes the scene.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, tostring

import mujoco
import numpy as np

from _bootstrap import add_src_to_path
ROOT = add_src_to_path()
from snooker_env.snooker_contact import SnookerContactStepper
from snooker_env.snooker_table_assets import CUSHION_PROFILE, jaw_sections
from snooker_env.snooker_geometry import BALL_RADIUS, SURFACE_Z
from snooker_env.snooker_parameters import SNOOKER_PHYSICS, SnookerContactProfile, as_mjcf

G = 9.81
# Five independently recorded oblique shots; Mathavan et al. (2014), Tables 1/2.
# v_in, cut_deg, cue_v_after_slip, object_v_after_slip, cue_angle, object_angle.
OBLIQUE_EXPERIMENTS = [
    (1.539, 33.83, .816, .836, 35.96, 33.83),
    (1.032, 26.36, .520, .629, 33.20, 26.36),
    (1.364, 40.52, .925, .700, 30.50, 40.51),
    (1.731, 46.50, 1.275, .787, 27.97, 46.50),
    (.942, 18.05, .365, .581, 29.86, 18.05),
]


def cushion_reference(speed: float) -> float:
    """2009 Fig. 9 experimental regression, valid for rolling 0.28–3.5 m/s."""
    return -.0877 * speed**2 + 1.131 * speed - .0953


def fixture(profile: SnookerContactProfile, family: str = "cloth", *,
            angle: float = 0., isolated: bool = False, timestep: float | None = None):
    root = Element("mujoco", model=f"snooker_calibration_{family}")
    SubElement(root, "compiler", angle="radian")
    SubElement(root, "option", timestep=str(timestep or profile.timestep),
               gravity="0 0 0" if isolated else "0 0 -9.81", integrator="Euler",
               cone="elliptic", solver="Newton", iterations="80", tolerance="1e-8")
    defaults = SubElement(root, "default")
    for name, attrs in profile.contact_defaults().items():
        SubElement(SubElement(defaults, "default", {"class": name}), "geom", **attrs)
    world = SubElement(root, "worldbody")
    if not isolated:
        SubElement(world, "geom", name="cloth", type="plane", size="20 20 .1",
                   **{"class": "snooker_cloth"})
    if family == "cushion" or family.startswith("jaw"):
        section = np.asarray(CUSHION_PROFILE)
        asset = SubElement(root, "asset")
        if family == "cushion":
            segments = [[(x, y, z) for y in (-2., 2.) for x, z in section]]
        else:
            name = "corner_1_1_side" if family == "jaw_corner" else "middle_1_1"
            rings = np.asarray(dict(jaw_sections())[name])
            nose = rings[[0, len(rings)//2, -1], 0, :2]
            center = np.linalg.solve(2*(nose[1:]-nose[0]),
                                     np.sum(nose[1:]**2, axis=1)-np.sum(nose[0]**2))
            radius = float(np.linalg.norm(center-nose[1]))
            outward = (center-nose[1])/radius
            tangent = np.array([-outward[1], outward[0]])
            local = np.empty_like(rings)
            local[:, :, 0] = (rings[:, :, :2]-center) @ outward + radius
            local[:, :, 1] = (rings[:, :, :2]-center) @ tangent
            local[:, :, 2] = rings[:, :, 2]-SURFACE_Z
            segments = [np.concatenate((a, b)) for a, b in zip(local, local[1:])]
        for i, vertices in enumerate(segments):
            name = f"rubber_{i}"
            SubElement(asset, "mesh", name=name, vertex=as_mjcf(np.asarray(vertices).ravel()))
            SubElement(world, "geom", name=name, type="mesh", mesh=name,
                       **{"class": "snooker_cushion" if family == "cushion" else "snooker_jaw"})
    count = 2 if family == "ball" else 1
    for n in range(count):
        body = SubElement(world, "body", name=f"ball_{n}", pos=f"0 0 {BALL_RADIUS}")
        SubElement(body, "freejoint")
        SubElement(body, "geom", name=f"ball_geom_{n}", type="sphere", size=str(BALL_RADIUS),
                   **{"class": "snooker_ball"})
    model = mujoco.MjModel.from_xml_string(tostring(root, encoding="unicode"))
    data = mujoco.MjData(model)
    return model, data


def settle(model, data) -> None:
    mujoco.mj_forward(model, data)
    mujoco.mj_step(model, data, nstep=round(.02 / model.opt.timestep))
    data.time = 0.


def cloth_measure(profile: SnookerContactProfile, speed: float, heading: float = 0.,
                  mode: str = "roll", timestep: float | None = None) -> dict:
    model, data = fixture(profile, timestep=timestep)
    settle(model, data)
    direction = np.array([math.cos(heading), math.sin(heading)])
    if mode == "spin":
        data.qvel[5] = speed
    else:
        data.qvel[:2] = speed * direction
        if mode == "roll":
            data.qvel[3:5] = speed / BALL_RADIUS * np.array([-direction[1], direction[0]])
    rows = []
    duration = .25 if mode in ("roll", "spin") else max(.3, speed / (2. * G * profile.sliding_friction))
    stride = 10
    for _ in range(math.ceil(duration / (model.opt.timestep * stride))):
        mujoco.mj_step(model, data, nstep=stride)
        velocity = data.qvel[:2]
        spin = data.xmat[model.body("ball_0").id].reshape(3, 3) @ data.qvel[3:6]
        slip = velocity + BALL_RADIUS * np.array([-spin[1], spin[0]])
        rows.append((data.time, float(velocity @ direction), float(np.linalg.norm(slip)), float(spin[2])))
    arr = np.asarray(rows)
    if mode == "spin":
        selected = arr[(arr[:, 0] > .01) & (arr[:, 3] > .1)]
        decel = -np.polyfit(selected[:, 0], selected[:, 3], 1)[0]
        return {"mode": mode, "initial_omega_rad_s": speed, "angular_deceleration_rad_s2": float(decel)}
    if mode == "roll":
        selected = arr[(arr[:, 0] > .02) & (arr[:, 1] > .05)]
        decel = -np.polyfit(selected[:, 0], selected[:, 1], 1)[0]
        return {"mode": mode, "initial_speed_m_s": speed, "heading_deg": math.degrees(heading),
                "deceleration_m_s2": float(decel), "slip_m_s": float(np.max(selected[:, 2]))}
    selected = arr[(arr[:, 0] > .005) & (arr[:, 2] > .05 * speed)]
    decel = -np.polyfit(selected[:, 0], selected[:, 1], 1)[0]
    transition = arr[np.flatnonzero(arr[:, 2] < .02)[0]]
    return {"mode": mode, "initial_speed_m_s": speed, "heading_deg": math.degrees(heading),
            "deceleration_m_s2": float(decel), "roll_transition_s": float(transition[0]),
            "roll_transition_speed_m_s": float(transition[1])}


def impact_measure(profile: SnookerContactProfile, family: str, speed: float,
                   angle_deg: float = 0., rolling: bool = True, *, isolated: bool = False,
                   timestep: float | None = None, sidespin: float = 0., dynamic: bool = False) -> dict:
    model, data = fixture(profile, family, isolated=isolated, timestep=timestep)
    theta = math.radians(angle_deg)
    incoming = np.array([math.cos(theta), math.sin(theta), 0.]) * speed
    gap = .0005
    if family == "ball":
        # Line of centres +X; incoming at angle theta. Short run-in eliminates
        # the long pre-impact slip that would contaminate the prescribed spin.
        data.qpos[:3] = (-2 * BALL_RADIUS, 0, BALL_RADIUS)
    else:
        # Approach along the local normal to the sharp nose then stand off.
        section = np.asarray(CUSHION_PROFILE)
        nose_z = section[0, 1]
        data.qpos[:3] = (-math.sqrt(BALL_RADIUS**2-(nose_z-BALL_RADIUS)**2), 0, BALL_RADIUS)
    data.qpos[:3] -= gap * incoming / max(speed, .01)
    settle(model, data)
    data.qvel[:3] = incoming
    if rolling:
        data.qvel[3:5] = np.array([-incoming[1], incoming[0]]) / BALL_RADIUS
    data.qvel[5] = sidespin
    ball_gid = model.geom("ball_geom_0").id
    target_gids = ({model.geom("ball_geom_1").id} if family == "ball" else
                   {g for g in range(model.ngeom) if (model.geom(g).name or "").startswith("rubber_")})
    stepper = (SnookerContactStepper(model, [ball_gid], target_gids, profile)
               if dynamic and family != "ball" else None)
    started = False
    last_incoming = data.qvel[:3].copy()
    preimpact = None
    postimpact = None
    contact_time = 0.
    max_penetration = 0.
    maximum_z = float(data.qpos[2])
    energy_initial = kinetic(model, data)
    energy_at_contact = energy_initial
    for _ in range(round(.1 / model.opt.timestep)):
        mujoco.mj_forward(model, data)
        contacts = [c for c in data.contact if ball_gid in (c.geom1, c.geom2)
                    and (int(c.geom1) in target_gids or int(c.geom2) in target_gids) and c.dist <= 0]
        if contacts:
            if not started:
                preimpact = last_incoming.copy()
                energy_at_contact = kinetic(model, data)
            started = True
            contact_time += model.opt.timestep
            max_penetration = max(max_penetration, max(-float(c.dist) for c in contacts))
        elif started:
            postimpact = data.qvel.copy()
            break
        else:
            last_incoming = data.qvel[:3].copy()
        if stepper is None:
            mujoco.mj_step(model, data)
        else:
            stepper.step(data)
        maximum_z = max(maximum_z, float(data.qpos[2]))
    if postimpact is None or preimpact is None:
        raise RuntimeError(f"No completed {family} collision: speed={speed}, angle={angle_deg}")
    if family == "ball":
        ratio = (postimpact[6] - postimpact[0]) / preimpact[0]
        rebound = float(np.linalg.norm(postimpact[6:8]))
    else:
        ratio = -postimpact[0] / preimpact[0]
        rebound = float(np.linalg.norm(postimpact[:2]))
    result = {"family": family, "dynamic_rubber": dynamic, "incident_speed_m_s": float(np.linalg.norm(preimpact[:2])),
              "angle_from_normal_deg": angle_deg, "rolling": rolling, "sidespin_rad_s": sidespin,
              "normal_restitution": float(ratio), "outgoing_speed_m_s": rebound,
              "rebound_angle_deg": math.degrees(math.atan2(abs(postimpact[1]), abs(postimpact[0]))),
              "contact_duration_ms": contact_time*1000,
              "maximum_penetration_mm": max_penetration*1000,
              "energy_ratio": kinetic(model, data) / energy_at_contact,
              "maximum_lift_mm": (maximum_z-BALL_RADIUS)*1000}
    if family != "ball" and angle_deg == 0 and rolling:
        expected = cushion_reference(result["incident_speed_m_s"])
        result.update(reference_speed_m_s=expected, error_m_s=rebound-expected)
    if family == "ball" and not isolated:
        # Record each ball separately at its first post-impact slide→roll
        # transition, matching the measured Tables 1/2, not at an arbitrary time.
        outcomes = {}
        for _ in range(round(.7 / model.opt.timestep)):
            for n in (0, 1):
                v = data.qvel[n*6:n*6+6]
                spin = data.xmat[model.body(f"ball_{n}").id].reshape(3, 3) @ v[3:6]
                slip = np.linalg.norm(v[:2]+BALL_RADIUS*np.array([-spin[1], spin[0]]))
                if n not in outcomes and slip < .02:
                    # World velocity relative to original incoming trajectory.
                    direction = math.atan2(v[1], v[0])
                    outcomes[n] = {"speed_m_s": float(np.linalg.norm(v[:2])),
                                   "angle_deg": abs(math.degrees(direction-theta)),
                                   "time_s": float(data.time)}
            if len(outcomes) == 2:
                break
            mujoco.mj_step(model, data)
        result["post_slip"] = outcomes
    return result


def kinetic(model, data) -> float:
    value = 0.
    for n in range(model.njnt):
        adr = model.jnt_dofadr[n]
        body = model.jnt_bodyid[n]
        v = data.qvel[adr:adr+6]
        value += .5 * model.body_mass[body] * float(v[:3] @ v[:3])
        value += .5 * float(model.body_inertia[body] @ (v[3:]**2))
    return value


def golden_minimum(function, low: float, high: float, steps: int = 22) -> float:
    """Bounded scalar fit; keeps the calibration dependency-free beyond NumPy."""
    ratio = (math.sqrt(5.) - 1.) / 2
    left, right = high-ratio*(high-low), low+ratio*(high-low)
    fleft, fright = function(left), function(right)
    for _ in range(steps):
        if fleft < fright:
            high, right, fright = right, left, fleft
            left = high-ratio*(high-low)
            fleft = function(left)
        else:
            low, left, fleft = left, right, fright
            right = low+ratio*(high-low)
            fright = function(right)
    return (low+high)/2


def fit_profile(initial: SnookerContactProfile) -> tuple[SnookerContactProfile, dict]:
    fitted = initial
    fits = {}
    for field, mode, speed, target in (
        ("rolling_friction", "roll", 1., .125),
        ("sliding_friction", "slide", 1., .212*G),
        ("spinning_friction", "spin", 30., 22.),
    ):
        def residual(coefficient):
            profile = replace(fitted, **{field: coefficient})
            obs = cloth_measure(profile, speed, mode=mode)
            measured = obs["angular_deceleration_rad_s2" if mode == "spin" else "deceleration_m_s2"]
            return (measured-target)/target
        initial_value = getattr(fitted, field)
        value = golden_minimum(lambda x: residual(x)**2, initial_value*.3, initial_value*2.)
        error = residual(value)
        fitted = replace(fitted, **{field: value})
        fits[field] = {"value": value, "relative_residual": error}
        print(f"Fitted {field}: {value:.9g}; relative error {error:.5g}", flush=True)
    def ball_residual(value):
        profile = replace(fitted, ball_solref=(-1e7, -value))
        return impact_measure(profile, "ball", 1.5, rolling=False, isolated=True)["normal_restitution"]-.89
    value = golden_minimum(lambda x: ball_residual(x)**2, 0., 1500.)
    error = ball_residual(value)
    fitted = replace(fitted, ball_solref=(-1e7, -value))
    fits["ball_solref"] = {"value": fitted.ball_solref, "residual": error}
    print(f"Fitted ball_solref: {fitted.ball_solref}; restitution error {error:.5g}", flush=True)
    def cushion_residual(value):
        profile = replace(fitted, cushion_solref=(-1e6, -value))
        return [impact_measure(profile, "cushion", v)["error_m_s"] for v in (.5, 1., 2.)]
    value = golden_minimum(lambda x: sum(e*e for e in cushion_residual(x)), 0., 900.)
    errors = cushion_residual(value)
    fitted = replace(fitted, cushion_solref=(-1e6, -value), jaw_solref=(-1e6, -value))
    fits["cushion_solref"] = {"value": fitted.cushion_solref, "residual_m_s": errors,
                              "fit_speeds_m_s": [.5, 1., 2.]}
    print(f"Fitted cushion_solref: {fitted.cushion_solref}; errors {errors}", flush=True)
    curve = []
    for speed in (.28, .5, .75, 1., 1.5, 2., 2.5, 3., 3.5):
        def residual_at_speed(damping):
            p = replace(fitted, cushion_solref=(-1e6, -damping))
            return impact_measure(p, "cushion", speed)["error_m_s"]
        damping = golden_minimum(lambda x: residual_at_speed(x)**2, 0., 400.)
        curve.append((speed, damping))
        print(f"Rubber curve {speed}: damping {damping}, error {residual_at_speed(damping)}", flush=True)
    fitted = replace(fitted, rubber_damping_curve=tuple(curve))
    fits["rubber_damping_curve"] = curve
    return fitted, fits


def audit(profile: SnookerContactProfile) -> dict:
    cloth = [cloth_measure(profile, speed, heading, mode)
             for mode, speeds in (("roll", (.3, .8, 1.8)), ("slide", (.6, 1.7, 3.)))
             for speed in speeds for heading in (0., math.radians(37), math.pi/2)]
    spin = [cloth_measure(profile, w, mode="spin") for w in (15., 60., 100.)]
    ball = [impact_measure(profile, "ball", v, rolling=False, isolated=True) for v in (.35, .8, 2.5, 3.5)]
    cushions = [impact_measure(profile, "cushion", v, a)
                for v in (.35, .75, 1.5, 2.5, 3.5) for a in (0., 25., 50.)]
    jaws = [impact_measure(profile, family, v, a)
            for family in ("jaw_corner", "jaw_middle")
            for v in (.5, 1.5, 2.5) for a in (0., 20., 40.)]
    dynamic_cushions = [impact_measure(profile, "cushion", v, a, dynamic=True)
                       for v in (.35, .625, 1.25, 1.75, 2.25, 2.75, 3.25)
                       for a in (0., 25., 50.)]
    dynamic_jaws = [impact_measure(profile, family, v, a, dynamic=True)
                   for family in ("jaw_corner", "jaw_middle")
                   for v in (.625, 1.25, 2.25, 3.25) for a in (0., 20., 40.)]
    oblique = []
    for speed, angle, cue_v, obj_v, cue_angle, obj_angle in OBLIQUE_EXPERIMENTS:
        row = impact_measure(profile, "ball", speed, angle)
        row["experiment"] = {"cue_speed_m_s": cue_v, "object_speed_m_s": obj_v,
                             "cue_angle_deg": cue_angle, "object_angle_deg": obj_angle}
        row["errors"] = {
            "cue_speed_m_s": row["post_slip"][0]["speed_m_s"]-cue_v,
            "object_speed_m_s": row["post_slip"][1]["speed_m_s"]-obj_v,
            "cue_angle_deg": row["post_slip"][0]["angle_deg"]-cue_angle,
            "object_angle_deg": row["post_slip"][1]["angle_deg"]-obj_angle,
        }
        oblique.append(row)
    convergence = []
    for family in ("ball", "cushion", "jaw_corner", "jaw_middle"):
        base = impact_measure(profile, family, 1.5, 25., dynamic=family != "ball")
        fine = impact_measure(profile, family, 1.5, 25., timestep=profile.timestep/2, dynamic=family != "ball")
        convergence.append({"family": family, "base": base, "half_timestep": fine,
                            "speed_change_m_s": fine["outgoing_speed_m_s"]-base["outgoing_speed_m_s"],
                            "angle_change_deg": fine["rebound_angle_deg"]-base["rebound_angle_deg"]})
    constant = replace(profile, rubber_damping_curve=((0., -profile.cushion_solref[1]),
                                                       (4., -profile.cushion_solref[1])))
    standard = impact_measure(constant, "cushion", 1.3, 25.)
    split = impact_measure(constant, "cushion", 1.3, 25., dynamic=True)
    equivalence = {"speed_error_m_s": abs(standard["outgoing_speed_m_s"]-split["outgoing_speed_m_s"]),
                   "angle_error_deg": abs(standard["rebound_angle_deg"]-split["rebound_angle_deg"])}
    sidespin = [impact_measure(profile, family, 1.5, 25., sidespin=w, dynamic=True)
                for family in ("cushion", "jaw_corner", "jaw_middle") for w in (-40., 40.)]
    return {"rolling_stop": [rolling_stop(profile, v) for v in (.4, .8, 1.2)],
            "sidespin": sidespin, "static_split_step_equivalence": equivalence, "cloth": cloth, "spin": spin, "isolated_ball_restitution": ball,
            "cushions": cushions, "jaws": jaws,
            "dynamic_cushions": dynamic_cushions, "dynamic_jaws": dynamic_jaws, "published_oblique_holdout": oblique,
            "timestep_convergence": convergence}


def rolling_stop(profile: SnookerContactProfile, speed: float) -> dict:
    model, data = fixture(profile)
    settle(model, data)
    data.qvel[0], data.qvel[4] = speed, speed/BALL_RADIUS
    start = float(data.qpos[0])
    while data.time < 30. and data.qvel[0] > .002:
        mujoco.mj_step(model, data, nstep=100)
    if data.time >= 30.:
        raise RuntimeError("Rolling ball failed to stop in 30 seconds")
    distance = float(data.qpos[0])-start
    ideal = speed**2/(2*.125)
    return {"initial_speed_m_s": speed, "stop_speed_threshold_m_s": .002,
            "time_s": float(data.time), "distance_m": distance,
            "ideal_constant_deceleration_distance_m": ideal,
            "distance_error_fraction": (distance-ideal)/ideal}


def summarize(report: dict) -> dict:
    audit_data = report["audit"]
    roll = [x["deceleration_m_s2"] for x in audit_data["cloth"] if x["mode"] == "roll"]
    slide = [x["deceleration_m_s2"] for x in audit_data["cloth"] if x["mode"] == "slide"]
    spin = [x["angular_deceleration_rad_s2"] for x in audit_data["spin"]]
    restitution = [x["normal_restitution"] for x in audit_data["isolated_ball_restitution"]]
    rebound_errors = [x["error_m_s"] for x in audit_data["dynamic_cushions"]
                      if x["angle_from_normal_deg"] == 0]
    transfer = audit_data["published_oblique_holdout"]
    results = {"roll_deceleration_range_m_s2": [min(roll), max(roll)],
               "sliding_deceleration_range_m_s2": [min(slide), max(slide)],
               "spin_deceleration_range_rad_s2": [min(spin), max(spin)],
               "ball_restitution_range": [min(restitution), max(restitution)],
               "heldout_normal_rebound_max_error_m_s": max(abs(x) for x in rebound_errors),
               "heldout_normal_rebound_rmse_m_s": float(np.sqrt(np.mean(np.square(rebound_errors)))),
               "object_ball_speed_max_error_m_s": max(abs(x["errors"]["object_speed_m_s"]) for x in transfer),
               "object_ball_angle_max_error_deg": max(abs(x["errors"]["object_angle_deg"]) for x in transfer),
               "cue_ball_speed_max_error_m_s": max(abs(x["errors"]["cue_speed_m_s"]) for x in transfer),
               "cue_ball_angle_max_error_deg": max(abs(x["errors"]["cue_angle_deg"]) for x in transfer)}
    return results


def check_report(report: dict) -> None:
    """Engineering regression budgets; none are experimental confidence bounds."""
    a, s = report["audit"], report["summary"]
    assert max(abs(x-.125) for x in s["roll_deceleration_range_m_s2"]) < .003
    assert 1.75 <= min(s["sliding_deceleration_range_m_s2"]) <= max(s["sliding_deceleration_range_m_s2"]) <= 2.4
    assert max(abs(x-22.) for x in s["spin_deceleration_range_rad_s2"]) < .5
    assert max(abs(x-.89) for x in s["ball_restitution_range"]) < .015
    assert s["heldout_normal_rebound_max_error_m_s"] < .05
    for row in a["dynamic_cushions"]+a["dynamic_jaws"]+a["sidespin"]:
        assert np.isfinite(list(v for v in row.values() if isinstance(v, (int, float)))).all()
        assert row["energy_ratio"] <= 1.001, row
        assert row["normal_restitution"] > 0, row
    for row in a["timestep_convergence"]:
        assert abs(row["speed_change_m_s"]) < .02, row
        assert abs(row["angle_change_deg"]) < .5, row
    for row in a["rolling_stop"]:
        assert abs(row["distance_error_fraction"]) < .05, row
    assert a["static_split_step_equivalence"]["speed_error_m_s"] < 1e-9
    assert a["static_split_step_equivalence"]["angle_error_deg"] < 1e-8


def write_plot(report: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a = report["audit"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ax = axes[0, 0]
    x = np.linspace(.28, 3.5, 200)
    ax.plot(x, [cushion_reference(v) for v in x], color="black", label="Published experimental regression")
    rows = [r for r in a["dynamic_cushions"] if r["angle_from_normal_deg"] == 0]
    ax.scatter([r["incident_speed_m_s"] for r in rows], [r["outgoing_speed_m_s"] for r in rows], label="MuJoCo held-out speeds", zorder=4)
    rows = [r for r in a["cushions"] if r["angle_from_normal_deg"] == 0]
    ax.plot([r["incident_speed_m_s"] for r in rows], [r["outgoing_speed_m_s"] for r in rows], "--", color="tab:red", label="Constant material damping")
    ax.set(xlabel="Incident speed (m/s)", ylabel="Immediate rebound speed (m/s)", title="Rolling normal cushion collisions")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    rows = [r for r in a["cloth"] if r["mode"] == "roll"]
    ax.scatter([r["initial_speed_m_s"] for r in rows], [r["deceleration_m_s2"] for r in rows], label="MuJoCo, three headings")
    ax.axhspan(.124, .126, color="tab:green", alpha=.25, label="Published 0.124–0.126")
    ax.set(xlabel="Initial rolling speed (m/s)", ylabel="Rolling deceleration (m/s²)", ylim=(.118, .132), title="Cloth rolling resistance")
    ax.legend(fontsize=8)
    rows = a["published_oblique_holdout"]
    ax = axes[1, 0]
    for ball, color in (("cue", "tab:orange"), ("object", "tab:blue")):
        ax.scatter([r["experiment"][f"{ball}_speed_m_s"] for r in rows],
                   [r["experiment"][f"{ball}_speed_m_s"]+r["errors"][f"{ball}_speed_m_s"] for r in rows], label=ball, color=color)
    ax.plot([.3, 1.4], [.3, 1.4], "--", color="gray")
    ax.set(xlabel="Published post-slip speed (m/s)", ylabel="MuJoCo post-slip speed (m/s)", title="Independent oblique ball collisions")
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    for family, marker in (("cushion", "o"), ("jaw_corner", "s"), ("jaw_middle", "^")):
        rows = [r for r in a["dynamic_cushions"]+a["dynamic_jaws"] if r["family"] == family]
        ax.scatter([r["angle_from_normal_deg"] for r in rows], [r["rebound_angle_deg"] for r in rows], marker=marker, alpha=.6, label=family)
    ax.plot([0, 55], [0, 55], "--", color="gray")
    ax.set(xlabel="Incident angle from normal (degrees)", ylabel="Rebound angle from normal (degrees)", title="Rubber transfer audit (no measured jaw data)")
    ax.legend(fontsize=8)
    fig.suptitle("Published-reference snooker calibration — MuJoCo " + report["mujoco_version"], fontsize=15)
    fig.savefig(output / "calibration.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", action="store_true")
    parser.add_argument("--check", action="store_true", help="Check documented numerical regression budgets")
    parser.add_argument("--plot", action="store_true", help="Write a comparison figure (requires matplotlib)")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/snooker_calibration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    profile = SNOOKER_PHYSICS
    fits = {}
    if args.fit:
        profile, fits = fit_profile(profile)
        print("FIT " + json.dumps(asdict(profile)), flush=True)
    report = {"mujoco_version": mujoco.__version__, "profile": asdict(profile), "fits": fits,
              "audit": audit(profile)}
    report["summary"] = summarize(report)
    if args.fit:
        (args.output / "fit.json").write_text(json.dumps({"profile": asdict(profile), "fits": fits}, indent=2)+"\n")
    (args.output / "calibration.json").write_text(json.dumps(report, indent=2)+"\n")
    if args.plot:
        write_plot(report, args.output)
    if args.check:
        check_report(report)
    print(json.dumps(report["summary"], indent=2))
    print(f"Report: {args.output / 'calibration.json'}")


if __name__ == "__main__":
    main()
