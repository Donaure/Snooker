"""Measure cloth and cushion response in the actual compiled snooker scene."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from _bootstrap import add_src_to_path
ROOT = add_src_to_path()
from snooker_env.snooker_geometry import BALL_RADIUS, BALL_Z, SURFACE_Z, TABLE_WIDTH
from snooker_env.snooker_physics import SnookerPhysics


def initialise(physics: SnookerPhysics, position: tuple, velocity: tuple,
               *, rolling: bool = True, spin: float = 0.) -> None:
    physics.reset({0: (*position, BALL_Z)})
    mujoco.mj_step(physics.model, physics.data, nstep=1000)
    v = int(physics.vadr[0])
    physics.data.qvel[v:v+3] = (*velocity, 0.)
    physics.data.qvel[v+3:v+6] = (-velocity[1]/BALL_RADIUS if rolling else 0.,
                                 velocity[0]/BALL_RADIUS if rolling else 0., spin)


def cloth_measurement(physics: SnookerPhysics, speed: float, *, rolling: bool) -> dict:
    initialise(physics, (0., -.6), (0., speed), rolling=rolling)
    start = float(physics.data.time)
    duration = 1.0 if rolling else .25
    stride = max(1, round(.002/physics.model.opt.timestep))
    samples = []
    while physics.data.time-start < duration:
        mujoco.mj_step(physics.model, physics.data, nstep=stride)
        velocity = physics.data.qvel[physics.vadr[0]:physics.vadr[0]+6]
        speed_now = float(np.linalg.norm(velocity[:2]))
        slip = np.linalg.norm(velocity[:2] + BALL_RADIUS*np.array([-velocity[4], velocity[3]]))
        samples.append([float(physics.data.time-start), speed_now, float(slip)])
    array = np.array(samples)
    selected = ((array[:, 0] > .03) & (array[:, 1] > .15) if rolling else
                (array[:, 0] > .005) & (array[:, 2] > .10))
    if np.count_nonzero(selected) < 3:
        raise RuntimeError('Insufficient moving samples for cloth fit')
    slope, intercept = np.polyfit(array[selected, 0], array[selected, 1], 1)
    transition = np.flatnonzero(array[:, 2] < .01)
    return {'initial_speed': speed, 'rolling': rolling, 'deceleration_m_s2': float(-slope),
            'friction_effective': float(-slope/9.81),
            'sliding_to_rolling_seconds': float(array[transition[0], 0]) if len(transition) else None,
            'fit_rmse_m_s': float(np.sqrt(np.mean((array[selected, 1]-(intercept+slope*array[selected, 0]))**2)))}


def cushion_measurement(physics: SnookerPhysics, speed: float, angle: float = 0., spin: float = 0.) -> dict:
    direction = np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
    initialise(physics, (TABLE_WIDTH/2-BALL_RADIUS-.06, .45), tuple(speed*direction), spin=spin)
    previous = physics.data.qvel[physics.vadr[0]:physics.vadr[0]+6].copy()
    incoming = None
    touching_before = False
    maximum_z = BALL_Z
    start = float(physics.data.time)
    contacts = set()
    while physics.data.time-start < 1.2:
        physics._step()
        velocity = physics.data.qvel[physics.vadr[0]:physics.vadr[0]+6].copy()
        touching = False
        for c in physics.data.contact:
            pair = {int(c.geom1), int(c.geom2)}
            if physics.geoms[0] in pair and pair & physics.rails and c.dist <= 0:
                touching = True
                contacts.update(physics.model.geom(g).name for g in pair & physics.rails)
        maximum_z = max(maximum_z, float(physics.data.qpos[physics.qadr[0]+2]))
        if touching and incoming is None:
            incoming = previous.copy()
        if touching_before and not touching and velocity[0] < 0:
            break
        touching_before = touching
        previous = velocity
    if incoming is None:
        raise RuntimeError('Cushion was not contacted')
    return {'initial_speed': speed, 'incidence_degrees': angle, 'initial_sidespin_rad_s': spin,
            'incoming_velocity': incoming.tolist(), 'outgoing_velocity': velocity.tolist(),
            'horizontal_normal_restitution': float(-velocity[0]/incoming[0]),
            'rebound_angle_degrees': float(math.degrees(math.atan2(velocity[1], -velocity[0]))),
            'maximum_center_height_above_cloth_m': maximum_z-SURFACE_Z,
            'contact_geoms': sorted(contacts)}


def pocket_measurement(physics: SnookerPhysics, pocket: str, speed: float, offset: float) -> dict:
    center = np.array(physics.pockets[pocket][:2])
    direction = np.sign(center)
    if abs(center[1]) < .1:
        direction[1] = 0.
    direction /= np.linalg.norm(direction)
    tangent = np.array([-direction[1], direction[0]])
    start_xy = center-.4*direction+offset*tangent
    initialise(physics, tuple(start_xy), tuple(speed*direction))
    start = float(physics.data.time)
    stride = max(1, round(.0005/physics.model.opt.timestep))
    contacts = set()
    status = 'stopped_or_rebounded'
    while physics.data.time-start < 3.5:
        for _ in range(stride):
            physics._step()
        for c in physics.data.contact:
            pair = {int(c.geom1), int(c.geom2)}
            if physics.geoms[0] in pair and pair & physics.rails and c.dist <= 0:
                contacts.update(physics.model.geom(g).name for g in pair & physics.rails)
        position = physics.data.qpos[physics.qadr[0]:physics.qadr[0]+3]
        exit_kind = physics._ball_exit_kind(position)
        if exit_kind is not None:
            status = exit_kind
            break
        velocity = physics.data.qvel[physics.vadr[0]:physics.vadr[0]+3]
        if physics.data.time-start > .2 and np.linalg.norm(velocity) < .015:
            break
    return {'pocket': pocket, 'speed_m_s': speed, 'offset_m': offset, 'outcome': status,
            'seconds': float(physics.data.time-start), 'final_position': position.tolist(),
            'contact_geoms': sorted(contacts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pockets', action='store_true', help='Include corner/middle acceptance sweeps')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    physics = SnookerPhysics()
    report = {'model': 'models/snooker_scene.xml', 'timestep': float(physics.model.opt.timestep),
              'cloth': [], 'cushions': [], 'pockets': []}
    for rolling in (True, False):
        for speed in ((.4, .7, 1.) if rolling else (.6, 1.)):
            row = cloth_measurement(physics, speed, rolling=rolling)
            report['cloth'].append(row)
            print('CLOTH '+json.dumps(row), flush=True)
    for speed in (.35, .7, 1.2, 2.):
        row = cushion_measurement(physics, speed)
        report['cushions'].append(row)
        print('CUSHION '+json.dumps(row), flush=True)
    for angle, spin in ((30., 0.), (60., 0.), (30., -20.), (30., 20.)):
        row = cushion_measurement(physics, 1., angle, spin)
        report['cushions'].append(row)
        print('CUSHION '+json.dumps(row), flush=True)
    if args.pockets:
        for pocket in ('pocket_3', 'pocket_5'):
            for speed in (.35, .7, 1.1):
                for offset in (0., .02, .04, .06):
                    row = pocket_measurement(physics, pocket, speed, offset)
                    report['pockets'].append(row)
                    print('POCKET '+json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(f'REPORT {args.output}', flush=True)


if __name__ == '__main__':
    main()
