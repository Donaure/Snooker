"""Render the generated table and actual native contact rollouts for inspection."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from _bootstrap import add_src_to_path
ROOT = add_src_to_path()
from snooker_env.snooker_geometry import BALL_RADIUS, BALL_Z, TABLE_WIDTH, rack_positions
from snooker_env.snooker_physics import SnookerPhysics


def camera(distance: float, azimuth: float, elevation: float, lookat: tuple) -> mujoco.MjvCamera:
    result = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(result)
    result.distance, result.azimuth, result.elevation = distance, azimuth, elevation
    result.lookat[:] = lookat
    return result


def caption(frame: np.ndarray, title: str, subtitle: str) -> np.ndarray:
    im = Image.fromarray(frame)
    draw = ImageDraw.Draw(im)
    font_path = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font = ImageFont.truetype(font_path, 26)
    detail_font = ImageFont.truetype(font_path, 19)
    draw.rectangle((0, 0, im.width, 84), fill=(18, 24, 24))
    draw.text((24, 12), title, fill='white', font=title_font)
    draw.text((24, 49), subtitle, fill=(210, 222, 214), font=detail_font)
    return np.asarray(im)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/snooker_realism')
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--images-only', action='store_true', help='Render the four asset views without physics rollouts')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    p = SnookerPhysics()
    p.reset(rack_positions())
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.geomgroup[3] = 0
    width, height, supersample = 1440, 1008, 2
    render_width, render_height = width*supersample, height*supersample
    p.model.vis.global_.offwidth = render_width
    p.model.vis.global_.offheight = render_height

    def render_frame(renderer: mujoco.Renderer) -> np.ndarray:
        # Average the supersampled image before adding text or encoding video.
        return cv2.resize(renderer.render(), (width, height), interpolation=cv2.INTER_AREA)

    with mujoco.Renderer(p.model, height=render_height, width=render_width) as renderer:
        for name in ('snooker_overhead', 'snooker_oblique', 'snooker_corner_detail', 'snooker_middle_detail'):
            renderer.update_scene(p.data, camera=name, scene_option=option)
            imageio.imwrite(args.output/f'{name}.png', render_frame(renderer))

    if args.images_only:
        print(f'IMAGES {args.output}', flush=True)
        return

    # Velocities below are measured diagnostic launch conditions, not cue actions.
    cases = []
    for name, offset in (('corner_clean', 0.), ('corner_jaw', .035),
                         ('middle_clean', 0.), ('middle_jaw', .035)):
        middle = name.startswith('middle')
        center = np.array(p.pockets['pocket_5' if middle else 'pocket_3'][:2])
        direction = np.sign(center)
        if middle:
            direction[1] = 0.
        direction /= np.linalg.norm(direction)
        tangent = np.array([-direction[1], direction[0]])
        xy = center - .4*direction + offset*tangent
        view = (camera(1.15, 8., -53., (*center, 1.01)) if middle else
                camera(1.35, 45., -55., (*(center-.12*direction), 1.02)))
        cases.append(dict(name=name, positions={0: (*xy, BALL_Z)}, velocity=.8*direction,
                          view=view, seconds=2.5,
                          title=('Middle' if middle else 'Corner')+' pocket: '+('centre approach' if offset == 0. else 'jaw contact'),
                          detail=f'Initial rolling speed 0.80 m/s | approach offset {offset*1000:.0f} mm | actual MuJoCo motion'))
    cases.append(dict(name='cushion', positions={0: (TABLE_WIDTH/2-.28, .5, BALL_Z)},
                      velocity=np.array([math.cos(math.pi/6), math.sin(math.pi/6)]),
                      view=camera(1.8, 0., -60., (.60, .65, 1.05)), seconds=2.,
                      title='Cushion rebound', detail='Initial rolling speed 1.00 m/s | 30 degree incidence | fitted rubber contact'))
    cases.append(dict(name='ball_collision', positions={0: (-.30, 0., BALL_Z), 1: (0., .025, BALL_Z)},
                      velocity=np.array([1., 0.]), view=camera(1.9, -90., -65., (0., .05, 1.05)), seconds=2.,
                      title='Oblique ball collision', detail='Initial rolling speed 1.00 m/s | 25 mm centre offset | native impulses and spin'))
    cases.append(dict(name='rolling', positions={0: (0., -.8, BALL_Z)}, velocity=np.array([0., .7]),
                      view='snooker_overhead', seconds=5.8,
                      title='Cloth rolling resistance', detail='Initial rolling speed 0.70 m/s | reference deceleration 0.125 m/s² | real-time playback'))
    report = []
    with mujoco.Renderer(p.model, height=render_height, width=render_width) as renderer, imageio.get_writer(
            args.output/'snooker_diagnostics.mp4', fps=args.fps, codec='libx264', quality=8,
            macro_block_size=16) as writer:
        p.reset(rack_positions())
        for _ in range(2*args.fps):
            renderer.update_scene(p.data, camera='snooker_oblique', scene_option=option)
            writer.append_data(caption(render_frame(renderer), 'Full-size snooker table',
                                      '3569 × 1778 mm playing area | 52.5 mm balls | curved jaws, slate shelves and net pockets'))
        for case in cases:
            p.reset(case['positions'])
            # Settle the initial ball/slate contact before prescribing the launch.
            mujoco.mj_step(p.model, p.data, nstep=1000)
            v = int(p.vadr[0])
            vx, vy = case['velocity']
            p.data.qvel[v:v+6] = (vx, vy, 0., -vy/BALL_RADIUS, vx/BALL_RADIUS, 0.)
            start = float(p.data.time)
            contacts, exits = set(), []
            for frame_index in range(round(case['seconds']*args.fps)):
                until = start+(frame_index+1)/args.fps
                while p.data.time < until:
                    p._step()
                    for c in p.data.contact:
                        pair = {int(c.geom1), int(c.geom2)}
                        if c.dist <= 0 and pair & set(p.geoms) and pair & p.rails:
                            contacts.update(p.model.geom(g).name for g in pair & p.rails)
                    for number in tuple(p.active):
                        position = p.data.qpos[p.qadr[number]:p.qadr[number]+3]
                        outcome = p._ball_exit_kind(position, number)
                        if outcome:
                            exits.append({'ball': number, 'outcome': outcome, 'time': float(p.data.time-start)})
                            p._remove(number)
                renderer.update_scene(p.data, camera=case['view'], scene_option=option)
                writer.append_data(caption(render_frame(renderer), case['title'], case['detail']))
            report.append({'name': case['name'], 'initial_positions': case['positions'],
                           'initial_velocity_m_s': case['velocity'].tolist(), 'seconds': case['seconds'],
                           'cushion_contacts': sorted(contacts), 'exits': exits, 'final_positions': p.positions()})
            print(json.dumps(report[-1]), flush=True)
    (args.output/'diagnostic_rollouts.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'VIDEO {args.output / "snooker_diagnostics.mp4"}', flush=True)


if __name__ == '__main__':
    main()
