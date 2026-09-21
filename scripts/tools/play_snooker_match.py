"""Run two JSON-lines players, optionally recording their actual match physics.

Use --record outputs/my_match --fps 20 with MUJOCO_GL=egl for headless video.
Even --max-shots runs produce a video, marked unfinished when appropriate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys

from _bootstrap import add_src_to_path
add_src_to_path()
from snooker_env.snooker_env import SnookerMatchEnv
from snooker_env.snooker_players import SnookerProgramPlayer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--player-0', required=True, help='Program and arguments (no shell expansion)')
    parser.add_argument('--player-1', required=True, help='Program and arguments (no shell expansion)')
    parser.add_argument('--max-shots', type=int, help='Optional diagnostic stop; does not adjudicate a winner')
    parser.add_argument('--record', type=Path, metavar='DIRECTORY',
                        help='Save match.mp4, match.json and trajectories in a new directory')
    parser.add_argument('--fps', type=int, default=20, help='Recording frames per second (default: 20)')
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error('--fps must be positive')
    if args.max_shots is not None and args.max_shots < 0:
        parser.error('--max-shots must be nonnegative')
    if args.record is not None:
        try:
            args.record.mkdir(parents=True, exist_ok=False)
        except OSError as error:
            parser.error(f'Cannot create recording directory: {error}')
    players = {'player_0': SnookerProgramPlayer(shlex.split(args.player_0)),
               'player_1': SnookerProgramPlayer(shlex.split(args.player_1))}
    with SnookerMatchEnv() as env:
        recorder = None
        if args.record is not None:
            from snooker_env.match_recording import MatchRecorder
            recorder = MatchRecorder(env.physics, fps=args.fps)
        state = env.get_match_state()
        stop_reason = 'max_shots'
        try:
            while not state['terminated'] and (args.max_shots is None or state['shot_count'] < args.max_shots):
                if recorder is not None:
                    recorder.begin_turn(state)
                try:
                    state = players[state['current_player']].play_turn(env)
                except BaseException as error:
                    stop_reason = f'{type(error).__name__}: {error}'
                    state = env.get_match_state()
                    if recorder is not None:
                        recorder.end_turn(state, error=stop_reason)
                    raise
                if recorder is not None:
                    recorder.end_turn(state)
                print(json.dumps(state), flush=True)
            if state['terminated']:
                stop_reason = 'terminated'
        finally:
            # Encoding and report I/O must not consume the next player's budget.
            env.close()
            if recorder is not None:
                recorder.close()
                interrupted = sys.exc_info()[0] is not None
                try:
                    video = recorder.save(args.record, state, stop_reason=stop_reason)
                    print(f'Recording saved: {video}', file=sys.stderr, flush=True)
                except Exception as error:
                    if not interrupted:
                        raise
                    print(f'Recording failed: {error}', file=sys.stderr, flush=True)


if __name__ == '__main__':
    main()
