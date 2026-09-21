"""Deadline-enforced JSON-lines transport for independent snooker programs."""
from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import time
from typing import Sequence

from .cue_stroke import validate_cue_offset
from .heyball_env import _scalar
from .snooker_env import SnookerMatchEnv


class SnookerProgramPlayer:
    """State on stdin; [angle, speed, u, v] or [angle, speed] on stdout.

    Omitted offsets default to zero. Supplied offsets lie in the open unit disk
    of the fixed horizontal view: right is +u and world up is +v.

    Requests are JSON objects with an ``op``: state, nominate_colour (ball),
    nominate_free_ball (ball), place_cue_ball (x, y), choose_foul (choice).
    Each request receives a new state line without extending the turn deadline.
    Requiring the opponent to replay ends this program's turn immediately.
    Diagnostics belong on stderr. Each turn starts a new process group, killed
    on completion or timeout. Protocol lines are limited to 64 KiB.
    """

    def __init__(self, command: Sequence[str]) -> None:
        if isinstance(command, str) or not command:
            raise ValueError("Supply a nonempty argv sequence")
        self.command = tuple(command)

    def play_turn(self, env: SnookerMatchEnv) -> dict:
        state = env.get_match_state()
        if state['terminated']:
            return state
        actor, token = state['current_player'], state['turn_id']
        deadline = time.monotonic() + state['seconds_remaining']
        try:
            process = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       start_new_session=True, bufsize=0)
        except OSError as error:
            return env.forfeit(actor, turn_id=token, reason=f'player_program_error: {error}')
        action = None
        try:
            with selectors.DefaultSelector() as selector:
                os.set_blocking(process.stdout.fileno(), False)
                os.set_blocking(process.stdin.fileno(), False)
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stdin, selectors.EVENT_WRITE)
                output = memoryview((json.dumps(state) + '\n').encode())
                incoming = bytearray()
                while action is None:
                    state = env.get_match_state()
                    if state['terminated'] or state['turn_id'] != token:
                        return state
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return env.forfeit(actor, turn_id=token, reason='timeout')
                    events = selector.select(remaining)
                    if not events:
                        return env.forfeit(actor, turn_id=token, reason='timeout')
                    for key, _ in events:
                        if key.fileobj is process.stdin:
                            try:
                                written = os.write(process.stdin.fileno(), output)
                            except BlockingIOError:
                                continue
                            output = output[written:]
                            if not output:
                                selector.unregister(process.stdin)
                            continue
                        chunk = os.read(process.stdout.fileno(), 65536)
                        if not chunk:
                            if incoming:
                                incoming += b'\n'  # Also accept a final array without newline.
                            else:
                                raise ValueError('Player exited without a shot')
                        else:
                            incoming += chunk
                        if len(incoming) > 65536:
                            raise ValueError('Player protocol line exceeds 64 KiB')
                        while b'\n' in incoming:
                            line, _, rest = incoming.partition(b'\n')
                            incoming = bytearray(rest)
                            message = json.loads(line)
                            if isinstance(message, list) and len(message) in (2, 4):
                                angle, speed = _scalar(message[0], 'angle'), _scalar(message[1], 'speed')
                                if speed <= 0:
                                    raise ValueError('Cue speed must be positive')
                                u, v = validate_cue_offset(*message[2:]) if len(message) == 4 else (0., 0.)
                                action = (angle, speed, u, v)
                                break
                            if not isinstance(message, dict):
                                raise ValueError('Expected [angle, speed], [angle, speed, u, v] or a request object')
                            # One request/response at a time bounds pending writes.
                            if output:
                                raise ValueError('Read the state response before sending another request')
                            state = self._request(env, actor, token, message)
                            if state['terminated'] or state['turn_id'] != token:
                                return state
                            output = memoryview((json.dumps(state) + '\n').encode())
                            selector.register(process.stdin, selectors.EVENT_WRITE)
            # Deadline includes process startup, reading, parsing and pre-shot choices.
            if time.monotonic() >= deadline:
                return env.forfeit(actor, turn_id=token, reason='timeout')
        except (ValueError, OSError, KeyError, TypeError) as error:
            return env.forfeit(actor, turn_id=token, reason=f'player_program_error: {error}')
        except RuntimeError:
            if env.get_match_state()['terminated']:
                return env.get_match_state()
            raise
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            process.stdin.close()
            process.stdout.close()
        # Physics errors propagate and preserve the shot; they are not forfeits.
        try:
            return env.submit_shot(actor, *action, turn_id=token)
        except RuntimeError:
            if env.get_match_state()['terminated']:
                return env.get_match_state()
            raise

    @staticmethod
    def _request(env: SnookerMatchEnv, actor: str, token: int, message: dict) -> dict:
        op = message['op']
        if op == 'state':
            return env.get_match_state(actor)
        if op == 'nominate_colour':
            return env.nominate_colour(actor, message['ball'], turn_id=token)
        if op == 'nominate_free_ball':
            return env.nominate_free_ball(actor, message['ball'], turn_id=token)
        if op == 'place_cue_ball':
            return env.place_cue_ball(actor, message['x'], message['y'], turn_id=token)
        if op == 'choose_foul':
            return env.choose_foul(actor, message['choice'], turn_id=token)
        raise ValueError(f'Unknown operation: {op}')
