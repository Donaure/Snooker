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
    Diagnostics belong on stderr. One instance owns one player's process for
    one frame, started on its first turn and reused until close(). Startup ends
    with {"op": "ready"}, before any match state is sent. Its separate timeout
    excludes startup from the decision clock. Use a context
    manager to clean up the process group, including descendants, on every exit.
    Protocol lines are limited to 64 KiB.
    """

    def __init__(self, command: Sequence[str], *, startup_seconds: float = 300.) -> None:
        if isinstance(command, str) or not command:
            raise ValueError("Supply a nonempty argv sequence")
        self.command = tuple(command)
        self.startup_seconds = _scalar(startup_seconds, 'startup_seconds')
        if self.startup_seconds <= 0:
            raise ValueError('startup_seconds must be positive')
        self._process: subprocess.Popen | None = None
        self._incoming = bytearray()
        self._closed = False

    def __enter__(self) -> SnookerProgramPlayer:
        if self._closed:
            raise RuntimeError('Player is closed; create a new player for each frame')
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        """Stop the entire player process group and release its pipes."""
        self._closed = True
        process = self._process
        if process is None:
            return
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()
        process.stdin.close()
        process.stdout.close()
        self._process = None
        self._incoming.clear()

    def start(self) -> None:
        """Start once and await ready; callers may also pre-initialize before a frame.

        No game state is supplied until readiness. Imports, model loading and
        warm-up must finish before the child writes its ready line and flushes.
        """
        if self._closed:
            raise RuntimeError('Player is closed; create a new player for each frame')
        if self._process is not None:
            return
        deadline = time.monotonic() + self.startup_seconds
        try:
            self._process = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                start_new_session=True, bufsize=0)
            os.set_blocking(self._process.stdout.fileno(), False)
            os.set_blocking(self._process.stdin.fileno(), False)
            with selectors.DefaultSelector() as selector:
                selector.register(self._process.stdout, selectors.EVENT_READ)
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError('Player initialization timed out')
                    newline = self._incoming.find(b'\n')
                    if newline > 65536 or (newline < 0 and len(self._incoming) > 65536):
                        raise ValueError('Player protocol line exceeds 64 KiB')
                    if newline >= 0:
                        line = bytes(self._incoming[:newline])
                        del self._incoming[:newline + 1]
                        if json.loads(line) != {'op': 'ready'}:
                            raise ValueError('Expected {"op": "ready"} after initialization')
                        if self._incoming:
                            raise ValueError('Read the state response before sending another request or shot')
                        return
                    if not selector.select(remaining):
                        raise TimeoutError('Player initialization timed out')
                    try:
                        chunk = os.read(self._process.stdout.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if not chunk:
                        raise ValueError('Player exited before ready')
                    self._incoming.extend(chunk)
        except BaseException:
            self.close()
            raise

    def play_turn(self, env: SnookerMatchEnv) -> dict:
        if self._closed:
            raise RuntimeError('Player is closed; create a new player for each frame')
        try:
            state = self._play_turn(env)
        except BaseException:
            self.close()
            raise
        if state['terminated']:
            self.close()
        return state

    def _play_turn(self, env: SnookerMatchEnv) -> dict:
        state = env.get_match_state()
        if state['terminated']:
            return state
        actor, token = state['current_player'], state['turn_id']
        action = None
        try:
            if self._process is None:
                with env._player_initialization(actor, turn_id=token):
                    self.start()
                state = env.get_match_state()
                if state['terminated']:
                    return state
            deadline = time.monotonic() + state['seconds_remaining']
            process = self._process
            if process.poll() is not None:
                raise ValueError('Player exited before its next turn')
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ)
                selector.register(process.stdin, selectors.EVENT_WRITE)
                output = memoryview((json.dumps(state) + '\n').encode())
                incoming = self._incoming
                while action is None:
                    state = env.get_match_state()
                    if state['terminated'] or state['turn_id'] != token:
                        return state
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return env.forfeit(actor, turn_id=token, reason='timeout')
                    newline = incoming.find(b'\n')
                    if newline > 65536 or (newline < 0 and len(incoming) > 65536):
                        raise ValueError('Player protocol line exceeds 64 KiB')
                    if newline >= 0:
                        line = bytes(incoming[:newline])
                        del incoming[:newline + 1]
                        message = json.loads(line)
                        # Do not let queued output skip a turn's state delivery.
                        if output:
                            raise ValueError('Read the state response before sending another request or shot')
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
                        state = self._request(env, actor, token, message)
                        if state['terminated'] or state['turn_id'] != token:
                            return state
                        output = memoryview((json.dumps(state) + '\n').encode())
                        selector.register(process.stdin, selectors.EVENT_WRITE)
                        continue
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
                        try:
                            chunk = os.read(process.stdout.fileno(), 65536)
                        except BlockingIOError:
                            continue
                        if not chunk:
                            if incoming:
                                incoming += b'\n'  # Also accept a final array without newline.
                            else:
                                raise ValueError('Player exited without a shot')
                        else:
                            incoming += chunk
            # Reading, parsing and pre-shot choices share the decision deadline.
            if time.monotonic() >= deadline:
                return env.forfeit(actor, turn_id=token, reason='timeout')
        except TimeoutError:
            return env.forfeit(actor, turn_id=token, reason='player_initialization_timeout')
        except (ValueError, OSError, KeyError, TypeError) as error:
            return env.forfeit(actor, turn_id=token, reason=f'player_program_error: {error}')
        except RuntimeError:
            if env.get_match_state()['terminated']:
                return env.get_match_state()
            raise
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
