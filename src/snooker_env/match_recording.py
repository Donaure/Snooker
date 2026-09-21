"""Record arbitrary program players without depending on a reference policy.

Capture poses while physics runs, then write trajectories and render after the
match clock has been closed. A diagnostic stop remains an unfinished match.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

from .heyball_physics import HeyballPhysics
from .snooker_physics import SnookerPhysics


class MatchRecorder:
    def __init__(self, physics: SnookerPhysics, *, fps: int = 20) -> None:
        if fps <= 0:
            raise ValueError("Recording FPS must be positive")
        self.physics = physics
        self.fps = fps
        self.turns: list[dict[str, Any]] = []
        self.frames: list[np.ndarray] = []
        self.times: list[float] = []
        self.before: dict[str, Any] = {}
        self.previous_callback = physics.frame_callback
        self.previous_interval = physics.frame_interval
        physics.frame_interval = 1 / fps
        physics.frame_callback = self.capture

    def capture(self, physics: HeyballPhysics) -> None:
        self.frames.append(physics.data.qpos.copy())
        self.times.append(float(physics.data.time))
        if self.previous_callback is not None:
            self.previous_callback(physics)

    def begin_turn(self, state: dict[str, Any]) -> None:
        self.before = state
        self.frames = [self.physics.data.qpos.copy()]
        self.times = [float(self.physics.data.time)]

    def end_turn(self, state: dict[str, Any], *, error: str | None = None) -> None:
        if error is not None:
            # Failed simulations roll back; never show them as executed shots.
            self.frames = self.frames[:1]
            self.times = self.times[:1]
        self.turns.append({
            "before": self.before, "after": state, "error": error,
            "frames": self.frames, "simulation_time": self.times,
            "final_qpos": self.physics.data.qpos.copy(),
        })
        self.frames, self.times = [], []

    def close(self) -> None:
        self.physics.frame_callback = self.previous_callback
        self.physics.frame_interval = self.previous_interval

    def save(self, output: Path, state: dict[str, Any], *, stop_reason: str) -> Path:
        """Persist JSON/NPZ first, so trajectories survive a rendering failure."""
        turns = []
        for index, turn in enumerate(self.turns, 1):
            name = f"turn_{index:04d}.npz"
            np.savez_compressed(
                output / name, qpos=np.asarray(turn["frames"], dtype=np.float64),
                simulation_time=np.asarray(turn["simulation_time"], dtype=np.float64),
                final_qpos=turn["final_qpos"],
            )
            turns.append({key: turn[key] for key in ("before", "after", "error")} | {"trace": name})
        report = {
            "completed": state["terminated"], "stop_reason": stop_reason,
            "winner": state["winner"], "scores": state["scores"],
            "final_state": state, "fps": self.fps, "turns": turns,
            "playback": "simulation time; decision waiting omitted",
        }
        (output / "match.json").write_text(json.dumps(report, indent=2) + "\n")
        return self._render(output, state, stop_reason)

    def _render(self, output: Path, state: dict[str, Any], stop_reason: str) -> Path:
        import cv2
        import imageio.v2 as imageio
        import mujoco

        model = copy.copy(self.physics.model)
        width, height = 960, 540
        model.vis.global_.offwidth = width
        model.vis.global_.offheight = height
        data = mujoco.MjData(model)
        options = mujoco.MjvOption()
        options.sitegroup[3] = 0
        options.geomgroup[3] = 0
        video = output / "match.mp4"
        with mujoco.Renderer(model, height=height, width=width) as renderer:
            def frame(qpos: np.ndarray, snapshot: dict[str, Any], title: str) -> np.ndarray:
                data.qpos[:] = qpos
                data.qvel[:] = 0
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera="snooker_overhead", scene_option=options)
                result = np.full((720, width, 3), (14, 24, 30), dtype=np.uint8)
                result[90:630] = renderer.render()
                scores = snapshot["scores"]
                labels = (
                    ("SNOOKER MATCH", 30, .75),
                    (f"player_0: {scores['player_0']}    player_1: {scores['player_1']}"
                     f"    Break: {snapshot['current_break']}", 65, .65),
                    (title[:115], 666, .55),
                    (f"Shots: {snapshot['shot_count']}  |  {snapshot['reason']}"[:115], 703, .5),
                )
                for label, y, scale in labels:
                    cv2.putText(result, label, (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                                scale, (224, 237, 235), 1, cv2.LINE_AA)
                return result

            with imageio.get_writer(video, fps=self.fps, codec="libx264", quality=8,
                                    ffmpeg_params=["-movflags", "+faststart"]) as writer:
                for turn in self.turns:
                    before, after = turn["before"], turn["after"]
                    executed = after["shot_count"] > before["shot_count"]
                    title = f"{before['current_player']} | rule choice"
                    if executed:
                        shot = after["last_shot"]
                        title = (f"{shot['player']} | Shot {after['shot_count']} | "
                                 f"Cue {shot['speed']:.2f} m/s | "
                                 f"Offset ({shot['u']:+.2f}, {shot['v']:+.2f})")
                    elif turn["error"]:
                        title = "Interrupted turn; no shot committed"
                    for qpos in turn["frames"]:
                        writer.append_data(frame(qpos, before, title))
                    ending = frame(turn["final_qpos"], after, title)
                    for _ in range(max(1, round(.75 * self.fps))):
                        writer.append_data(ending)
                if state["terminated"]:
                    title = "DRAW" if state["winner"] is None else f"WINNER: {state['winner']}"
                else:
                    title = f"UNFINISHED | {stop_reason}"
                ending = frame(self.physics.data.qpos, state, title)
                for _ in range(2 * self.fps):
                    writer.append_data(ending)
        return video
