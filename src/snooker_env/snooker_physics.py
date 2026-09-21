"""Twenty-two-ball snooker scene with curved slate drops and pocket jaws."""
from __future__ import annotations

from typing import Callable, Mapping

import numpy as np

from .cue_stroke import validate_cue_offset
from .heyball_physics import HeyballPhysics, ShotEvents, SimulationError
from .snooker_contact import SnookerContactStepper
from .snooker_geometry import BALL_RADIUS, BALL_Z, SURFACE_Z, TABLE_LENGTH, TABLE_WIDTH
from .snooker_table_assets import POCKETS, cloth_supports


class SnookerPhysics(HeyballPhysics):
    scene_file = "snooker_scene.xml"
    ball_count = 22
    ball_radius = BALL_RADIUS
    ball_center_z = BALL_Z
    surface_z = SURFACE_Z
    pocket_entry_z = SURFACE_Z - .10
    exit_half_extents = (TABLE_WIDTH / 2 + .18, TABLE_LENGTH / 2 + .18)
    nominal_bounds = (-TABLE_WIDTH / 2, TABLE_WIDTH / 2, -TABLE_LENGTH / 2, TABLE_LENGTH / 2)
    cushion_name_prefix = "snooker_cushion_"
    cloth_name_prefix = "snooker_cloth_"

    def __init__(self, *, max_simulation_seconds: float = 60.0,
                 frame_callback: Callable[[HeyballPhysics], None] | None = None,
                 frame_interval: float = 1 / 30) -> None:
        super().__init__(max_simulation_seconds=max_simulation_seconds,
                         frame_callback=frame_callback, frame_interval=frame_interval)
        self._pocket_entries: dict[int, int] = {}
        self.contact_stepper = SnookerContactStepper(self.model, self.geoms, self.rails)

    def reset(self, positions: Mapping[int, tuple[float, float, float]]) -> None:
        self._pocket_entries: dict[int, int] = {}
        super().reset(positions)
        self.contact_stepper.reset()

    def _step(self) -> None:
        self.contact_stepper.step(self.data)

    def execute(self, angle: float, speed: float, u: float = 0., v: float = 0.) -> ShotEvents:
        """Strike a fixed surface point in the horizontal view's unit disk.

        Positive u is right, positive v is world up, and omitted coordinates
        select the facing equator point even when the cue must be elevated.
        """
        offset = validate_cue_offset(u, v)
        return self._execute_with_offset(angle, speed, cue_offset=offset)

    def _execute(self, angle: float, speed: float, *,
                 cue_offset: tuple[float, float] | None = None) -> ShotEvents:
        self.contact_stepper.reset()
        return super()._execute(angle, speed, cue_offset=(0., 0.) if cue_offset is None else cue_offset)

    def _cloth_supports(self, x: float, y: float) -> bool:
        return cloth_supports(x, y)

    def _ball_exit_kind(self, position: np.ndarray, number: int | None = None) -> str | None:
        """A pot requires entering a real slate opening and completing the drop.

        Keep the entry identity while falling: a fast ball can move sideways
        beyond the opening's XY projection by the time it reaches the bag.
        A ball returning to cloth height clears its entry and can rattle out.
        """
        key = -1 if number is None else number
        if position[2] >= self.ball_center_z - .002:
            self._pocket_entries.pop(key, None)
        elif position[2] < self.surface_z + .005 and key not in self._pocket_entries:
            for index, pocket in enumerate(POCKETS):
                if np.linalg.norm(position[:2] - pocket.center) <= pocket.drop_radius + BALL_RADIUS:
                    self._pocket_entries[key] = index
                    break
        if position[2] < self.pocket_entry_z:
            return "pocket" if key in self._pocket_entries else "off_table"
        if (key not in self._pocket_entries
                and (abs(position[0]) > self.exit_half_extents[0] + BALL_RADIUS
                     or abs(position[1]) > self.exit_half_extents[1] + BALL_RADIUS)):
            return "off_table"
        return None


__all__ = ["SnookerPhysics", "SimulationError"]
