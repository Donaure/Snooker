"""Published-reference snooker contact profile shared by scene and calibration.

MuJoCo's torsional and rolling friction coefficients are lengths (metres), not
ordinary dimensionless friction coefficients. See docs/snooker_calibration.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .snooker_geometry import BALL_RADIUS


def as_mjcf(values: Iterable[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


@dataclass(frozen=True)
class SnookerContactProfile:
    timestep: float = .00005
    ball_mass: float = .1406
    sliding_friction: float = 0.21528167485044
    rolling_friction: float = 0.00047020330326692
    spinning_friction: float = 0.00061809042165803
    ball_friction: float = .05
    ball_solref: tuple[float, float] = (-10000000, -236.98049786045)
    cloth_solref: tuple[float, float] = (-1e6, -1000.)
    cushion_friction: float = .14
    cushion_solref: tuple[float, float] = (-1000000, -50.336452183851)
    jaw_friction: float = .14
    jaw_solref: tuple[float, float] = (-1000000, -50.336452183851)
    solimp: tuple[float, float, float] = (.95, .99, .001)
    cushion_nose_height: float = 1.4 * BALL_RADIUS
    rolling_deceleration: float = .125
    sliding_deceleration: float = .212 * 9.81
    spin_deceleration: float = 22.
    ball_restitution: float = .89

    # (horizontal normal incident speed m/s, direct-format damping s^-1).
    # Fitted to Mathavan 2009 rolling normal rebounds at the production dt.
    rubber_damping_curve: tuple[tuple[float, float], ...] = (
        (0.28, 183.027017715),
        (0.5, 84.7245262343),
        (0.75, 54.9307745788),
        (1.0, 43.4510452552),
        (1.5, 57.187385775),
        (2.0, 74.06543397),
        (2.5, 107.54314432),
        (3.0, 136.678773573),
        (3.5, 157.414306103),
    )

    def rubber_damping(self, speed: float) -> float:
        """Piecewise-linear empirical fit, clamped outside measured speeds."""
        curve = self.rubber_damping_curve
        if speed <= curve[0][0]:
            return curve[0][1]
        for (low, a), (high, b) in zip(curve, curve[1:]):
            if speed <= high:
                return a + (b-a) * (speed-low)/(high-low)
        return curve[-1][1]

    @property
    def ball_friction_vector(self) -> tuple[float, float, float]:
        return self.ball_friction, 0., 0.

    @property
    def cloth_friction_vector(self) -> tuple[float, float, float]:
        return self.sliding_friction, self.spinning_friction, self.rolling_friction

    @property
    def cushion_friction_vector(self) -> tuple[float, float, float]:
        return self.cushion_friction, 0., 0.

    @property
    def jaw_friction_vector(self) -> tuple[float, float, float]:
        return self.jaw_friction, 0., 0.

    def contact_defaults(self) -> dict[str, dict[str, str]]:
        profiles = {}
        for name, friction, solref, condim, priority in (
            ("snooker_ball", self.ball_friction_vector, self.ball_solref, 3, 0),
            ("snooker_cloth", self.cloth_friction_vector, self.cloth_solref, 6, 1),
            ("snooker_cushion", self.cushion_friction_vector, self.cushion_solref, 3, 1),
            ("snooker_jaw", self.jaw_friction_vector, self.jaw_solref, 3, 1),
        ):
            profiles[name] = {"friction": as_mjcf(friction), "solref": as_mjcf(solref),
                              "solimp": as_mjcf(self.solimp), "condim": str(condim),
                              "priority": str(priority)}
        profiles["snooker_ball"]["mass"] = str(self.ball_mass)
        return profiles


SNOOKER_PHYSICS = SnookerContactProfile()
CONTACT_DEFAULTS = SNOOKER_PHYSICS.contact_defaults()
