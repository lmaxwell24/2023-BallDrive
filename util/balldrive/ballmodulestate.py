from math import hypot
from wpimath.geometry import Rotation2d, Translation2d


class BallDriveState:
    translation: Translation2d

    def __init__(self, x: float = 0, y: float = 0) -> None:
        self.translation = Translation2d(x, y)

    def speed(self) -> float:
        return hypot(self.X(), self.Y())

    def angle(self) -> Rotation2d:
        return Rotation2d(self.X(), self.Y())

    def X(self) -> float:
        return self.translation.X()

    def Y(self) -> float:
        return self.translation.Y()

    def __eq__(self, __value: object) -> bool:
        if type(__value) == BallDriveState:
            return (
                abs(__value.speed() - self.speed()) < 1e-9
                and self.angle() == __value.angle()
            )
        return False

    def __lt__(self, other: object):
        if type(other) == BallDriveState:
            return self.speed() < other.speed()
        return False
