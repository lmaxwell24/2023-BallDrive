import functools
import operator
import typing
import numpy as np
from wpimath.geometry import Translation2d, Twist2d
from wpimath.kinematics import ChassisSpeeds
from util.balldrive.ballmoduleposition import BallModulePosition

from util.balldrive.ballmodulestate import BallDriveState


class BallDriveKinematics:
    inverseKinematics: np.ndarray
    forwardKinematics: np.ndarray

    numModules: int
    modules: typing.Tuple[Translation2d]
    moduleStates: typing.List[BallDriveState]

    prevCoR = Translation2d()

    def __init__(self, wheelPositions: typing.Tuple[Translation2d]) -> None:
        if len(wheelPositions) < 2:
            raise SyntaxError("Ball drive requires at least two modules")

        self.numModules = len(wheelPositions)
        self.modules = wheelPositions
        self.moduleStates = [BallDriveState()] * self.numModules

        self.inverseKinematics = np.array(
            functools.reduce(  # a bit of fun python to make arrays inline
                operator.add,
                [[[1, 0, -module.Y()], [0, 1, +module.X()]] for module in self.modules],
            )
        )

        self.forwardKinematics = np.linalg.inv(self.inverseKinematics)

    def toSwerveModuleStates(
        self, speeds: ChassisSpeeds, cor: Translation2d = Translation2d()
    ) -> typing.List[BallDriveState]:
        """
        * Performs inverse kinematics to return the module states from a desired chassis velocity. This
        * method is often used to convert joystick values into module speeds and angles.
        *
        * <p>This function also supports variable centers of rotation. During normal operations, the
        * center of rotation is usually the same as the physical center of the robot; therefore, the
        * argument is defaulted to that use case. However, if you wish to change the center of rotation
        * for evasive maneuvers, vision alignment, or for any other use case, you can do so.
        """
        if speeds.vx == 0 and speeds.vy == 0 and speeds.omega == 0:
            return [BallDriveState()] * self.numModules

        if self.prevCoR != cor:
            self.inverseKinematics = np.array(
                functools.reduce(
                    operator.add,
                    [
                        [[1, 0, -module.Y() + cor.Y()], [0, 1, +module.X() - cor.X()]]
                        for module in self.modules
                    ],
                )
            )
            self.prevCoR = cor

        chassisSpeedVector = np.array([[speeds.vx], [speeds.vy], [speeds.omega]])

        moduleStateMatrix = self.inverseKinematics @ chassisSpeedVector

        for i in range(0, self.numModules):
            self.moduleStates[i] = BallDriveState(
                moduleStateMatrix[i * 2][0], moduleStateMatrix[i * 2 + 1][0]
            )
        return self.moduleStates

    def toChassisSpeeds(self, states: typing.List[BallDriveState]) -> ChassisSpeeds:
        if len(states) != self.numModules:
            raise SyntaxError(
                "Number of modules is not consistent with number of wheel locations defined for this kinematics"
            )

        moduleStateMatrix = np.ndarray((self.numModules, 1))
        for i, module in enumerate(states):
            moduleStateMatrix[i * 2][0] = module.X()
            moduleStateMatrix[i * 2 + 1][0] = module.Y()

        chassisSpeedVector = self.forwardKinematics @ moduleStateMatrix
        return ChassisSpeeds(
            chassisSpeedVector[0][0], chassisSpeedVector[1][0], chassisSpeedVector[2, 0]
        )

    def toTwist2d(self, wheelDeltas: typing.List[BallModulePosition]) -> Twist2d:
        if len(wheelDeltas) != self.numModules:
            raise SyntaxError(
                "Number of modules is not consistent with number of wheel locations defined for this kinematics"
            )

        moduleDeltaMatrix = np.ndarray((self.numModules, 1))
        for i, module in enumerate(wheelDeltas):
            moduleDeltaMatrix[i * 2][0] = module.X()
            moduleDeltaMatrix[i * 2 + 1][0] = module.Y()

        chassisDeltaVector = self.forwardKinematics @ moduleDeltaMatrix
        return Twist2d(
            chassisDeltaVector[0][0], chassisDeltaVector[1][0], chassisDeltaVector[2, 0]
        )

    def desaturateWheelSpeeds(self, states: typing.List[BallDriveState],maxSpeed: float) -> typing.List[BallDriveState]:
        realMax = max([max(a.X(),a.Y()) for a in states])
        if realMax > maxSpeed:
            return [BallDriveState(a.X() / realMax * maxSpeed, a.Y() / realMax * maxSpeed) for a in states]
        return states
        

