import typing
from wpimath.geometry import Pose2d, Rotation2d
from util.balldrive.balldrivekinematics import BallDriveKinematics

from util.balldrive.ballmoduleposition import BallModulePosition


class BallDriveOdometry:
    kinematics: BallDriveKinematics
    pose: Pose2d

    gyroOffset: Rotation2d
    previousAngle: Rotation2d
    numModules: int
    previousModulePositions: typing.List[BallModulePosition]

    def __init__(
        self,
        kinematics: BallDriveKinematics,
        gyroAngle: Rotation2d,
        modulePositions: typing.List[BallModulePosition],
        initialPose: Pose2d = Pose2d(),
    ) -> None:
        self.kinematics = kinematics
        self.pose = initialPose
        self.gyroOffset = self.pose.rotation() - gyroAngle
        self.previousAngle = self.pose.rotation()
        self.numModules = len(modulePositions)
        self.previousModulePositions = modulePositions

    def resetPosition(
        self,
        gyroAngle: Rotation2d,
        modulePositions: typing.List[BallModulePosition],
        pose: Pose2d = Pose2d(),
    ):
        if len(modulePositions) != self.numModules:
            raise SyntaxError(
                "Number of modules is not consistent with number of wheel locations defined for this kinematics"
            )

        self.pose = pose
        self.previousAngle = pose.rotation()
        self.gyroOffset = self.pose.rotation() - gyroAngle
        self.previousModulePositions = modulePositions

    def getPose(self) -> Pose2d:
        return self.pose

    def update(
        self, gyroAngle: Rotation2d, modulePositions: typing.List[BallModulePosition]
    ):
        if len(modulePositions) != self.numModules:
            raise SyntaxError(
                "Number of modules is not consistent with number of wheel locations defined for this kinematics"
            )

        moduleDeltas = [BallModulePosition(a.X() - b.X(), a.Y() - b.Y()) for a,b in zip(modulePositions, self.previousModulePositions)]
        angle = gyroAngle + self.gyroOffset
        
        twist = self.kinematics.toTwist2d(moduleDeltas)
        twist.dtheta = (angle - self.previousAngle).radians()
        
        newPose = self.pose.exp(twist)

        self.previousAngle = angle
        self.pose = Pose2d(newPose.translation(), angle)

        return self.pose
