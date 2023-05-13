from enum import Enum, auto

from typing import List, Tuple
from commands2 import SubsystemBase
from wpilib import (
    Encoder,
    PWMVictorSPX,
    RobotBase,
    SmartDashboard,
    Timer,
    DataLogManager,
)
from ctre import ControlMode, WPI_TalonFX

from ctre.sensors import CANCoder, SensorInitializationStrategy, AbsoluteSensorRange
from navx import AHRS
from wpimath.geometry import Pose2d, Rotation2d, Translation2d
from wpimath.filter import SlewRateLimiter
from wpimath.kinematics import (
    ChassisSpeeds,
)

import constants
from util import convenientmath
from util.balldrive.balldrivekinematics import BallDriveKinematics
from util.balldrive.balldriveodometry import BallDriveOdometry
from util.balldrive.ballmoduleposition import BallModulePosition
from util.balldrive.ballmodulestate import BallDriveState
from util.ctrecheck import ctreCheckError


class BallModule:
    def __init__(self, name: str) -> None:
        self.name = name

    def getBallLinearVelocity(self) -> Translation2d:
        raise NotImplementedError("Must be implemented by subclass")

    def getTotalXPosition(self) -> float:
        raise NotImplementedError("Must be implemented by subclass")

    def getTotalYPosition(self) -> float:
        raise NotImplementedError("Must be implemented by subclass")

    def setBallLinearVelocityTarget(
        self, wheelLinearVelocityTarget: Translation2d
    ) -> None:
        raise NotImplementedError("Must be implemented by subclass")

    def reset(self) -> None:
        raise NotImplementedError("Must be implemented by subclass")

    def getPosition(self) -> BallModulePosition:
        return BallModulePosition(self.getTotalXPosition(), self.getTotalYPosition())

    def getState(self) -> BallDriveState:
        velocity = self.getBallLinearVelocity()
        return BallDriveState(velocity.X(), velocity.Y())

    def applyState(self, state: BallDriveState) -> None:
        self.setBallLinearVelocityTarget(state.Translation())


# pylint: disable-next=abstract-method
class PWMSwerveModule(BallModule):
    """
    Implementation of SwerveModule designed for ease of simulation:
        wheelMotor: 1:1 gearing with wheel
        swerveMotor: 1:1 gearing with swerve
        wheelEncoder: wheel distance (meters)
        swerveEncoder: swerve angle (radians)
    """

    def __init__(
        self,
        name: str,
        xMotor: PWMVictorSPX,
        yMotor: PWMVictorSPX,
        xEncoder: Encoder,
        yEncoder: Encoder,
    ) -> None:
        BallModule.__init__(self, name)
        self.xMotor = xMotor
        self.yMotor = yMotor
        self.xEncoder = xEncoder
        self.yEncoder = yEncoder

        self.xEncoder.setDistancePerPulse(1 / constants.kWheelEncoderPulsesPerMeter)
        self.yEncoder.setDistancePerPulse(1 / constants.kSwerveEncoderPulsesPerRadian)

    def getBallLinearVelocity(self) -> Translation2d:
        return Translation2d(self.xEncoder.getRate(), self.yEncoder.getRate())

    def getTotalXPosition(self) -> float:
        return self.xEncoder.getDistance()

    def getTotalYPosition(self) -> float:
        return self.yEncoder.getDistance()

    def setBallLinearVelocityTarget(
        self, wheelLinearVelocityTarget: Translation2d
    ) -> None:
        xSpeedFactor = wheelLinearVelocityTarget.X() / constants.kMaxBallLinearVelocity
        xSpeedFactorClamped = min(max(xSpeedFactor, -1), 1)
        self.xMotor.set(xSpeedFactorClamped)

        ySpeedFactor = wheelLinearVelocityTarget.X() / constants.kMaxBallLinearVelocity
        ySpeedFactorClamped = min(max(ySpeedFactor, -1), 1)
        self.yMotor.set(ySpeedFactorClamped)

    def reset(self) -> None:
        pass


class CTREBallModule(BallModule):
    def __init__(
        self,
        name: str,
        xMotor: WPI_TalonFX,
        xMotorInverted: bool,
        yMotor: WPI_TalonFX,
        yMotorInverted: bool,
    ) -> None:
        BallModule.__init__(self, name)
        self.xMotor = xMotor
        self.xMotorInverted = xMotorInverted
        self.yMotor = yMotor
        self.yMotorInverted = yMotorInverted

        DataLogManager.log(f"Initializing ball module: {self.name}")
        DataLogManager.log(
            f"   Configuring X motor: CAN ID: {self.xMotor.getDeviceID()}"
        )
        if not ctreCheckError(
            "configFactoryDefault",
            self.xMotor.configFactoryDefault(constants.kConfigurationTimeoutLimit),
        ):
            return
        self.xMotor.setInverted(self.xMotorInverted)
        if not ctreCheckError(
            "config_kP",
            self.xMotor.config_kP(
                constants.kXDrivePIDSlot,
                constants.kXDrivePGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        if not ctreCheckError(
            "config_kI",
            self.xMotor.config_kP(
                constants.kXDrivePIDSlot,
                constants.kXDriveIGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        if not ctreCheckError(
            "config_kD",
            self.xMotor.config_kP(
                constants.kXDrivePIDSlot,
                constants.kXDriveDGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        if not ctreCheckError(
            "config_SupplyLim",
            self.xMotor.configSupplyCurrentLimit(
                constants.kDriveSupplyCurrentLimitConfiguration,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        DataLogManager.log("   ... Done")

        DataLogManager.log(
            f"   Configuring Y motor: CAN ID: {self.yMotor.getDeviceID()}"
        )
        if not ctreCheckError(
            "configFactoryDefault",
            self.yMotor.configFactoryDefault(constants.kConfigurationTimeoutLimit),
        ):
            return
        self.yMotor.setInverted(self.yMotorInverted)
        if not ctreCheckError(
            "config_kP",
            self.yMotor.config_kP(
                constants.kYDrivePIDSlot,
                constants.kYDrivePGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        if not ctreCheckError(
            "config_kI",
            self.yMotor.config_kP(
                constants.kYDrivePIDSlot,
                constants.kYDriveIGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        if not ctreCheckError(
            "config_kD",
            self.yMotor.config_kP(
                constants.kYDrivePIDSlot,
                constants.kYDriveDGain,
                constants.kConfigurationTimeoutLimit,
            ),
        ):
            return
        DataLogManager.log("   ... Done")

        DataLogManager.log("... Done")

    def getBallLinearVelocity(self) -> Translation2d:
        xEncoderPulsesPerSecond = (
            self.xMotor.getSelectedSensorVelocity()
            * constants.k100MillisecondsPerSecond
        )
        yEncoderPulsesPerSecond = (
            self.yMotor.getSelectedSensorVelocity()
            * constants.k100MillisecondsPerSecond
        )
        return Translation2d(
            xEncoderPulsesPerSecond / constants.kBallEncoderPulsesPerMeter,
            yEncoderPulsesPerSecond / constants.kBallEncoderPulsesPerMeter,
        )

    def getTotalXPosition(self) -> float:
        xEncoderPulses = self.xMotor.getSelectedSensorPosition()
        distance = (
            xEncoderPulses
            / constants.kBallEncoderPulsesPerRadian
            * constants.kBallRadius
        )
        return distance

    def getTotalYPosition(self) -> float:
        yEncoderPulses = self.yMotor.getSelectedSensorPosition()
        distance = (
            yEncoderPulses
            / constants.kBallEncoderPulsesPerRadian
            * constants.kBallRadius
        )
        return distance

    def setBallLinearVelocityTarget(
        self, wheelLinearVelocityTarget: Translation2d
    ) -> None:
        xEncoderPulsesPerSecond = (
            wheelLinearVelocityTarget.X() * constants.kBallEncoderPulsesPerMeter
        )
        self.xMotor.set(
            ControlMode.Velocity,
            xEncoderPulsesPerSecond / constants.k100MillisecondsPerSecond,
        )

        yEncoderPulsesPerSecond = (
            wheelLinearVelocityTarget.X() * constants.kBallEncoderPulsesPerMeter
        )
        self.yMotor.set(
            ControlMode.Velocity,
            yEncoderPulsesPerSecond / constants.k100MillisecondsPerSecond,
        )

    def reset(self) -> None:
        pass


class DriveSubsystem(SubsystemBase):
    class CoordinateMode(Enum):
        RobotRelative = auto()
        FieldRelative = auto()
        TargetRelative = auto()

    def __init__(self) -> None:
        SubsystemBase.__init__(self)
        self.setName(__class__.__name__)
        SmartDashboard.putBoolean(constants.kRobotPoseArrayKeys.validKey, False)

        self.rotationOffset = 0

        if RobotBase.isReal():
            self.frontLeftModule = CTREBallModule(
                constants.kFrontLeftModuleName,
                WPI_TalonFX(constants.kFrontLeftDriveMotorId, constants.kCANivoreName),
                constants.kFrontLeftXDriveInverted,
                WPI_TalonFX(constants.kFrontLeftSteerMotorId, constants.kCANivoreName),
                constants.kFrontLeftYDriveInverted,
            )
            self.frontRightModule = CTREBallModule(
                constants.kFrontRightModuleName,
                WPI_TalonFX(constants.kFrontRightDriveMotorId, constants.kCANivoreName),
                constants.kFrontRightXDriveInverted,
                WPI_TalonFX(constants.kFrontRightSteerMotorId, constants.kCANivoreName),
                constants.kFrontRightYDriveInverted,
            )
            self.backLeftModule = CTREBallModule(
                constants.kBackLeftModuleName,
                WPI_TalonFX(constants.kBackLeftDriveMotorId, constants.kCANivoreName),
                constants.kBackLeftXDriveInverted,
                WPI_TalonFX(constants.kBackLeftSteerMotorId, constants.kCANivoreName),
                constants.kBackLeftYDriveInverted,
            )
            self.backRightModule = CTREBallModule(
                constants.kBackRightModuleName,
                WPI_TalonFX(constants.kBackRightDriveMotorId, constants.kCANivoreName),
                constants.kBackRightXDriveInverted,
                WPI_TalonFX(constants.kBackRightSteerMotorId, constants.kCANivoreName),
                constants.kBackRightYDriveInverted,
            )
        else:
            self.frontLeftModule = PWMSwerveModule(
                constants.kFrontLeftModuleName,
                PWMVictorSPX(constants.kSimFrontLeftDriveMotorPort),
                PWMVictorSPX(constants.kSimFrontLeftSteerMotorPort),
                Encoder(*constants.kSimFrontLeftDriveEncoderPorts),
                Encoder(*constants.kSimFrontLeftSteerEncoderPorts),
            )
            self.frontRightModule = PWMSwerveModule(
                constants.kFrontRightModuleName,
                PWMVictorSPX(constants.kSimFrontRightDriveMotorPort),
                PWMVictorSPX(constants.kSimFrontRightSteerMotorPort),
                Encoder(*constants.kSimFrontRightDriveEncoderPorts),
                Encoder(*constants.kSimFrontRightSteerEncoderPorts),
            )
            self.backLeftModule = PWMSwerveModule(
                constants.kBackLeftModuleName,
                PWMVictorSPX(constants.kSimBackLeftDriveMotorPort),
                PWMVictorSPX(constants.kSimBackLeftSteerMotorPort),
                Encoder(*constants.kSimBackLeftDriveEncoderPorts),
                Encoder(*constants.kSimBackLeftSteerEncoderPorts),
            )
            self.backRightModule = PWMSwerveModule(
                constants.kBackRightModuleName,
                PWMVictorSPX(constants.kSimBackRightDriveMotorPort),
                PWMVictorSPX(constants.kSimBackRightSteerMotorPort),
                Encoder(*constants.kSimBackRightDriveEncoderPorts),
                Encoder(*constants.kSimBackRightSteerEncoderPorts),
            )

        self.modules = (
            self.frontLeftModule,
            self.frontRightModule,
            self.backLeftModule,
            self.backRightModule,
        )

        self.kinematics = BallDriveKinematics(
            [
                constants.kFrontLeftWheelPosition,
                constants.kFrontRightWheelPosition,
                constants.kBackLeftWheelPosition,
                constants.kBackRightWheelPosition,
            ],
        )

        # Create the gyro, a sensor which can indicate the heading of the robot relative
        # to a customizable position.
        self.gyro = AHRS.create_spi()

        # Create the an object for our odometry, which will utilize sensor data to
        # keep a record of our position on the field.
        self.odometry = BallDriveOdometry(
            self.kinematics,
            self.getRotation(),
            [
                self.frontLeftModule.getPosition(),
                self.frontRightModule.getPosition(),
                self.backLeftModule.getPosition(),
                self.backRightModule.getPosition(),
            ],
            Pose2d(),
        )
        self.printTimer = Timer()
        self.vxLimiter = SlewRateLimiter(constants.kDriveAccelLimit)
        self.vyLimiter = SlewRateLimiter(constants.kDriveAccelLimit)

        self.visionEstimate = Pose2d()

    def resetSwerveModules(self):
        for module in self.modules:
            module.reset()
        self.resetGyro(Pose2d())

    def setOdometryPosition(self, pose: Pose2d):
        # self.gyro.setAngleAdjustment(pose.rotation().degrees())
        self.rotationOffset = pose.rotation().degrees()
        self.resetOdometryAtPosition(pose)

    def resetGyro(self, pose: Pose2d):
        self.gyro.reset()
        # self.gyro.setAngleAdjustment(pose.rotation().degrees())
        self.rotationOffset = pose.rotation().degrees()
        self.resetOdometryAtPosition(pose)

    def getPose(self) -> Pose2d:
        translation = self.odometry.getPose().translation()
        rotation = self.getRotation()
        return Pose2d(translation, rotation)

    def applyStates(self, moduleStates: List[BallDriveState]) -> None:
        (
            frontLeftState,
            frontRightState,
            backLeftState,
            backRightState,
        ) = BallDriveKinematics.desaturateWheelSpeeds(
            moduleStates, constants.kMaxBallLinearVelocity
        )

        # SmartDashboard.putNumberArray(
        #     constants.kSwerveExpectedStatesKey,
        #     [
        #         frontLeftState.angle.degrees(),
        #         frontLeftState.speed,
        #         frontRightState.angle.degrees(),
        #         frontRightState.speed,
        #         backLeftState.angle.degrees(),
        #         backLeftState.speed,
        #         backRightState.angle.degrees(),
        #         backRightState.speed,
        #     ],
        # )
        self.frontLeftModule.applyState(frontLeftState)
        self.frontRightModule.applyState(frontRightState)
        self.backLeftModule.applyState(backLeftState)
        self.backRightModule.applyState(backRightState)

    def getRotation(self) -> Rotation2d:
        return Rotation2d.fromDegrees(
            ((self.gyro.getRotation2d().degrees() / 0.98801) % 360)
            + self.rotationOffset
        )

    def getPitch(self) -> Rotation2d:
        return Rotation2d.fromDegrees(-self.gyro.getPitch() + 180)

    def resetOdometryAtPosition(self, pose: Pose2d):
        self.odometry.resetPosition(
            self.getRotation(),
            [self.frontLeftModule.getPosition(),
            self.frontRightModule.getPosition(),
            self.backLeftModule.getPosition(),
            self.backRightModule.getPosition()],
            pose,
        )

    def periodic(self):
        """
        Called periodically when it can be called. Updates the robot's
        odometry with sensor data.
        """

        pastPose = self.odometry.getPose()

        self.odometry.update(
            self.getRotation(),
            [self.frontLeftModule.getPosition(),
            self.frontRightModule.getPosition(),
            self.backLeftModule.getPosition(),
            self.backRightModule.getPosition()]
        )
        robotPose = self.getPose()

        deltaPose = robotPose - pastPose
        # SmartDashboard.putNumberArray(
        #     constants.kSwerveActualStatesKey,
        #     [
        #         self.frontLeftModule.getSwerveAngle().degrees(),
        #         self.frontLeftModule.getWheelLinearVelocity(),
        #         self.frontRightModule.getSwerveAngle().degrees(),
        #         self.frontRightModule.getWheelLinearVelocity(),
        #         self.backLeftModule.getSwerveAngle().degrees(),
        #         self.backLeftModule.getWheelLinearVelocity(),
        #         self.backRightModule.getSwerveAngle().degrees(),
        #         self.backRightModule.getWheelLinearVelocity(),
        #     ],
        # )
        SmartDashboard.putNumberArray(
            constants.kDriveVelocityKeys,
            [
                deltaPose.X()
                / constants.kRobotUpdatePeriod,  # velocity is delta pose / delta time
                deltaPose.Y() / constants.kRobotUpdatePeriod,
                deltaPose.rotation().radians() / constants.kRobotUpdatePeriod,
            ],
        )

        robotPoseArray = [robotPose.X(), robotPose.Y(), robotPose.rotation().radians()]

        if SmartDashboard.getBoolean(
            constants.kRobotVisionPoseArrayKeys.validKey, False
        ):
            visionPose = self.visionEstimate

            weightedPose = Pose2d(
                visionPose.X() * constants.kRobotVisionPoseWeight
                + robotPose.X() * (1 - constants.kRobotVisionPoseWeight),
                visionPose.Y() * constants.kRobotVisionPoseWeight
                + robotPose.Y() * (1 - constants.kRobotVisionPoseWeight),
                robotPose.rotation(),
            )
            self.resetOdometryAtPosition(weightedPose)

        SmartDashboard.putNumberArray(
            constants.kRobotPoseArrayKeys.valueKey, robotPoseArray
        )
        SmartDashboard.putBoolean(constants.kRobotPoseArrayKeys.validKey, True)

    def arcadeDriveWithFactors(
        self,
        forwardSpeedFactor: float,
        sidewaysSpeedFactor: float,
        rotationSpeedFactor: float,
        coordinateMode: CoordinateMode,
    ) -> None:
        """
        Drives the robot using arcade controls.

        :param forwardSpeedFactor: the commanded forward movement
        :param sidewaysSpeedFactor: the commanded sideways movement
        :param rotationSpeedFactor: the commanded rotation
        """

        forwardSpeedFactor = convenientmath.clamp(forwardSpeedFactor, -1, 1)
        sidewaysSpeedFactor = convenientmath.clamp(sidewaysSpeedFactor, -1, 1)
        rotationSpeedFactor = convenientmath.clamp(rotationSpeedFactor, -1, 1)

        combinedLinearFactor = Translation2d(
            forwardSpeedFactor, sidewaysSpeedFactor
        ).norm()

        # prevent combined forward & sideways inputs from exceeding the max linear velocity
        if combinedLinearFactor > 1.0:
            forwardSpeedFactor = forwardSpeedFactor / combinedLinearFactor
            sidewaysSpeedFactor = sidewaysSpeedFactor / combinedLinearFactor

        chassisSpeeds = ChassisSpeeds(
            forwardSpeedFactor * constants.kMaxForwardLinearVelocity,
            sidewaysSpeedFactor * constants.kMaxSidewaysLinearVelocity,
            rotationSpeedFactor * constants.kMaxRotationAngularVelocity,
        )

        self.arcadeDriveWithSpeeds(chassisSpeeds, coordinateMode)

    def arcadeDriveWithSpeeds(
        self, chassisSpeeds: ChassisSpeeds, coordinateMode: CoordinateMode
    ) -> None:
        targetAngle = Rotation2d(
            SmartDashboard.getNumber(
                constants.kTargetAngleRelativeToRobotKeys.valueKey, 0
            )
        )

        robotChassisSpeeds = None
        if coordinateMode is DriveSubsystem.CoordinateMode.RobotRelative:
            robotChassisSpeeds = chassisSpeeds
        elif coordinateMode is DriveSubsystem.CoordinateMode.FieldRelative:
            robotChassisSpeeds = ChassisSpeeds.fromFieldRelativeSpeeds(
                chassisSpeeds.vx,
                chassisSpeeds.vy,
                chassisSpeeds.omega,
                self.getRotation(),
            )
        elif coordinateMode is DriveSubsystem.CoordinateMode.TargetRelative:
            if SmartDashboard.getBoolean(
                constants.kTargetAngleRelativeToRobotKeys.validKey, False
            ):
                robotSpeeds = Translation2d(chassisSpeeds.vx, chassisSpeeds.vy)
                targetAlignedSpeeds = robotSpeeds.rotateBy(targetAngle)
                robotChassisSpeeds = ChassisSpeeds(
                    targetAlignedSpeeds.X(),
                    targetAlignedSpeeds.Y(),
                    chassisSpeeds.omega,
                )
            else:
                robotChassisSpeeds = ChassisSpeeds()

        moduleStates = self.kinematics.toSwerveModuleStates(robotChassisSpeeds)
        self.applyStates(moduleStates)
