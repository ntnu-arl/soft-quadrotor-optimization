#!/usr/bin/env python

import argparse
import math
import struct
import sys

import rospy
from mavros_msgs.msg import Mavlink
from nav_msgs.msg import Odometry


MAVLINK_MSG_ID_ODOMETRY = 331
MAVLINK_MSG_ID_ODOMETRY_LEN = 233
MAVLINK_MSG_ID_ODOMETRY_CRC = 91
MAV_FRAME_LOCAL_NED = 1
MAV_ESTIMATOR_TYPE_VISION = 2

ENU_TO_NED = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, -1.0),
)

FLU_TO_FRD = (
    (1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, -1.0),
)

# Empirical world-NED linear velocity variances replacing the Qualisys-reported
# twist covariance. Measured against differentiated mocap positions on flight
# 2026_06_11_163114: horizontal twist noise is ~0.2 m/s broadband (the Qualisys
# default of 0.1 m/s is optimistic), vertical is ~0.008 m/s (0.02 m/s used as a
# conservative bound). PX4 fuses EV velocity with per-axis
# max(EKF2_EVV_NOISE^2, reported variance), so these values set the fusion
# weight; EKF2_EVV_NOISE must stay at or below the smallest sigma here.
EV_LINEAR_VELOCITY_VARIANCE_NED = (0.04, 0.04, 4.0e-4)


def parse_args():
    parser = argparse.ArgumentParser(description="Publish Qualisys odometry to PX4 as raw MAVLink ODOMETRY.")
    parser.add_argument("--source-topic", required=True)
    parser.add_argument("--target-topic", required=True)
    parser.add_argument("--system-id", required=True, type=int)
    parser.add_argument("--component-id", required=True, type=int)
    parser.add_argument("--quality", required=True, type=int)
    return parser.parse_args()


def is_finite(value):
    return not math.isnan(value) and not math.isinf(value)


def require_finite_scalar(value, name):
    if not is_finite(value):
        raise RuntimeError("%s is not finite: %r" % (name, value))


def require_finite_array(values, name):
    for index, value in enumerate(values):
        require_finite_scalar(value, "%s[%d]" % (name, index))


def matrix_vector_mul(matrix, vector):
    return [
        matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2],
        matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2],
        matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2],
    ]


def matrix_mul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    result = []
    for row in range(rows):
        out_row = []
        for col in range(cols):
            value = 0.0
            for idx in range(inner):
                value += a[row][idx] * b[idx][col]
            out_row.append(value)
        result.append(out_row)
    return result


def transpose(matrix):
    rows = len(matrix)
    cols = len(matrix[0])
    result = []
    for col in range(cols):
        out_row = []
        for row in range(rows):
            out_row.append(matrix[row][col])
        result.append(out_row)
    return result


def quaternion_norm_squared(quaternion_wxyz):
    return (
        quaternion_wxyz[0] * quaternion_wxyz[0]
        + quaternion_wxyz[1] * quaternion_wxyz[1]
        + quaternion_wxyz[2] * quaternion_wxyz[2]
        + quaternion_wxyz[3] * quaternion_wxyz[3]
    )


def quaternion_to_matrix(quaternion_wxyz):
    norm_sq = quaternion_norm_squared(quaternion_wxyz)
    require_finite_scalar(norm_sq, "quaternion_norm_sq")
    if abs(norm_sq - 1.0) > 1e-3:
        raise RuntimeError("source quaternion is not normalized: norm_sq=%r" % norm_sq)

    w = quaternion_wxyz[0]
    x = quaternion_wxyz[1]
    y = quaternion_wxyz[2]
    z = quaternion_wxyz[3]

    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def matrix_to_quaternion(matrix):
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = [
            0.25 * scale,
            (matrix[2][1] - matrix[1][2]) / scale,
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[1][0] - matrix[0][1]) / scale,
        ]
    elif matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        quaternion = [
            (matrix[2][1] - matrix[1][2]) / scale,
            0.25 * scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
        ]
    elif matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        quaternion = [
            (matrix[0][2] - matrix[2][0]) / scale,
            (matrix[0][1] + matrix[1][0]) / scale,
            0.25 * scale,
            (matrix[1][2] + matrix[2][1]) / scale,
        ]
    else:
        scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
        quaternion = [
            (matrix[1][0] - matrix[0][1]) / scale,
            (matrix[0][2] + matrix[2][0]) / scale,
            (matrix[1][2] + matrix[2][1]) / scale,
            0.25 * scale,
        ]

    norm_sq = quaternion_norm_squared(quaternion)
    require_finite_scalar(norm_sq, "converted_quaternion_norm_sq")
    if norm_sq <= 0.0:
        raise RuntimeError("converted quaternion norm is zero")

    norm = math.sqrt(norm_sq)
    return [value / norm for value in quaternion]


def ros_quaternion_to_wxyz(ros_quaternion):
    quaternion = [
        ros_quaternion.w,
        ros_quaternion.x,
        ros_quaternion.y,
        ros_quaternion.z,
    ]
    require_finite_array(quaternion, "pose.orientation")
    return quaternion


def enu_flu_quaternion_to_ned_frd(quaternion_wxyz):
    rotation_enu_flu = quaternion_to_matrix(quaternion_wxyz)
    rotation_ned_frd = matrix_mul(ENU_TO_NED, matrix_mul(rotation_enu_flu, FLU_TO_FRD))
    return matrix_to_quaternion(rotation_ned_frd)


def covariance36_to_matrix(covariance):
    require_finite_array(covariance, "covariance36")
    matrix = []
    index = 0
    for row in range(6):
        out_row = []
        for col in range(6):
            out_row.append(covariance[index])
            index += 1
        matrix.append(out_row)
    return matrix


def covariance_matrix_to_urt(covariance_matrix):
    urt = []
    for row in range(6):
        for col in range(row, 6):
            urt.append(covariance_matrix[row][col])
    return urt


def transform_covariance_matrix(covariance36, first_rotation, second_rotation):
    covariance_matrix = covariance36_to_matrix(covariance36)
    transform = [
        [first_rotation[0][0], first_rotation[0][1], first_rotation[0][2], 0.0, 0.0, 0.0],
        [first_rotation[1][0], first_rotation[1][1], first_rotation[1][2], 0.0, 0.0, 0.0],
        [first_rotation[2][0], first_rotation[2][1], first_rotation[2][2], 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, second_rotation[0][0], second_rotation[0][1], second_rotation[0][2]],
        [0.0, 0.0, 0.0, second_rotation[1][0], second_rotation[1][1], second_rotation[1][2]],
        [0.0, 0.0, 0.0, second_rotation[2][0], second_rotation[2][1], second_rotation[2][2]],
    ]
    return matrix_mul(transform, matrix_mul(covariance_matrix, transpose(transform)))


def pack_payload64(payload_bytes):
    payload = bytearray(payload_bytes)
    while len(payload) % 8 != 0:
        payload.append(0)
    word_count = len(payload) // 8
    packed = ''.join(chr(value) for value in payload)
    return struct.unpack("<%dQ" % word_count, packed)


def trim_mavlink_v2_payload(payload_bytes):
    payload = bytearray(payload_bytes)
    while len(payload) > 1 and payload[-1] == 0:
        payload.pop()
    return ''.join(chr(value) for value in payload)


def x25_crc_accumulate(byte, crc):
    tmp = byte ^ (crc & 0xff)
    tmp = (tmp ^ (tmp << 4)) & 0xff
    crc = ((crc >> 8) ^ (tmp << 8) ^ (tmp << 3) ^ (tmp >> 4)) & 0xffff
    return crc


def mavlink_crc(payload, seq, system_id, component_id):
    payload_length = len(payload)
    crc = 0xffff
    header_without_magic = [
        payload_length,
        0,
        0,
        seq,
        system_id,
        component_id,
        MAVLINK_MSG_ID_ODOMETRY & 0xff,
        (MAVLINK_MSG_ID_ODOMETRY >> 8) & 0xff,
        (MAVLINK_MSG_ID_ODOMETRY >> 16) & 0xff,
    ]
    for value in header_without_magic:
        crc = x25_crc_accumulate(value, crc)
    for byte in bytearray(payload):
        crc = x25_crc_accumulate(byte, crc)
    crc = x25_crc_accumulate(MAVLINK_MSG_ID_ODOMETRY_CRC, crc)
    return crc


class QualisysOdomToPx4Mavlink(object):
    def __init__(self, source_topic, target_topic, system_id, component_id, quality):
        self._system_id = system_id
        self._component_id = component_id
        self._quality = quality
        self._sequence = 0
        self._publisher = rospy.Publisher(target_topic, Mavlink, queue_size=10)
        self._subscriber = rospy.Subscriber(source_topic, Odometry, self._callback, queue_size=10)

    def _callback(self, msg):
        try:
            mavlink_message = self._convert(msg)
            self._publisher.publish(mavlink_message)
            self._sequence = (self._sequence + 1) % 256
        except Exception as exc:
            rospy.logfatal("qualisys odometry to px4 MAVLink failed: %s", exc)
            rospy.signal_shutdown(str(exc))
            raise

    def _convert(self, msg):
        require_finite_scalar(msg.header.stamp.to_sec(), "header.stamp")
        message_timestamp_usec = self._measurement_time_usec(msg)

        position_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        linear_velocity_twist = [
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
        ]
        angular_velocity_flu = [
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]

        require_finite_array(position_enu, "pose.position")
        require_finite_array(linear_velocity_twist, "twist.linear")
        require_finite_array(angular_velocity_flu, "twist.angular")

        rotation_enu_raw = quaternion_to_matrix(ros_quaternion_to_wxyz(msg.pose.pose.orientation))
        rotation_ned_frd = matrix_mul(ENU_TO_NED, matrix_mul(rotation_enu_raw, FLU_TO_FRD))
        quaternion_ned_frd = matrix_to_quaternion(rotation_ned_frd)
        position_ned = matrix_vector_mul(ENU_TO_NED, position_enu)
        # The Qualisys driver publishes twist.linear as R_body_to_world @ v_world (neither
        # plain world ENU nor the ROS body-frame convention), validated against
        # differentiated mocap positions on flight 2026_06_11_125731. World velocity is
        # therefore recovered with the transpose of the published orientation and sent to
        # PX4 in world NED so the EKF fuses it without involving its own attitude.
        twist_to_ned = matrix_mul(ENU_TO_NED, transpose(rotation_enu_raw))
        linear_velocity_ned = matrix_vector_mul(twist_to_ned, linear_velocity_twist)
        angular_velocity_frd = matrix_vector_mul(FLU_TO_FRD, angular_velocity_flu)

        pose_covariance = covariance_matrix_to_urt(
            transform_covariance_matrix(
                msg.pose.covariance,
                ENU_TO_NED,
                FLU_TO_FRD,
            )
        )
        velocity_covariance_matrix = transform_covariance_matrix(msg.twist.covariance, twist_to_ned, FLU_TO_FRD)
        for axis in range(3):
            for col in range(6):
                velocity_covariance_matrix[axis][col] = 0.0
                velocity_covariance_matrix[col][axis] = 0.0
            velocity_covariance_matrix[axis][axis] = EV_LINEAR_VELOCITY_VARIANCE_NED[axis]
        velocity_covariance = covariance_matrix_to_urt(velocity_covariance_matrix)

        full_payload = struct.pack(
            "<Qfff4fffffff21f21fBBBBb",
            message_timestamp_usec,
            position_ned[0],
            position_ned[1],
            position_ned[2],
            quaternion_ned_frd[0],
            quaternion_ned_frd[1],
            quaternion_ned_frd[2],
            quaternion_ned_frd[3],
            linear_velocity_ned[0],
            linear_velocity_ned[1],
            linear_velocity_ned[2],
            angular_velocity_frd[0],
            angular_velocity_frd[1],
            angular_velocity_frd[2],
            *(pose_covariance + velocity_covariance + [
                MAV_FRAME_LOCAL_NED,
                MAV_FRAME_LOCAL_NED,
                0,
                MAV_ESTIMATOR_TYPE_VISION,
                self._quality,
            ])
        )

        if len(full_payload) != MAVLINK_MSG_ID_ODOMETRY_LEN:
            raise RuntimeError("unexpected ODOMETRY payload length: %d" % len(full_payload))

        payload = trim_mavlink_v2_payload(full_payload)

        rosmsg = Mavlink()
        rosmsg.header.stamp = msg.header.stamp
        rosmsg.framing_status = Mavlink.FRAMING_OK
        rosmsg.magic = Mavlink.MAVLINK_V20
        rosmsg.len = len(payload)
        rosmsg.incompat_flags = 0
        rosmsg.compat_flags = 0
        rosmsg.seq = self._sequence
        rosmsg.sysid = self._system_id
        rosmsg.compid = self._component_id
        rosmsg.msgid = MAVLINK_MSG_ID_ODOMETRY
        rosmsg.checksum = mavlink_crc(payload, self._sequence, self._system_id, self._component_id)
        rosmsg.payload64 = pack_payload64(payload)
        rosmsg.signature = []
        return rosmsg

    def _measurement_time_usec(self, msg):
        # MAVROS TIMESYNC in MAVLINK mode exchanges ros::Time::now() values with PX4,
        # so ODOMETRY.time_usec must stay in that same remote ROS/system clock domain.
        return msg.header.stamp.to_nsec() // 1000


def main():
    args = parse_args()
    if args.quality < -1 or args.quality > 100:
        raise RuntimeError("quality must be in [-1, 100], got %r" % args.quality)
    if args.system_id < 0 or args.system_id > 255:
        raise RuntimeError("system-id must be in [0, 255], got %r" % args.system_id)
    if args.component_id < 0 or args.component_id > 255:
        raise RuntimeError("component-id must be in [0, 255], got %r" % args.component_id)

    rospy.init_node("qualisys_odom_to_px4_mavlink", anonymous=False)
    QualisysOdomToPx4Mavlink(
        source_topic=args.source_topic,
        target_topic=args.target_topic,
        system_id=args.system_id,
        component_id=args.component_id,
        quality=args.quality,
    )
    rospy.loginfo(
        "publishing %s to %s as MAVLink ODOMETRY with system_id=%d component_id=%d quality=%d",
        args.source_topic,
        args.target_topic,
        args.system_id,
        args.component_id,
        args.quality,
    )
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("qualisys_odom_to_px4_mavlink failed: %s\n" % exc)
        raise
