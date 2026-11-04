#!/usr/bin/env python

import argparse
import math
import sys

import rospy
from nav_msgs.msg import Odometry


def parse_args():
    parser = argparse.ArgumentParser(description="Relay Qualisys odometry to MAVROS odometry with explicit frame ids.")
    parser.add_argument("--source-topic", required=True)
    parser.add_argument("--target-topic", required=True)
    parser.add_argument("--map-frame", required=True)
    parser.add_argument("--base-link-frame", required=True)
    return parser.parse_args()


def is_finite(value):
    return not math.isnan(value) and not math.isinf(value)


def require_finite_scalar(value, name):
    if not is_finite(value):
        raise RuntimeError("%s is not finite: %r" % (name, value))


def require_finite_array(values, name):
    for index, value in enumerate(values):
        require_finite_scalar(value, "%s[%d]" % (name, index))


def quaternion_norm_squared(quaternion):
    return (
        quaternion.w * quaternion.w
        + quaternion.x * quaternion.x
        + quaternion.y * quaternion.y
        + quaternion.z * quaternion.z
    )


class QualisysOdomRelay(object):
    def __init__(self, source_topic, target_topic, map_frame, base_link_frame):
        self._map_frame = map_frame
        self._base_link_frame = base_link_frame
        self._publisher = rospy.Publisher(target_topic, Odometry, queue_size=10)
        self._subscriber = rospy.Subscriber(source_topic, Odometry, self._callback, queue_size=10)

    def _callback(self, msg):
        try:
            self._validate(msg)
            relay_msg = Odometry()
            relay_msg.header.seq = msg.header.seq
            relay_msg.header.stamp = msg.header.stamp
            relay_msg.header.frame_id = self._map_frame
            relay_msg.child_frame_id = self._base_link_frame
            relay_msg.pose = msg.pose
            relay_msg.twist = msg.twist
            self._publisher.publish(relay_msg)
        except Exception as exc:
            rospy.logfatal("qualisys odometry relay failed: %s", exc)
            rospy.signal_shutdown(str(exc))
            raise

    def _validate(self, msg):
        require_finite_scalar(msg.header.stamp.to_sec(), "header.stamp")

        require_finite_scalar(msg.pose.pose.position.x, "pose.position.x")
        require_finite_scalar(msg.pose.pose.position.y, "pose.position.y")
        require_finite_scalar(msg.pose.pose.position.z, "pose.position.z")

        require_finite_scalar(msg.pose.pose.orientation.w, "pose.orientation.w")
        require_finite_scalar(msg.pose.pose.orientation.x, "pose.orientation.x")
        require_finite_scalar(msg.pose.pose.orientation.y, "pose.orientation.y")
        require_finite_scalar(msg.pose.pose.orientation.z, "pose.orientation.z")

        norm_sq = quaternion_norm_squared(msg.pose.pose.orientation)
        require_finite_scalar(norm_sq, "pose.orientation.norm_sq")
        if abs(norm_sq - 1.0) > 1e-3:
            raise RuntimeError("pose orientation quaternion is not normalized: norm_sq=%r" % norm_sq)

        require_finite_scalar(msg.twist.twist.linear.x, "twist.linear.x")
        require_finite_scalar(msg.twist.twist.linear.y, "twist.linear.y")
        require_finite_scalar(msg.twist.twist.linear.z, "twist.linear.z")

        require_finite_scalar(msg.twist.twist.angular.x, "twist.angular.x")
        require_finite_scalar(msg.twist.twist.angular.y, "twist.angular.y")
        require_finite_scalar(msg.twist.twist.angular.z, "twist.angular.z")

        require_finite_array(msg.pose.covariance, "pose.covariance")
        require_finite_array(msg.twist.covariance, "twist.covariance")


def main():
    args = parse_args()
    rospy.init_node("qualisys_odom_to_mavros_odometry", anonymous=False)
    QualisysOdomRelay(
        source_topic=args.source_topic,
        target_topic=args.target_topic,
        map_frame=args.map_frame,
        base_link_frame=args.base_link_frame,
    )
    rospy.loginfo(
        "relaying %s to %s with frame_id=%s child_frame_id=%s",
        args.source_topic,
        args.target_topic,
        args.map_frame,
        args.base_link_frame,
    )
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("qualisys_odom_to_mavros_odometry failed: %s\n" % exc)
        raise
