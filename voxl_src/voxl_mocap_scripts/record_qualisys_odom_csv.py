#!/usr/bin/env python

import argparse
import csv
import sys

import rospy
from nav_msgs.msg import Odometry


RAW_HEADER = [
    "timestamp_ns",
    "position_x",
    "position_y",
    "position_z",
    "orientation_w",
    "orientation_x",
    "orientation_y",
    "orientation_z",
    "linear_velocity_x",
    "linear_velocity_y",
    "linear_velocity_z",
    "angular_velocity_x",
    "angular_velocity_y",
    "angular_velocity_z",
]
RAW_HEADER.extend(["pose_covariance_%02d" % index for index in range(36)])
RAW_HEADER.extend(["twist_covariance_%02d" % index for index in range(36)])


def parse_args():
    parser = argparse.ArgumentParser(description="Record a raw Qualisys nav_msgs/Odometry topic to a strict CSV format.")
    parser.add_argument("--source-topic", required=True)
    parser.add_argument("--output-csv", required=True)
    return parser.parse_args()


def open_csv_handle(path):
    if sys.version_info[0] < 3:
        return open(path, "wb")
    return open(path, "w", newline="")


class QualisysOdomCsvRecorder(object):
    def __init__(self, source_topic, output_csv):
        self._output_handle = open_csv_handle(output_csv)
        self._writer = csv.writer(self._output_handle)
        self._writer.writerow(RAW_HEADER)
        self._output_handle.flush()
        self._subscriber = rospy.Subscriber(source_topic, Odometry, self._callback, queue_size=200)

    def _callback(self, msg):
        row = [
            msg.header.stamp.to_nsec(),
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z,
            msg.twist.twist.angular.x,
            msg.twist.twist.angular.y,
            msg.twist.twist.angular.z,
        ]
        row.extend(msg.pose.covariance)
        row.extend(msg.twist.covariance)
        self._writer.writerow(row)
        self._output_handle.flush()

    def close(self):
        self._output_handle.close()


def main():
    args = parse_args()
    rospy.init_node("record_qualisys_odom_csv", anonymous=False)
    recorder = QualisysOdomCsvRecorder(args.source_topic, args.output_csv)
    rospy.on_shutdown(recorder.close)
    rospy.loginfo("recording raw Qualisys odometry from %s to %s", args.source_topic, args.output_csv)
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        sys.stderr.write("record_qualisys_odom_csv failed: %s\n" % exc)
        raise
