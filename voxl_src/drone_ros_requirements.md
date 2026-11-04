# Drone ROS/MAVROS Requirements

Generated from read-only inspection of the current drone over `adb shell` on
2026-05-07. No changes were made on the drone.

## Current Drone Identity

- ADB device: `51326bba`
- Hostname: `m0104`
- User used for shell: `root`
- Hardware platform: ModalAI VOXL 2 / QRB5165
- `voxl-version` platform: `M0104`
- SKU: `TF-M0104-6-V1-C6`
- System image: `1.8.04-M0104-14.1a-perf`
- Kernel: `Linux m0104 4.19.125 #1 SMP PREEMPT Mon Mar 24 21:03:36 UTC 2025 aarch64`
- VOXL suite: `1.5.0`
- Ubuntu: `18.04.5 LTS (Bionic Beaver)`
- Architecture: `arm64` / `aarch64`

## Required Base OS and Repositories

The new drone should use the same ModalAI/Ubuntu base image, not a newer Ubuntu
or ROS distribution.

Required OS:

```text
Ubuntu 18.04.5 LTS (Bionic Beaver)
VERSION_ID=18.04
VERSION_CODENAME=bionic
Architecture: arm64
```

Required APT repositories observed:

```text
deb http://ports.ubuntu.com/ubuntu-ports/ bionic main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ bionic universe
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates universe
deb http://ports.ubuntu.com/ubuntu-ports/ bionic multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-updates multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-backports main restricted universe multiverse
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security main restricted
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security universe
deb http://ports.ubuntu.com/ubuntu-ports/ bionic-security multiverse
deb [trusted=yes] http://voxl-packages.modalai.com/ ./dists/qrb5165/sdk-1.5/binary-arm64/
deb [trusted=yes] file:/data/voxl-suite-offline-packages ./
deb http://packages.ros.org/ros/ubuntu bionic main
```

## Required VOXL Packages

The current drone is on `voxl-suite 1.5.0`. To match the ROS/MAVROS behavior in
context, the new drone should start from the same VOXL image and suite version.

Important observed VOXL package versions:

```text
voxl-suite                  1.5.0
voxl-px4                    1.14.0-2.0.105
voxl-px4-params             0.7.5
voxl-mavlink                0.1.5
voxl-mavlink-server         1.4.9
voxl-vision-hub             1.8.23
voxl-mpa-to-ros             0.3.9
voxl-mpa-tools              1.4.0
voxl-qvio-server            1.2.0
voxl-open-vins              0.4.19
voxl-open-vins-server       0.3.12
voxl-camera-server          2.2.4
voxl-imu-server package     qrb5165-imu-server 1.1.3
voxl-cpu-monitor            0.6.0
voxl-portal                 0.7.11
voxl-utils                  1.4.8
```

Observed enabled/running services relevant to flight stack:

```text
Enabled:
voxl-cpu-monitor.service
voxl-imu-server.service
voxl-portal.service
voxl-px4.service
voxl-vision-hub.service
voxl-wait-for-fs.service

Running at inspection time:
voxl-cpu-monitor.service
voxl-imu-server.service
voxl-portal.service
voxl-px4.service
voxl-vision-hub.service
```

No `roscore`, `roslaunch`, `rosmaster`, `rosout`, or `mavros_node` process was
running at inspection time.

## ROS Requirement

The current drone uses ROS 1 Melodic from `/opt/ros/melodic`.

Required ROS environment after sourcing `/opt/ros/melodic/setup.bash`:

```text
ROS_DISTRO=melodic
ROS_VERSION=1
ROS_PYTHON_VERSION=2
ROS_PACKAGE_PATH=/opt/ros/melodic/share
CMAKE_PREFIX_PATH=/opt/ros/melodic
PYTHONPATH=/opt/ros/melodic/lib/python2.7/dist-packages
LD_LIBRARY_PATH=/opt/ros/melodic/lib
ROS_ETC_DIR=/opt/ros/melodic/etc/ros
ROS_ROOT=/opt/ros/melodic/share/ros
```

Observed ROS version command results:

```text
rosversion -d -> melodic
Python -> 2.7.17
```

Required ROS shell setup in `/home/root/.bashrc`:

```bash
export ROS_DISTRO=melodic
source /opt/ros/melodic/setup.bash
source /home/root/catkin_ws/devel/setup.bash

unset ROS_HOSTNAME
export ROS_MASTER_URI=http://localhost:11311
export ROS_IP=192.168.5.5
```

The fixed `ROS_IP=192.168.5.5` is part of the current setup and should be
replicated only if the new drone will use the same network addressing.

## MAVROS Requirement

MAVROS is installed from ROS APT packages, not from a source checkout.

Required MAVROS versions:

```text
rosversion mavros        -> 1.16.0
rosversion mavros_extras -> 1.16.0
```

Required Debian packages:

```text
ros-melodic-libmavconn       1.16.0-1bionic.20230605.154504 arm64
ros-melodic-mavlink          2022.12.30-1bionic.20230605.154004 arm64
ros-melodic-mavros           1.16.0-1bionic.20230605.155036 arm64
ros-melodic-mavros-extras    1.16.0-1bionic.20230605.160238 arm64
ros-melodic-mavros-msgs      1.16.0-1bionic.20230505.180431 arm64
```

Required MAVROS package locations:

```text
/opt/ros/melodic/share/mavros
/opt/ros/melodic/share/mavros_extras
/opt/ros/melodic/share/mavros_msgs
/opt/ros/melodic/lib/mavros/mavros_node
```

Required MAVROS launch files present in `/opt/ros/melodic/share/mavros/launch`:

```text
apm.launch
apm2.launch
apm_config.yaml
apm_pluginlists.yaml
event_launcher.yaml
mavlink_bridge.launch
node.launch
px4.launch
px4_config.yaml
px4_pluginlists.yaml
```

## GeographicLib Requirement

MAVROS GeographicLib support is installed and datasets are present.

Required packages/data:

```text
geographiclib-tools 1.49-2 arm64
libgeographic-dev   1.49-2 arm64
libgeographic17     1.49-2 arm64
```

Required dataset files:

```text
/usr/share/GeographicLib/geoids/egm96-5.pgm
/usr/share/GeographicLib/geoids/egm96-5.pgm.aux.xml
/usr/share/GeographicLib/geoids/egm96-5.wld
/usr/share/GeographicLib/gravity/egm96.egm
/usr/share/GeographicLib/gravity/egm96.egm.cof
/usr/share/GeographicLib/magnetic/emm2015.wmm
/usr/share/GeographicLib/magnetic/emm2015.wmm.cof
```

If these files are absent on the new drone after installing MAVROS, run the
MAVROS dataset installer during provisioning:

```bash
rosrun mavros install_geographiclib_datasets.sh
```

## ROS Build Tools

Observed development/build support packages:

```text
build-essential       12.4ubuntu1 arm64
cmake                 3.10.2-1ubuntu2.18.04.2 arm64
python-rosdep         0.23.1-1 all
python-rosinstall     0.7.8-1 all
python-vcstools       0.1.42-1 all
python-wstool         0.1.17-1 all
```

Observed rosdep source:

```text
/etc/ros/rosdep/sources.list.d/20-default.list
```

Observed `rosdep --version`:

```text
0.23.1
```

## Installed ROS Package Set

Install the same ROS APT packages and versions for exact compatibility:

```text
ros-melodic-actionlib                 1.12.1-1bionic.20221025.194012 arm64
ros-melodic-actionlib-msgs            1.12.8-1bionic.20221025.175455 arm64
ros-melodic-bond                      1.8.5-1bionic.20221025.175658 arm64
ros-melodic-bond-core                 1.8.5-1bionic.20221025.185611 arm64
ros-melodic-bondcpp                   1.8.5-1bionic.20221025.184439 arm64
ros-melodic-bondpy                    1.8.5-1bionic.20221025.185149 arm64
ros-melodic-catkin                    0.7.29-1bionic.20201014.191647 arm64
ros-melodic-class-loader              0.4.1-0bionic.20200802.114445 arm64
ros-melodic-cmake-modules             0.4.2-0bionic.20200802.113658 arm64
ros-melodic-common-msgs               1.12.8-1bionic.20221025.224028 arm64
ros-melodic-cpp-common                0.6.14-1bionic.20200801.064627 arm64
ros-melodic-diagnostic-msgs           1.12.8-1bionic.20221025.180100 arm64
ros-melodic-diagnostic-updater        1.9.7-1bionic.20221025.191550 arm64
ros-melodic-dynamic-reconfigure       1.6.5-1bionic.20221025.203747 arm64
ros-melodic-eigen-conversions         1.12.1-1bionic.20221025.181629 arm64
ros-melodic-gencpp                    0.6.6-1bionic.20221025.174526 arm64
ros-melodic-geneus                    2.2.6-0bionic.20221025.174528 arm64
ros-melodic-genlisp                   0.4.16-0bionic.20221025.174632 arm64
ros-melodic-genmsg                    0.5.17-1bionic.20221025.174340 arm64
ros-melodic-gennodejs                 2.0.1-0bionic.20221025.174517 arm64
ros-melodic-genpy                     0.6.16-1bionic.20221025.174636 arm64
ros-melodic-geographic-msgs           0.5.3-0bionic.20221025.181608 arm64
ros-melodic-geometry-msgs             1.12.8-1bionic.20221025.180147 arm64
ros-melodic-libmavconn                1.16.0-1bionic.20230605.154504 arm64
ros-melodic-mavlink                   2022.12.30-1bionic.20230605.154004 arm64
ros-melodic-mavros                    1.16.0-1bionic.20230605.155036 arm64
ros-melodic-mavros-extras             1.16.0-1bionic.20230605.160238 arm64
ros-melodic-mavros-msgs               1.16.0-1bionic.20230505.180431 arm64
ros-melodic-message-filters           1.14.13-1bionic.20221025.190613 arm64
ros-melodic-message-generation        0.4.1-1bionic.20221025.174821 arm64
ros-melodic-message-runtime           0.4.12-0bionic.20221025.174748 arm64
ros-melodic-mk                        1.14.9-1bionic.20221025.175408 arm64
ros-melodic-nav-msgs                  1.12.8-1bionic.20221025.180311 arm64
ros-melodic-nodelet                   1.9.16-0bionic.20221025.184728 arm64
ros-melodic-nodelet-core              1.9.16-0bionic.20221025.211853 arm64
ros-melodic-nodelet-topic-tools       1.9.16-0bionic.20221025.211133 arm64
ros-melodic-orocos-kdl                1.4.0-0bionic.20201019.054953 arm64
ros-melodic-pluginlib                 1.12.1-0bionic.20221025.181818 arm64
ros-melodic-ros                       1.14.9-1bionic.20221025.181604 arm64
ros-melodic-ros-base                  1.4.1-0bionic.20230216.105845 arm64
ros-melodic-ros-comm                  1.14.13-1bionic.20230215.205555 arm64
ros-melodic-ros-core                  1.4.1-0bionic.20230216.104302 arm64
ros-melodic-ros-environment           1.2.3-1bionic.20200929.212812 arm64
ros-melodic-rosbag                    1.14.13-1bionic.20221025.192355 arm64
ros-melodic-rosbag-migration-rule     1.0.0-0bionic.20200802.115028 arm64
ros-melodic-rosbag-storage            1.14.13-1bionic.20221025.190608 arm64
ros-melodic-rosbash                   1.14.9-1bionic.20201019.055341 arm64
ros-melodic-rosboost-cfg              1.14.9-1bionic.20200802.045513 arm64
ros-melodic-rosbuild                  1.14.9-1bionic.20221025.175043 arm64
ros-melodic-rosclean                  1.14.9-1bionic.20200802.103628 arm64
ros-melodic-rosconsole                1.13.18-1bionic.20221025.175407 arm64
ros-melodic-rosconsole-bridge         0.5.3-0bionic.20221025.181841 arm64
ros-melodic-roscpp                    1.14.13-1bionic.20221025.182332 arm64
ros-melodic-roscpp-core               0.6.14-1bionic.20200802.160732 arm64
ros-melodic-roscpp-serialization      0.6.14-1bionic.20200802.155320 arm64
ros-melodic-roscpp-traits             0.6.14-1bionic.20200802.021535 arm64
ros-melodic-roscreate                 1.14.9-1bionic.20201019.120913 arm64
ros-melodic-rosgraph                  1.14.13-1bionic.20220127.142010 arm64
ros-melodic-rosgraph-msgs             1.11.2-0bionic.20221025.181405 arm64
ros-melodic-roslang                   1.14.9-1bionic.20221025.174523 arm64
ros-melodic-roslaunch                 1.14.13-1bionic.20221025.185550 arm64
ros-melodic-roslib                    1.14.9-1bionic.20201019.094326 arm64
ros-melodic-roslisp                   1.9.25-1bionic.20230215.205048 arm64
ros-melodic-roslz4                    1.14.13-1bionic.20220127.143341 arm64
ros-melodic-rosmake                   1.14.9-1bionic.20201019.054221 arm64
ros-melodic-rosmaster                 1.14.13-1bionic.20220127.142855 arm64
ros-melodic-rosmsg                    1.14.13-1bionic.20221025.192717 arm64
ros-melodic-rosnode                   1.14.13-1bionic.20221025.192946 arm64
ros-melodic-rosout                    1.14.13-1bionic.20221025.183811 arm64
ros-melodic-rospack                   2.5.6-1bionic.20200930.170738 arm64
ros-melodic-rosparam                  1.14.13-1bionic.20220127.143141 arm64
ros-melodic-rospy                     1.14.13-1bionic.20221025.183903 arm64
ros-melodic-rosservice                1.14.13-1bionic.20221025.192948 arm64
ros-melodic-rostest                   1.14.13-1bionic.20221025.190102 arm64
ros-melodic-rostime                   0.6.14-1bionic.20200802.014906 arm64
ros-melodic-rostopic                  1.14.13-1bionic.20221025.192756 arm64
ros-melodic-rosunit                   1.14.9-1bionic.20201019.120904 arm64
ros-melodic-roswtf                    1.14.13-1bionic.20221025.193646 arm64
ros-melodic-sensor-msgs               1.12.8-1bionic.20221025.192749 arm64
ros-melodic-shape-msgs                1.12.8-1bionic.20221025.180915 arm64
ros-melodic-smclib                    1.8.5-1bionic.20200802.045700 arm64
ros-melodic-std-msgs                  0.5.12-0bionic.20221025.175123 arm64
ros-melodic-std-srvs                  1.11.2-0bionic.20221025.175135 arm64
ros-melodic-stereo-msgs               1.12.8-1bionic.20221025.222633 arm64
ros-melodic-tf                        1.12.1-1bionic.20221025.195522 arm64
ros-melodic-tf2                       0.6.5-0bionic.20221025.181843 arm64
ros-melodic-tf2-eigen                 0.6.5-0bionic.20221025.182332 arm64
ros-melodic-tf2-msgs                  0.6.5-0bionic.20221025.180330 arm64
ros-melodic-tf2-py                    0.6.5-0bionic.20221025.185123 arm64
ros-melodic-tf2-ros                   0.6.5-0bionic.20221025.194946 arm64
ros-melodic-topic-tools               1.14.13-1bionic.20221025.191235 arm64
ros-melodic-trajectory-msgs           1.12.8-1bionic.20221025.180926 arm64
ros-melodic-urdf                      1.13.2-1bionic.20221025.190621 arm64
ros-melodic-uuid-msgs                 1.0.6-0bionic.20221025.181523 arm64
ros-melodic-visualization-msgs        1.12.8-1bionic.20221025.181059 arm64
ros-melodic-xmlrpcpp                  1.14.13-1bionic.20220127.143713 arm64
```

Also required from the VOXL ROS integration:

```text
voxl_mpa_to_ros is present in /opt/ros/melodic/share/voxl_mpa_to_ros
```

## Catkin Workspace Requirement

The current drone sources a root workspace:

```text
/home/root/catkin_ws
/home/root/catkin_ws/src
/home/root/catkin_ws/devel
/home/root/catkin_ws/build
```

Workspace package list:

```text
hexbug /home/root/catkin_ws/src/hexbug
```

After sourcing both setup files, `ROS_PACKAGE_PATH` is:

```text
/home/root/catkin_ws/src:/opt/ros/melodic/share
```

The `hexbug` package has build/runtime dependencies on:

```text
catkin
mavlink
mavros
```

The new drone must copy/recreate `/home/root/catkin_ws/src/hexbug`, then build
the workspace so `/home/root/catkin_ws/devel/setup.bash` exists.

Expected package files:

```text
/home/root/catkin_ws/src/hexbug/CMakeLists.txt
/home/root/catkin_ws/src/hexbug/package.xml
/home/root/catkin_ws/src/hexbug/config/px4.yaml
/home/root/catkin_ws/src/hexbug/config/px4_pluginlists.yaml
/home/root/catkin_ws/src/hexbug/launch/mavros.launch
/home/root/catkin_ws/src/hexbug/launch/relay.launch
```

## Custom MAVROS Launch Configuration

The drone does not use the stock `mavros/launch/px4.launch` directly for its
flight script. It launches the custom `hexbug` MAVROS launch file.

Current `/home/root/catkin_ws/prepare_flight.sh`:

```bash
#!/bin/bash

set -eo pipefail

source /opt/ros/melodic/setup.bash
source /home/root/catkin_ws/devel/setup.bash
set -u

roscore &
roslaunch hexbug mavros.launch \
    fcu_url:=udp://127.0.0.1:14557@:14557 \
    gcs_url:=udp://@192.168.5.221 \
    tgt_system:=1 \
    tgt_component:=1 \
    log_output:=screen \
    fcu_protocol:=v2.0 \
    respawn_mavros:=false \
    namespace:=mavros &
roslaunch hexbug relay.launch &
wait
```

Required MAVROS runtime arguments:

```text
fcu_url=udp://127.0.0.1:14557@:14557
gcs_url=udp://@192.168.5.221
tgt_system=1
tgt_component=1
log_output=screen
fcu_protocol=v2.0
respawn_mavros=false
namespace=mavros
```

Current `/home/root/catkin_ws/src/hexbug/launch/mavros.launch`:

```xml
<launch>
	<arg name="fcu_url" />
	<arg name="gcs_url" />
	<arg name="tgt_system" />
	<arg name="tgt_component" />
	<arg name="log_output" />
	<arg name="fcu_protocol" />
	<arg name="respawn_mavros" />
	<arg name="namespace" />

	<include file="$(find mavros)/launch/node.launch">
		<arg name="pluginlists_yaml" value="$(find hexbug)/config/px4_pluginlists.yaml" />
		<arg name="config_yaml" value="$(find hexbug)/config/px4.yaml" />

		<arg name="fcu_url" value="$(arg fcu_url)" />
		<arg name="gcs_url" value="$(arg gcs_url)" />
		<arg name="tgt_system" value="$(arg tgt_system)" />
		<arg name="tgt_component" value="$(arg tgt_component)" />
		<arg name="log_output" value="$(arg log_output)" />
		<arg name="fcu_protocol" value="$(arg fcu_protocol)" />
		<arg name="respawn_mavros" value="$(arg respawn_mavros)" />
	</include>

	<node pkg="tf" type="static_transform_publisher" name="tf0" args="0 0 0 0 0 0 world map 10" />
	<node pkg="tf" type="static_transform_publisher" name="tf1" args="0 0 0 0 0 0 world mocap 10" />
</launch>
```

Current `/home/root/catkin_ws/src/hexbug/launch/relay.launch`:

```xml
<launch>
	<node pkg="topic_tools" type="relay" name="relay_morphy_prog_vision_pose" args="/qualisys/morphy_prog/pose /mavros/vision_pose/pose" output="screen" />
	<node pkg="topic_tools" type="relay" name="relay_morphy_prog_mocap_pose" args="/qualisys/morphy_prog/pose /mavros/mocap/pose" output="screen" />
</launch>
```

Current `/home/root/catkin_ws/src/hexbug/config/px4_pluginlists.yaml`:

```yaml
plugin_blacklist: []

plugin_whitelist:
- 'sys_*'
- setpoint_raw
- setpoint_trajectory
- imu_pub
- imu
- sys_status
- sys_time
- command
- rc_io
- local_position
- vision_pose_estimate
- esc_status
# - manual_control
```

Current `/home/root/catkin_ws/src/hexbug/config/px4.yaml`:

```yaml
startup_px4_usb_quirk: true

conn:
  heartbeat_rate: 1.0
  timeout: 10.0
  timesync_rate: 10.0
  system_time_rate: 1.0

sys:
  min_voltage: 10.0
  disable_diag: false

time:
  time_ref_source: "fcu"
  timesync_mode: MAVLINK
  timesync_avg_alpha: 0.6

cmd:
  use_comp_id_system_control: false

imu:
  frame_id: "base_link"
  linear_acceleration_stdev: 0.0003
  angular_velocity_stdev: 0.0003490659
  orientation_stdev: 1.0
  magnetic_stdev: 0.0

setpoint_attitude:
  reverse_thrust: false
  use_quaternion: false
  tf:
    listen: false
    frame_id: "map"
    child_frame_id: "target_attitude"
    rate_limit: 50.0

setpoint_raw:
  thrust_scaling: 1.0
```

## Additional Root Scripts

The following root-level scripts are present and related to ROS/MAVROS/mocap:

```text
/home/root/launch-relay.sh
/home/root/mocap_pane1_init.sh
/home/root/qualisys_odom_to_mavros_odometry.py
/home/root/qualisys_odom_to_px4_mavlink.py
/home/root/record_qualisys_odom_csv.py
```

Current `/home/root/launch-relay.sh`:

```bash
#!/bin/bash

set -eo pipefail

source /opt/ros/melodic/setup.bash
source /home/root/catkin_ws/devel/setup.bash
set -u

rosrun topic_tools relay /qualisys/morphy_prog/pose /mavros/vision_pose/pose &
rosrun topic_tools relay /qualisys/morphy_prog/pose /mavros/mocap/pose
```

The mocap helper script `/home/root/mocap_pane1_init.sh` additionally uses:

```text
ROS_MASTER_URI=http://192.168.5.5:11311/
/qualisys/prog_morphy/odom
/mavlink/to
/tmp/qualisys_odom.csv
/tmp/qualisys_odom_record.log
/tmp/qualisys_odom_relay.log
```

It restarts/starts these services when run:

```text
init_slpi
voxl-imu-server
voxl-px4
```

Do not run this script during passive validation, because it intentionally
restarts services and removes `/tmp/qualisys_odom.csv`.

## New Drone Provisioning Checklist

1. Flash/install the same ModalAI VOXL 2 system image:
   `1.8.04-M0104-14.1a-perf`.
2. Confirm Ubuntu is `18.04.5 LTS` and architecture is `arm64`.
3. Configure APT repositories exactly as listed above, including ROS Melodic
   for Ubuntu Bionic and the ModalAI SDK 1.5 QRB5165 repo.
4. Install/hold `voxl-suite=1.5.0` and the VOXL package versions listed above.
5. Install ROS Melodic packages from the exact package list in this document.
6. Install MAVROS packages at exactly:
   - `ros-melodic-mavros=1.16.0-1bionic.20230605.155036`
   - `ros-melodic-mavros-extras=1.16.0-1bionic.20230605.160238`
   - `ros-melodic-mavros-msgs=1.16.0-1bionic.20230505.180431`
   - `ros-melodic-libmavconn=1.16.0-1bionic.20230605.154504`
   - `ros-melodic-mavlink=2022.12.30-1bionic.20230605.154004`
7. Ensure GeographicLib datasets exist under `/usr/share/GeographicLib`.
8. Recreate `/home/root/catkin_ws/src/hexbug` with the same package, launch,
   and config files.
9. Build `/home/root/catkin_ws` so `/home/root/catkin_ws/devel/setup.bash`
   exists.
10. Recreate `/home/root/.bashrc` ROS lines exactly, adjusting `ROS_IP` only if
    the new drone must use a different static address.
11. Recreate `/home/root/catkin_ws/prepare_flight.sh`,
    `/home/root/launch-relay.sh`, and the Qualisys helper scripts if the new
    drone is expected to support the same mocap workflow.
12. Match service enablement for `voxl-px4`, `voxl-vision-hub`,
    `voxl-imu-server`, `voxl-cpu-monitor`, `voxl-portal`, and
    `voxl-wait-for-fs`.

## Read-Only Validation Commands for the New Drone

Run these checks after provisioning:

```bash
cat /etc/os-release
uname -a
voxl-version
dpkg-query -W -f='${Package}\t${Version}\t${Architecture}\n' \
  'ros-melodic-mavros*' 'ros-melodic-libmavconn' 'ros-melodic-mavlink'
source /opt/ros/melodic/setup.bash
rosversion -d
rosversion mavros
rosversion mavros_extras
source /home/root/catkin_ws/devel/setup.bash
echo "$ROS_PACKAGE_PATH"
rospack find hexbug
rospack find mavros
ls /usr/share/GeographicLib/geoids /usr/share/GeographicLib/gravity /usr/share/GeographicLib/magnetic
systemctl list-unit-files --no-pager | grep -E 'voxl-(px4|vision-hub|imu-server|cpu-monitor|portal|wait-for-fs)'
```

Expected key validation output:

```text
ROS distro: melodic
ROS version: 1
ROS Python version: 2
MAVROS: 1.16.0
MAVROS extras: 1.16.0
ROS_PACKAGE_PATH: /home/root/catkin_ws/src:/opt/ros/melodic/share
hexbug package found in /home/root/catkin_ws/src/hexbug
mavros package found in /opt/ros/melodic/share/mavros
```

## Recovered Qualisys Helper Scripts

Recovered from the original drone over `adb -s 51326bba` on 2026-05-07 so
the new drone can be reprovisioned with the exact helper script contents.
Canonical repo copies are also stored under `voxl_src/drone_ros_files/`.

### `/home/root/mocap_pane1_init.sh`

Metadata observed on the original drone: `-rw-r--r-- root:root 3379 bytes`

```bash
#!/bin/bash

wait_for_service() {
    local service_name="$1"
    local timeout_s="$2"
    local deadline_s=$(($(date +%s) + timeout_s))
    while [ $(date +%s) -le "$deadline_s" ]; do
        if systemctl is-active --quiet "$service_name"; then
            echo "[OK] $service_name is active"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] $service_name did not become active within ${timeout_s}s"
    systemctl status --no-pager "$service_name" || true
    return 1
}

check_background_process() {
    local process_label="$1"
    local process_pid="$2"
    if ps -p "$process_pid" >/dev/null 2>&1; then
        echo "[OK] $process_label is running with pid $process_pid"
        return 0
    fi
    echo "[ERROR] $process_label failed to stay alive"
    return 1
}

wait_for_ros_master() {
    local timeout_s="$1"
    local deadline_s=$(($(date +%s) + timeout_s))
    while [ $(date +%s) -le "$deadline_s" ]; do
        if rostopic list >/dev/null 2>&1; then
            echo "[OK] ROS master is reachable"
            return 0
        fi
        sleep 0.5
    done
    echo "[ERROR] ROS master did not become reachable within ${timeout_s}s"
    return 1
}

mocap_pane1_setup() {
    set -eo pipefail
    source /home/root/catkin_ws/devel/setup.bash
    set -u

    echo "[INFO] Preparing Qualisys recorder and MAVLink relay on $(hostname)"
    chmod +x /home/root/qualisys_odom_to_px4_mavlink.py /home/root/record_qualisys_odom_csv.py
    rm -f /tmp/qualisys_odom.csv
    export ROS_MASTER_URI=http://192.168.5.5:11311/
    wait_for_ros_master 30

    echo "[INFO] Stopping any previous recorder or relay processes"
    pkill -f "[r]ecord_qualisys_odom_csv.py" || true
    pkill -f "[q]ualisys_odom_to_px4_mavlink.py" || true
    sleep 1

    echo "[INFO] Starting raw Qualisys recorder -> /tmp/qualisys_odom.csv"
    python /home/root/record_qualisys_odom_csv.py --source-topic /qualisys/prog_morphy/odom --output-csv /tmp/qualisys_odom.csv >/tmp/qualisys_odom_record.log 2>&1 &
    QUALISYS_RECORDER_PID=$!
    sleep 1
    check_background_process "Raw Qualisys recorder" "$QUALISYS_RECORDER_PID"

    echo "[INFO] Starting MAVLink ODOMETRY relay -> /mavlink/to"
    python /home/root/qualisys_odom_to_px4_mavlink.py --source-topic /qualisys/prog_morphy/odom --target-topic /mavlink/to --system-id 1 --component-id 197 --quality 0 >/tmp/qualisys_odom_relay.log 2>&1 &
    QUALISYS_RELAY_PID=$!
    sleep 1
    check_background_process "Qualisys MAVLink relay" "$QUALISYS_RELAY_PID"

    echo "[INFO] Restarting init_slpi"
    systemctl restart init_slpi
    wait_for_service init_slpi 20

    echo "[INFO] Restarting voxl-imu-server"
    systemctl restart voxl-imu-server
    wait_for_service voxl-imu-server 20

    echo "[INFO] Starting voxl-px4"
    systemctl start voxl-px4
    wait_for_service voxl-px4 20

    sleep 5

    echo "[INFO] Qualisys recorder log: /tmp/qualisys_odom_record.log"
    echo "[INFO] Qualisys relay log: /tmp/qualisys_odom_relay.log"
    echo "[INFO] Raw Qualisys CSV: /tmp/qualisys_odom.csv"
    echo "[INFO] Pane 1 setup complete"
}

if mocap_pane1_setup; then
    echo "[INFO] Remote shell is ready"
else
    setup_status=$?
    echo "[ERROR] Pane 1 setup failed with status $setup_status"
fi

unset -f mocap_pane1_setup
unset -f wait_for_service
unset -f check_background_process
unset -f wait_for_ros_master
```

### `/home/root/qualisys_odom_to_mavros_odometry.py`

Metadata observed on the original drone: `-rwxr-xr-x system:system 4306 bytes`

```python
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
```

### `/home/root/qualisys_odom_to_px4_mavlink.py`

Metadata observed on the original drone: `-rwxr-xr-x root:root 13163 bytes`

```python
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
MAV_FRAME_BODY_FRD = 12
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


def transform_covariance(covariance36, first_rotation, second_rotation):
    covariance_matrix = covariance36_to_matrix(covariance36)
    transform = [
        [first_rotation[0][0], first_rotation[0][1], first_rotation[0][2], 0.0, 0.0, 0.0],
        [first_rotation[1][0], first_rotation[1][1], first_rotation[1][2], 0.0, 0.0, 0.0],
        [first_rotation[2][0], first_rotation[2][1], first_rotation[2][2], 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, second_rotation[0][0], second_rotation[0][1], second_rotation[0][2]],
        [0.0, 0.0, 0.0, second_rotation[1][0], second_rotation[1][1], second_rotation[1][2]],
        [0.0, 0.0, 0.0, second_rotation[2][0], second_rotation[2][1], second_rotation[2][2]],
    ]
    transformed = matrix_mul(transform, matrix_mul(covariance_matrix, transpose(transform)))
    return covariance_matrix_to_urt(transformed)


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

        position_enu = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        linear_velocity_flu = [
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
        require_finite_array(linear_velocity_flu, "twist.linear")
        require_finite_array(angular_velocity_flu, "twist.angular")

        rotation_enu_raw = quaternion_to_matrix(ros_quaternion_to_wxyz(msg.pose.pose.orientation))
        rotation_ned_frd = matrix_mul(ENU_TO_NED, matrix_mul(rotation_enu_raw, FLU_TO_FRD))
        quaternion_ned_frd = matrix_to_quaternion(rotation_ned_frd)
        position_ned = matrix_vector_mul(ENU_TO_NED, position_enu)
        linear_velocity_frd = matrix_vector_mul(FLU_TO_FRD, linear_velocity_flu)
        angular_velocity_frd = matrix_vector_mul(FLU_TO_FRD, angular_velocity_flu)

        pose_covariance = transform_covariance(msg.pose.covariance, ENU_TO_NED, FLU_TO_FRD)
        velocity_covariance = transform_covariance(msg.twist.covariance, FLU_TO_FRD, FLU_TO_FRD)

        full_payload = struct.pack(
            "<Qfff4fffffff21f21fBBBBb",
            msg.header.stamp.to_nsec() // 1000,
            position_ned[0],
            position_ned[1],
            position_ned[2],
            quaternion_ned_frd[0],
            quaternion_ned_frd[1],
            quaternion_ned_frd[2],
            quaternion_ned_frd[3],
            linear_velocity_frd[0],
            linear_velocity_frd[1],
            linear_velocity_frd[2],
            angular_velocity_frd[0],
            angular_velocity_frd[1],
            angular_velocity_frd[2],
            *(pose_covariance + velocity_covariance + [
                MAV_FRAME_LOCAL_NED,
                MAV_FRAME_BODY_FRD,
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
```

### `/home/root/record_qualisys_odom_csv.py`

Metadata observed on the original drone: `-rwxr-xr-x root:root 2709 bytes`

```python
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
```
