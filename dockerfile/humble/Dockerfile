# base image
FROM nvidia/opengl:1.2-glvnd-devel-ubuntu22.04

# setting environmental variables
ENV DEBIAN_FRONTEND noninteractive
ARG MAKEFLAGS
ARG PARALLELWORKERS
ARG DISPLAY
ARG WAYLAND_DISPLAY
ARG XDG_RUNTIME_DIR
ARG PULSE_SERVER
ENV MAKEFLAGS $MAKEFLAGS
ENV PARALLELWORKERS $PARALLELWORKERS
ENV DISPLAY $DISPLAY
ENV WAYLAND_DISPLAY $WAYLAND_DISPLAY
ENV XDG_RUNTIME_DIR $XDG_RUNTIME_DIR
ENV PULSE_SERVER $PULSE_SERVER
ENV __NV_PRIME_RENDER_OFFLOAD 1
ENV __GLX_VENDOR_LIBRARY_NAME nvidia

# setting time zone
RUN apt update -y && apt upgrade -y && apt install -y --no-install-recommends tzdata
ENV TZ=Asia/Tokyo

RUN apt update -y && apt install -y --no-install-recommends \
    software-properties-common \
    libxkbcommon-x11-0 \
    libcanberra-gtk* \
    gnome-screenshot \
    libgl1-mesa-dev \
    build-essential \
    nvidia-settings \
    nvidia-prime \
    lsb-release \
    libudev-dev \
    imagemagick \
    python3-pip \
    python3-tk \
    mesa-utils \
    net-tools \
    aptitude \
    xorg-dev \
    x11-apps \
    python3 \
    libxcb* \
    dvipng \
    psmisc \
    ffmpeg \
    gnupg \
    byobu \
    unzip \
    curl \
    dbus \
    wget \
    git \
    vim \
    zip \
    && rm -rf /var/lib/apt/lists/*

# adding the ROS2 GPG key with apt and add the repository to your sources list
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
RUN /bin/bash -c 'echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null'

# updating mesa (fix the issue of displaying black world in Rviz2 and Gazebo)
# https://github.com/ros2/rviz/issues/948
RUN add-apt-repository ppa:kisak/kisak-mesa -y && apt update -y

# installing ROS2 packages and colcon build system with mixin
# changing the middleware (CycloneDDS recommended for MoveIt2)
RUN apt install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-colcon-mixin \
    ros-humble-desktop \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-gazebo-* \
    ros-humble-ros-gz \
    ros-humble-ign-ros2-control \
    ros-humble-ign-ros2-control-demos \
    ros-humble-warehouse-* \
    ros-humble-ros-testing \
    ros-humble-moveit \
    ros-humble-moveit-visual-tools \
    ros-dev-tools \
    gazebo \
    && echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> ~/.bashrc

RUN /bin/bash -c "colcon mixin add default https://raw.githubusercontent.com/colcon/colcon-mixin-repository/master/index.yaml && colcon mixin update default"

RUN mkdir -p /root/colcon_ws/src
WORKDIR /root/colcon_ws
RUN	/bin/bash -c "source /opt/ros/humble/setup.bash && colcon build"
RUN	echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc && \
    echo "source /root/colcon_ws/install/setup.bash" >> ~/.bashrc && \
    echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/root/colcon_ws" >> ~/.bashrc && \
    echo "export ROS_WORKSPACE=/root/colcon_ws" >> ~/.bashrc && \
    echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc && \
    echo "export _colcon_cd_root=/opt/ros/humble/" >> ~/.bashrc

# building ros2 packages
WORKDIR /root/colcon_ws/src
RUN git clone https://github.com/ros-simulation/gazebo_ros2_control.git -b humble --recursive && \
    cd gazebo_ros2_control && \
    git reset --hard b49ed0b
RUN git clone https://github.com/ros-planning/moveit_msgs.git -b humble --recursive && \
    cd moveit_msgs && \
    git reset --hard 99e6898
RUN git clone https://github.com/ros-planning/moveit_resources.git -b humble --recursive && \
    cd moveit_resources && \
    git reset --hard 45ea355
RUN git clone https://github.com/moveit/moveit2.git -b humble --recursive && \
    cd moveit2 && \
    git reset --hard 7ffdfd3
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/COLCON_IGNORE"
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/moveit_ros/robot_interaction/COLCON_IGNORE"
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/moveit_ros/hybrid_planning/COLCON_IGNORE"
RUN git clone https://github.com/ros-planning/moveit_task_constructor.git -b humble --recursive && \
    cd moveit_task_constructor && \
    git reset --hard e704159
RUN git clone https://github.com/ros-planning/srdfdom.git -b ros2 --recursive && \
    cd srdfdom && \
    git reset --hard c118b35
RUN git clone https://github.com/ros-controls/ros2_control.git -b humble --recursive && \
    cd ros2_control && \
    git reset --hard 0af922d
RUN git clone https://github.com/ros-controls/ros2_controllers.git -b humble --recursive && \
    cd ros2_controllers && \
    git reset --hard 3557ed0
RUN git clone https://github.com/takuya-ki/gazebo-pkgs.git -b humble && \
    cd gazebo-pkgs && \
    git reset --hard 233cc8b
RUN git clone https://github.com/roboticsgroup/roboticsgroup_upatras_gazebo_plugins.git -b ros2 && \
    cd roboticsgroup_upatras_gazebo_plugins && \
    git reset --hard 8bf90f5
RUN echo "export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/colcon_ws/build/gazebo_grasp_plugin" >> ~/.bashrc
RUN echo "export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/colcon_ws/build/roboticsgroup_upatras_gazebo_plugins" >> ~/.bashrc
RUN echo "export PLUGINLIB_EXPORT_CLASS=$PLUGINLIB_EXPORT_CLASS:/root/colcon_ws/install/iiwa_gazebo/lib" >> ~/.bashrc
RUN echo "export IGN_IP=127.0.0.1" >> ~/.bashrc

WORKDIR /root/colcon_ws/
RUN /bin/bash -c "rosdep init && rosdep update && rosdep install -r --from-paths . --ignore-src --rosdistro humble -y"

# to remove warnings on setuptools that appear when building with colcon
RUN pip install setuptools==58.2.0

WORKDIR /root/colcon_ws/
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --executor parallel --parallel-workers $PARALLELWORKERS --symlink-install --mixin release"
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && rm /root/colcon_ws/src/moveit2/COLCON_IGNORE && colcon build --executor sequential --symlink-install --mixin release"
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && rm /root/colcon_ws/src/moveit2/moveit_ros/hybrid_planning/COLCON_IGNORE && colcon build --executor parallel --parallel-workers $PARALLELWORKERS --symlink-install --mixin release"
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && rm /root/colcon_ws/src/moveit2/moveit_ros/robot_interaction/COLCON_IGNORE && colcon build --executor parallel --parallel-workers $PARALLELWORKERS --symlink-install --mixin release"

# Generates bash commands
WORKDIR /root/colcon_ws/
RUN echo "source /opt/ros/humble/setup.bash && source /root/colcon_ws/install/setup.bash && ros2 launch iiwa_description display_iiwa.launch.py" >> ~/.bash_history
RUN echo "source /opt/ros/humble/setup.bash && source /root/colcon_ws/install/setup.bash && ros2 daemon stop; sleep 1; ros2 daemon start; kill 0 && ros2 launch iiwa_gazebo iiwa_example.launch.py" >> ~/.bash_history
