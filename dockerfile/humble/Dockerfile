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
COPY ./colcon_ws/src/moveit_configs_utils/launches.py /root/colcon_ws/src/moveit2/moveit_configs_utils/moveit_configs_utils/
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/COLCON_IGNORE"
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/moveit_ros/robot_interaction/COLCON_IGNORE"
RUN /bin/bash -c "touch /root/colcon_ws/src/moveit2/moveit_ros/hybrid_planning/COLCON_IGNORE"
RUN git clone https://github.com/ros-planning/moveit_task_constructor.git -b humble --recursive && \
    cd moveit_task_constructor && \
    git reset --hard 7566349
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
RUN echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/colcon_ws/build/gazebo_grasp_plugin' >> ~/.bashrc
RUN echo 'export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:/root/colcon_ws/build/roboticsgroup_upatras_gazebo_plugins' >> ~/.bashrc
RUN echo 'export PLUGINLIB_EXPORT_CLASS=$PLUGINLIB_EXPORT_CLASS:/root/colcon_ws/install/iiwa_gazebo/lib' >> ~/.bashrc
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

# lbr_fri_ros2_stack
# https://raw.githubusercontent.com/lbr-stack/lbr_fri_ros2_stack/humble/lbr_fri_ros2_stack/repos-fri-${FRI_CLIENT_VERSION}.yaml
WORKDIR /root/colcon_ws/src
RUN pip install pyoptas casadi pynput
COPY ./yaml/repos-fri-1.15.yaml /tmp/
RUN /bin/bash -c "source /opt/ros/humble/setup.bash" && \
    mkdir -p lbr-stack/src && cd lbr-stack && \
    vcs import src --input /tmp/repos-fri-1.15.yaml && \
    rosdep install --from-paths src -i -r -y
WORKDIR /root/colcon_ws
RUN apt install -y --no-install-recommends ros-humble-kinematics-interface
RUN /bin/bash -c "source /opt/ros/humble/setup.bash && colcon build --executor parallel --parallel-workers $PARALLELWORKERS --symlink-install --mixin release" && \
    /bin/bash -c "source /root/colcon_ws/install/setup.bash"

# Anaconda (for conda-based simulator environments)
WORKDIR /
RUN wget -P /opt https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
    bash /opt/Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm /opt/Anaconda3-2020.02-Linux-x86_64.sh && \
    export PATH=/opt/anaconda3/bin:$PATH && \
    . ~/.bashrc && conda init && conda update -n base -c defaults conda

# pybullet
WORKDIR /
RUN . ~/.bashrc && conda create --name=pybullet python=3.8 -y
RUN . ~/.bashrc && conda activate pybullet && pip3 install gym numpy sympy pyglet Pillow pybullet numpy-quaternion pickle-mixin seaborn pandas pyassimp==4.1.3
RUN git clone https://github.com/caelan/pybullet-planning.git /opt/pybullet_planning --recursive && \
    cd /opt/pybullet_planning && \
    git config --global --add safe.directory /opt/pybullet_planning && \
    git reset --hard d536384 && \
    git pull --recurse-submodules
RUN . ~/.bashrc && conda activate pybullet && cd /opt/pybullet_planning && pip3 install -r requirements.txt scipy networkx
RUN mkdir -p /opt/anaconda3/envs/pybullet/etc/conda/activate.d && \
    echo 'export PYTHONPATH=$PYTHONPATH:/opt' > /opt/anaconda3/envs/pybullet/etc/conda/activate.d/pybullet_env.sh

# isaacgym (download IsaacGym_Preview_4_Package before building this)
WORKDIR /
COPY ./isaacgym/ /tmp/isaacgym/
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/isaacgym/python/isaacgym/_bindings/linux-x86_64' >> ~/.bashrc
RUN git clone https://github.com/isaac-sim/IsaacGymEnvs.git --recursive /tmp/IsaacGymEnvs && \
    cd /tmp/IsaacGymEnvs && \
    git reset --hard aeed298
RUN . ~/.bashrc && conda create --name=isaacgym python=3.8 -y
RUN . ~/.bashrc && conda activate isaacgym && pip3 install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 numpy-quaternion matplotlib==3.6.3 seaborn
RUN . ~/.bashrc && conda activate isaacgym && cd /tmp/isaacgym/python && pip3 install -e .
RUN . ~/.bashrc && conda activate isaacgym && cd /tmp/IsaacGymEnvs && pip3 install -e . && pip3 install numpy==1.19.5 setuptools==69.5.1
RUN mkdir -p /opt/anaconda3/envs/isaacgym/etc/conda/activate.d && \
    echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib' > /opt/anaconda3/envs/isaacgym/etc/conda/activate.d/isaacgym_lib.sh

# mujoco
WORKDIR /
RUN . ~/.bashrc && conda create --name=mujoco python=3.8 -y
RUN . ~/.bashrc && conda activate mujoco && pip3 install mujoco mujoco-python-viewer jupyter numpy matplotlib

# genesis
WORKDIR /
RUN . ~/.bashrc && conda create --name=genesis python=3.10 -y
RUN . ~/.bashrc && conda activate genesis && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN . ~/.bashrc && conda activate genesis && pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
WORKDIR /opt/
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis --recursive

# Generates bash commands
WORKDIR /root/colcon_ws/

# visualize a model with rviz
RUN echo "ros2 launch lbr_bringup mock.launch.py model:=iiwa14" >> ~/.bash_history
RUN echo "ros2 launch lbr_bringup rviz.launch.py rviz_cfg_pkg:=lbr_bringup rviz_cfg:=config/mock.rviz" >> ~/.bash_history

# moveit
RUN echo "ros2 launch lbr_bringup mock.launch.py model:=iiwa14" >> ~/.bash_history
RUN echo "ros2 launch lbr_bringup move_group.launch.py mode:=mock rviz:=true model:=iiwa14" >> ~/.bash_history

# gazebo
RUN echo "ros2 launch lbr_bringup gazebo.launch.py ctrl:=joint_trajectory_controller model:=iiwa14" >> ~/.bash_history
RUN echo "ros2 run lbr_demos_py joint_trajectory_client --ros-args -r __ns:=/lbr" >> ~/.bash_history

RUN echo "ros2 daemon stop; sleep 1; ros2 daemon start; kill 0" >> ~/.bash_history
RUN echo "source /opt/ros/humble/setup.bash && source /root/colcon_ws/install/setup.bash" >> ~/.bash_history

# pybullet commands
RUN echo "cd /scripts && conda activate pybullet && python pybullet_vis.py" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg naiveRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg RRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg BiRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg RRTStar" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg BiRRTStar" >> ~/.bash_history

# isaacgym commands
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_vis.py --num_envs 1" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_vis.py --num_envs 1 --sdf" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg naiveRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg RRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg BiRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg RRTStar" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg BiRRTStar" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_pick.py --num_envs 1" >> ~/.bash_history

# mujoco commands
RUN echo "cd /scripts && conda activate mujoco && python mujoco_vis.py" >> ~/.bash_history

# genesis commands
RUN echo "cd /scripts && conda activate genesis && python genesis_hello.py" >> ~/.bash_history
RUN echo "cd /scripts && conda activate genesis && python genesis_motion.py" >> ~/.bash_history
RUN echo "cd /scripts && conda activate genesis && python genesis_grasp.py" >> ~/.bash_history
