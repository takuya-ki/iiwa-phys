FROM ros:noetic

RUN apt update && apt install -y --no-install-recommends curl lsb-release wget gnupg
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
RUN curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add -

# setting environmental variables
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y && apt install -y --no-install-recommends tzdata
RUN apt update -y && apt install -y --no-install-recommends \
    ros-noetic-desktop-full \
    ros-noetic-image-proc \
    ros-noetic-image-view \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-control-toolbox \
    ros-noetic-moveit \
    ros-noetic-moveit-visual-tools \
    ros-noetic-effort-controllers \
    ros-noetic-forward-command-controller \
    ros-noetic-joint-state-controller \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-position-controllers \
    ros-noetic-velocity-controllers \
    ros-noetic-tf \
    ros-noetic-tf2-tools \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-genpy \
    ros-noetic-industrial-core \
    ros-noetic-roslint \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-urdf-tutorial \
    ros-noetic-trac-ik-kinematics-plugin \
    python3-osrf-pycommon \
    python3-catkin-tools \
    python3-rosdep \
    python3-pip \
    python3-tk \
    software-properties-common \
    build-essential \
    libxkbcommon-x11-0 \
    libcanberra-gtk* \
    libgl1-mesa-dev \
    libudev-dev \
    libxcb* \
    xorg-dev \
    net-tools \
    psmisc \
    byobu \
    dbus \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /tmp/
RUN mkdir -p /catkin_ws/src && \
	/bin/bash -c "cd /catkin_ws/src && source /opt/ros/noetic/setup.bash && catkin_init_workspace" && \
	/bin/bash -c "cd /catkin_ws && source /opt/ros/noetic/setup.bash && catkin build" && \
	echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc && \
	echo "source /catkin_ws/devel/setup.bash" >> ~/.bashrc && \
	echo "export ROS_PACKAGE_PATH=\${ROS_PACKAGE_PATH}:/catkin_ws" >> ~/.bashrc && \
	echo "export ROS_WORKSPACE=/catkin_ws" >> ~/.bashrc

WORKDIR /
RUN wget -P /opt https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh && \
    bash /opt/Anaconda3-2020.02-Linux-x86_64.sh -b -p /opt/anaconda3 && \
    rm /opt/Anaconda3-2020.02-Linux-x86_64.sh && \
    export PATH=/opt/anaconda3/bin:$PATH && \
    . ~/.bashrc && conda init && conda update -n base -c defaults conda

RUN pip3 install --target=/opt/ros/noetic/lib/python3/dist-packages rospkg
RUN rm /etc/apt/sources.list.d/ros1-latest.list
RUN . ~/.bashrc && conda deactivate && pip3 uninstall netifaces -y && pip3 install netifaces==0.11.0

# gazebo on ros1
WORKDIR /catkin_ws/src/
RUN git clone https://github.com/roboticsgroup/roboticsgroup_gazebo_plugins.git && \
    cd roboticsgroup_gazebo_plugins && \
    git reset --hard 509a32e

# pybullet
WORKDIR /
RUN . ~/.bashrc && conda create --name=pybullet python=3.8 -y
RUN . ~/.bashrc && conda activate pybullet && pip3 install gym numpy sympy pyglet Pillow pybullet numpy-quaternion pickle-mixin seaborn pandas pyassimp==4.1.3
RUN git clone https://github.com/caelan/pybullet-planning.git /scripts/pybullet_planning --recursive && \
    cd /scripts/pybullet_planning && \
    git config --global --add safe.directory /scripts/pybullet_planning && \
    git reset --hard d536384 && \
    git pull --recurse-submodules
RUN . ~/.bashrc && conda activate pybullet && cd /scripts/pybullet_planning && pip3 install -r requirements.txt scipy networkx

# isaacgym (download IsaacGym_Preview_4_Package before building this)
WORKDIR /
COPY ./isaacgym/ /tmp/isaacgym/
RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/tmp/isaacgym/python/isaacgym/_bindings/linux-x86_64" >> ~/.bashrc
RUN git clone https://github.com/isaac-sim/IsaacGymEnvs.git --recursive /tmp/IsaacGymEnvs && \
    cd /tmp/IsaacGymEnvs && \
    git reset --hard aeed298
RUN . ~/.bashrc && conda create --name=isaacgym python=3.8 -y
RUN . ~/.bashrc && conda activate isaacgym && pip3 install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 numpy-quaternion matplotlib==3.6.3 seaborn
RUN . ~/.bashrc && conda activate isaacgym && cd /tmp/isaacgym/python && pip3 install -e .
RUN . ~/.bashrc && conda activate isaacgym && cd /tmp/IsaacGymEnvs && pip3 install -e . && pip3 install numpy==1.19.5 setuptools==69.5.1

# mujoco
WORKDIR /
RUN . ~/.bashrc && conda create --name=mujoco python=3.8 -y
RUN . ~/.bashrc && conda activate mujoco && pip3 install mujoco mujoco-python-viewer jupyter numpy matplotlib

# genesys
WORKDIR /
RUN . ~/.bashrc && conda create --name=genesys python=3.10 -y
RUN . ~/.bashrc && conda activate genesys && pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
RUN . ~/.bashrc && conda activate genesys && pip install git+https://github.com/Genesis-Embodied-AI/Genesis.git
WORKDIR /scripts/
RUN git clone https://github.com/Genesis-Embodied-AI/Genesis --recursive

# Generates bash commands
WORKDIR /catkin_ws/src/
RUN echo "conda deactivate && source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && roslaunch iiwa_description display_iiwa.launch end_effector:=rq140" >> ~/.bash_history
RUN echo "conda deactivate && source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && roslaunch iiwa_moveit_config demo.launch end_effector:=rq140 use_gui:=true" >> ~/.bash_history
RUN echo "conda deactivate && source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && roslaunch iiwa_example_motion iiwa_example.launch" >> ~/.bash_history
RUN echo "conda deactivate && source /opt/ros/noetic/setup.bash && source /catkin_ws/devel/setup.bash && roslaunch iiwa_moveit_config demo_gazebo.launch end_effector:=rq140" >> ~/.bash_history

RUN echo "cd /scripts && conda activate pybullet && python pybullet_vis.py" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg naiveRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg RRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg BiRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg RRTStar" >> ~/.bash_history
RUN echo "cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg BiRRTStar" >> ~/.bash_history

RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_vis.py --num_envs 1" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_vis.py --num_envs 1 --sdf" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg naiveRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg RRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg BiRRT" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg RRTStar" >> ~/.bash_history
RUN echo "cd /scripts && conda activate isaacgym && CUDA_LAUNCH_BLOCKING=1 python isaacgym_rrt.py --alg BiRRTStar" >> ~/.bash_history

RUN echo "cd /scripts && conda activate mujoco && python mujoco_vis.py" >> ~/.bash_history

RUN echo "cd /scripts && conda activate genesys && python genesys_hello.py" >> ~/.bash_history
RUN echo "cd /scripts && conda activate genesys && python genesys_motion.py" >> ~/.bash_history