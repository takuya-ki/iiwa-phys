# iiwa-phys

[![support level: community](https://img.shields.io/badge/support%20level-community-lightgray.svg)](https://rosindustrial.org/news/2016/10/7/better-supporting-a-growing-ros-industrial-software-platform)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

- Configuration files, example programs, and docker environment for physics simulations for variable workstation with a LBR iiwa 14 R820 robot
- Note that this is still under development with several hand-crafted parameters

- [iiwa-phys](#iiwa-phys)
  - [Features](#features)
  - [Dependency (tested as a host machine)](#dependency-tested-as-a-host-machine)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Gazebo Classic on ROS1](#gazebo-classic-on-ros1)
    - [Pybullet](#pybullet)
    - [Isaac Gym](#isaac-gym)
    - [MuJoCo](#mujoco)
    - [Gazebo Ignition on ROS2](#gazebo-ignition-on-ros2)

## Features

- Support the following physics simulators
  1. [Gazebo Classic](https://classic.gazebosim.org/)
  2. [PyBullet](https://pybullet.org/wordpress/)
  3. [Isaac Gym](https://developer.nvidia.com/isaac-gym)
  4. [MuJoCo](https://mujoco.org/)
  5. [Gazebo Ignition](https://gazebosim.org/)
  - Please refer to the comparison below
    - [A brief summary for physics simulators](https://simulately.wiki/docs/comparison/)
    - [A Review of Nine Physics Engines for Reinforcement Learning Research](https://arxiv.org/pdf/2407.08590v1)

## Dependency (tested as a host machine)

- [Ubuntu 22.04 PC](https://ubuntu.com/certified/laptops?q=&limit=20&vendor=Dell&vendor=Lenovo&vendor=HP&release=22.04+LTS)
  - NVIDIA GeForce RTX 3070
  - NVIDIA Driver 470.256.02
  - Docker 27.4.1
  - Docker Compose 2.32.1
  - nvidia-container-runtime 1.17.5

## Installation

1. Download the software
    ```bash
    git clone git@github.com:takuya-ki/iiwa-phys.git --recursive --depth 1
    ```  
2. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
3. Unzip the file via  
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz -C pathto/iiwa-phys/dockerfile/noetic
    ```  
4. Build the dockerfile
    ```bash
    cd iiwa-phys && COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker compose build --no-cache --parallel 
    ```  

## Usage

1. Create and run the docker container from the docker image
    ```bash
    cd iiwa-phys && docker compose up --timeout 600
    ```  
2. Execute the docker container
    ```bash
    xhost + && docker exec -it iiwa_phys_[noetic/humble]_container bash
    ```  
    - iiwa_phys_noetic_container
      - [Gazebo Classic on ROS1](#gazebo-classic-on-ros1)
      - [Pybullet](#pybullet)
      - [Isaac Gym](#isaac-gym)
      - [MuJoCo](#mujoco)
    - iiwa_phys_humble_container
      - [Gazebo Ignition on ROS2](#gazebo-ignition-on-ros2)
3. Run a command in the docker container

#### Pybullet
- Show a LBR iiwa robot
    ```bash
    cd /scripts && conda activate pybullet && python pybullet_vis.py
    ```  
    <img src=dataset/images/iiwa_bullet_vis.png width=320>  

- Example motion planning for a LBR iiwa robot
    ```bash
    cd /scripts && conda activate pybullet && python pybullet_rrt.py --alg '[naiveRRT, RRT, BiRRT, RRTStar, BiRRTStar]'
    ```  
    <img src=dataset/images/iiwa_bullet_rrt.gif width=320>  

#### MuJoCo
- Show a LBR iiwa robot
    ```bash
    cd /scripts && conda activate mujoco && python mujoco_vis.py
    ```  
    <img src=dataset/images/iiwa_mujoco_vis.png width=320>

## Contributors

We always welcome collaborators!

## Author

[Takuya Kiyokawa](https://takuya-ki.github.io/)  
