#!/usr/bin/env python3

import os
import time
import numpy as np

import pybullet as p
import pybullet_data

from pybullet_planning.pybullet_tools.utils import *


def draw_sphere_marker(position, radius, color):
    vs_id = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=color)
    marker_id = p.createMultiBody(
        basePosition=position,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=vs_id)
    return marker_id


def update_bullet_sim_sec(wait_sec):
    tic = time.time()
    while p.isConnected():
        p.stepSimulation()
        toc = time.time()
        if wait_sec > 0.0:
            if toc - tic > wait_sec:
                break


if __name__ == "__main__":

    # start and goal position and configuration
    start_conf = np.array([-1.6, -1.3, 0., -1.0, 0., 1.2, -1.4])
    goal_conf = np.array([1.6, -1.4, 0., -1.4, 0., 0.8, -1.4])

    # set up simulator
    physicsClient = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.setGravity(0, 0, -9.8)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, True)
    camdist = 2.000
    campitch = -42.20
    campos = (0.0, 0.0, 0.0)
    p.resetDebugVisualizerCamera(
        cameraDistance=camdist,
        cameraYaw=58.000,
        cameraPitch=campitch,
        cameraTargetPosition=campos)

    asset_root = "/dataset"
    plane = p.loadURDF("plane.urdf")

    obstacle1 = p.loadURDF(
        os.path.join(
            asset_root,
            'urdfs',
            'block.urdf'),
        basePosition=[-0.2, -0.05, 0.7],
        useFixedBase=True)
    p.stepSimulation()
    obstacle2 = p.loadURDF(
        os.path.join(
            asset_root,
            'urdfs',
            'block.urdf'),
        basePosition=[0., -0.1, 0.7],
        useFixedBase=True)
    p.stepSimulation()
    obstacle3 = p.loadURDF(
        os.path.join(
            asset_root,
            'urdfs',
            'block.urdf'),
        basePosition=[0.2, -0.05, 0.8],
        useFixedBase=True)
    p.stepSimulation()
    obstacles = [plane, obstacle1, obstacle2, obstacle3]

    iiwa_urdfname = 'iiwa14_rq140.urdf'
    iiwa = p.loadURDF(
        os.path.join(
            asset_root,
            'urdfs',
            iiwa_urdfname),
        basePosition=[0, 0, 0.02],
        useFixedBase=True)
    p.stepSimulation()

    # extract link information
    _link_name_to_index = {p.getBodyInfo(iiwa)[0].decode('UTF-8'):-1,}        
    for _id in range(p.getNumJoints(iiwa)):
        _name = p.getJointInfo(iiwa, _id)[12].decode('UTF-8')
        _link_name_to_index[_name] = _id

    iiwa_joint_indices = [
        _link_name_to_index['iiwa_link_1'],
        _link_name_to_index['iiwa_link_2'],
        _link_name_to_index['iiwa_link_3'],
        _link_name_to_index['iiwa_link_4'],
        _link_name_to_index['iiwa_link_5'],
        _link_name_to_index['iiwa_link_6'],
        _link_name_to_index['iiwa_link_7']]

    set_joint_positions(
        iiwa, iiwa_joint_indices, start_conf)
    p.stepSimulation()
    start_position = p.getLinkState(
        iiwa,
        _link_name_to_index['iiwa_link_ee'],
        computeForwardKinematics=True)[0]
    start_marker = draw_sphere_marker(
        position=start_position, radius=0.01, color=[0, 1, 0, 1])
    time.sleep(3.0)

    set_joint_positions(
        iiwa, iiwa_joint_indices, goal_conf)
    p.stepSimulation()
    goal_position = p.getLinkState(
        iiwa,
        _link_name_to_index['iiwa_link_ee'],
        computeForwardKinematics=True)[0]
    goal_marker = draw_sphere_marker(
        position=goal_position, radius=0.01, color=[1, 0, 0, 1])
    time.sleep(3.0)
