#!/usr/bin/env python3

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

import os
import math
import torch
import numpy as np

# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments
custom_parameters = [
    {"name": '--num_envs',
     "type": int,
     "default": 1,
     "help": "Number of environments to create"},
    {"name": '--sdf',
     "action": 'store_true',
     "help": "Use SDF-based collision check"}]

args = gymutil.parse_arguments(
    description="LBR iiwa",
    custom_parameters=custom_parameters)

# Force GPU:
if not args.use_gpu or args.use_gpu_pipeline:
    print("Forcing GPU sim - CPU sim not supported by SDF")
    args.use_gpu = True
    args.use_gpu_pipeline = True

# set torch device
device = args.sim_device

# configure sim
sim_params = gymapi.SimParams()
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
sim_params.dt = 1.0 / 60.0
sim_params.substeps = 2
sim_params.use_gpu_pipeline = args.use_gpu_pipeline
if args.physics_engine == gymapi.SIM_PHYSX:
    sim_params.physx.solver_type = 1
    sim_params.physx.num_position_iterations = 32
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.005
    sim_params.physx.friction_offset_threshold = 0.01
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = '/dataset'

# load iiwa asset
sdf_label = ''
if args.sdf:
    sdf_label = '_sdf'

iiwa_asset_file = os.path.join(
    'urdfs', 'iiwa14_rqhe' + sdf_label + '.urdf')

spacing = 1.0
cam_pos = gymapi.Vec3(1.5, 0, 1.5)

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
iiwa_asset = gym.load_asset(sim, asset_root, iiwa_asset_file, asset_options)

# configure iiwa dofs
iiwa_dof_props = gym.get_asset_dof_properties(iiwa_asset)
iiwa_lower_limits = iiwa_dof_props["lower"]
iiwa_upper_limits = iiwa_dof_props["upper"]
iiwa_ranges = iiwa_upper_limits - iiwa_lower_limits
iiwa_mids = 0.3 * (iiwa_upper_limits + iiwa_lower_limits)

# use position drive for all dofs
iiwa_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
iiwa_dof_props["stiffness"][:7].fill(400.0)
iiwa_dof_props["damping"][:7].fill(40.0)
# grippers
iiwa_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
iiwa_dof_props["stiffness"][7:].fill(800.0)
iiwa_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
iiwa_num_dofs = gym.get_asset_dof_count(iiwa_asset)
default_dof_pos = np.zeros(iiwa_num_dofs, dtype=np.float32)
default_dof_pos[:7] = iiwa_mids[:7]
# grippers open
default_dof_pos[7:] = iiwa_upper_limits[7:]

default_dof_state = np.zeros(iiwa_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# get link index of robotiq hand, which we will use as end effector
iiwa_link_dict = gym.get_asset_rigid_body_dict(iiwa_asset)
iiwa_hand_index = iiwa_link_dict["iiwa_link_ee"]

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

iiwa_pose = gymapi.Transform()
iiwa_pose.p = gymapi.Vec3(0, 0, 0)

# fsm parameters:
fsm_device = 'cpu'

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)

    # add iiwa
    iiwa_handle = gym.create_actor(env, iiwa_asset, iiwa_pose, "iiwa", 0, 0)

    # set dof properties
    gym.set_actor_dof_properties(env, iiwa_handle, iiwa_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, iiwa_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, iiwa_handle, default_dof_pos)

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, iiwa_handle, "iiwa_link_ee", gymapi.DOMAIN_SIM)

cam_target = gymapi.Vec3(-1, 0, 0.5)
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# get rigid body state tensor
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# simulation loop
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
