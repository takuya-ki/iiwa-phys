#!/usr/bin/env python3

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

import os
import math
import torch
import numpy as np


def quat_axis(q, axis=0):
    basis_vec = torch.zeros(q.shape[0], 3, device=q.device)
    basis_vec[:, axis] = 1
    return quat_rotate(q, basis_vec)


def orientation_error(desired, current):
    cc = quat_conjugate(current)
    q_r = quat_mul(desired, cc)
    return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)


def cube_grasping_yaw(q, corners):
    """ returns horizontal rotation required to grasp cube """
    rc = quat_rotate(q, corners)
    yaw = (torch.atan2(rc[:, 1], rc[:, 0]) - 0.25 * math.pi) % (0.5 * math.pi)
    theta = 0.5 * yaw
    w = theta.cos()
    x = torch.zeros_like(w)
    y = torch.zeros_like(w)
    z = theta.sin()
    yaw_quats = torch.stack([x, y, z, w], dim=-1)
    return yaw_quats


def control_ik(dpose):
    global damping, j_eef, num_envs
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


def control_osc(dpose):
    global kp, kd, kp_null, kd_null, default_dof_pos_tensor, mm, j_eef, num_envs, dof_pos, dof_vel, hand_vel
    mm_inv = torch.inverse(mm)
    m_eef_inv = j_eef @ mm_inv @ torch.transpose(j_eef, 1, 2)
    m_eef = torch.inverse(m_eef_inv)
    u = torch.transpose(j_eef, 1, 2) @ m_eef @ (
        kp * dpose - kd * hand_vel.unsqueeze(-1))

    # Nullspace control torques `u_null` prevents large changes in joint configuration
    # They are added into the nullspace of OSC so that the end effector orientation remains constant
    # roboticsproceedings.org/rss07/p31.pdf
    j_eef_inv = m_eef @ j_eef @ mm_inv
    u_null = kd_null * -dof_vel + kp_null * (
        (default_dof_pos_tensor.view(1, -1, 1) - dof_pos + np.pi) % (2 * np.pi) - np.pi)
    u_null = u_null[:, :7]
    u_null = mm @ u_null
    u += (torch.eye(7, device=device).unsqueeze(0) - torch.transpose(j_eef, 1, 2) @ j_eef_inv) @ u_null
    return u.squeeze(-1)


# set random seed
np.random.seed(42)

torch.set_printoptions(precision=4, sci_mode=False)

# acquire gym interface
gym = gymapi.acquire_gym()

# parse arguments

# Add custom arguments
custom_parameters = [
    {"name": '--controller',
     "type": str,
     "default": 'ik',
     "help": "Controller to use for LBR iiwa. Options are {ik, osc}"},
    {"name": '--num_envs',
     "type": int,
     "default": 1,
     "help": "Number of environments to create"},
    {"name": '--sdf',
     "action": 'store_true',
     "help": "Use SDF-based collision check"}]

args = gymutil.parse_arguments(
    description="LBR iiwa",
    custom_parameters=custom_parameters,
)

# Grab controller
controller = args.controller
assert controller in {"ik", "osc"}, f"Invalid controller specified -- options are (ik, osc). Got: {controller}"

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
    sim_params.physx.num_position_iterations = 8
    sim_params.physx.num_velocity_iterations = 1
    sim_params.physx.rest_offset = 0.0
    sim_params.physx.contact_offset = 0.001
    sim_params.physx.friction_offset_threshold = 0.001
    sim_params.physx.friction_correlation_distance = 0.0005
    sim_params.physx.num_threads = args.num_threads
    sim_params.physx.use_gpu = args.use_gpu
else:
    raise Exception("This example can only be used with PhysX")

# Set controller parameters
# IK params
damping = 0.3

# OSC params
kp = 150.
kd = 2.0 * np.sqrt(kp)
kp_null = 10.
kd_null = 2.0 * np.sqrt(kp_null)

# create sim
sim = gym.create_sim(args.compute_device_id, args.graphics_device_id, args.physics_engine, sim_params)
if sim is None:
    raise Exception("Failed to create sim")

# create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
if viewer is None:
    raise Exception("Failed to create viewer")

asset_root = '/dataset'

# create table asset
table_dims = gymapi.Vec3(0.8, 1.0, 0.4)
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True
table_asset = gym.create_box(sim, table_dims.x, table_dims.y, table_dims.z, asset_options)

# create box asset
box_size = 0.03
asset_options = gymapi.AssetOptions()
box_asset = gym.create_box(sim, box_size, box_size, box_size, asset_options)

# load iiwa asset
sdf_label = ''
if args.sdf:
    sdf_label = '_sdf'

iiwa_asset_file = os.path.join(
    'urdfs', 'iiwa14_rqhe' + sdf_label + '.urdf')

asset_options = gymapi.AssetOptions()
asset_options.armature = 0.01
asset_options.fix_base_link = True
asset_options.disable_gravity = True
asset_options.flip_visual_attachments = False
asset_options.use_mesh_materials = True
asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
asset_options.vhacd_enabled = True
asset_options.convex_decomposition_from_submeshes = True
iiwa_asset = gym.load_asset(sim, asset_root, iiwa_asset_file, asset_options)

# configure iiwa dofs
# iiwa_dof_names = gym.get_asset_dof_names(iiwa_asset)
iiwa_dof_props = gym.get_asset_dof_properties(iiwa_asset)
iiwa_lower_limits = iiwa_dof_props["lower"]
iiwa_upper_limits = iiwa_dof_props["upper"]
iiwa_ranges = iiwa_upper_limits - iiwa_lower_limits
# iiwa_mids = 0.3 * (iiwa_upper_limits + iiwa_lower_limits)

# use position drive for all dofs
if controller == 'ik':
    iiwa_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
    iiwa_dof_props["stiffness"][:7].fill(400.0)
    iiwa_dof_props["damping"][:7].fill(40.0)
else: # osc
    iiwa_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_EFFORT)
    iiwa_dof_props["stiffness"][:7].fill(0.0)
    iiwa_dof_props["damping"][:7].fill(0.0)
# grippers
iiwa_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
iiwa_dof_props["stiffness"][7:].fill(400.0)
iiwa_dof_props["damping"][7:].fill(40.0)

# default dof states and position targets
iiwa_num_dofs = gym.get_asset_dof_count(iiwa_asset)
default_dof_pos = np.zeros(iiwa_num_dofs, dtype=np.float32)
# default_dof_pos[:7] = iiwa_mids[:7]
default_dof_pos[:7] = [0, np.pi/7., 0., -2*np.pi/3., 0., -np.pi/5, np.pi/2.]
spacing = 1.0
cam_pos = gymapi.Vec3(2.0, 0, 1.5)

# grippers open
default_dof_pos[7:] = iiwa_upper_limits[7:]

default_dof_state = np.zeros(iiwa_num_dofs, gymapi.DofState.dtype)
default_dof_state["pos"] = default_dof_pos

# send to torch
default_dof_pos_tensor = to_torch(default_dof_pos, device=device)

# get link index of panda hand, which we will use as end effector
iiwa_link_dict = gym.get_asset_rigid_body_dict(iiwa_asset)
iiwa_hand_index = iiwa_link_dict["iiwa_link_ee"]
# iiwa_num_dofs == 9
# iiwa_hand_index == 10

# configure env grid
num_envs = args.num_envs
num_per_row = int(math.sqrt(num_envs))
env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
env_upper = gymapi.Vec3(spacing, spacing, spacing)
print("Creating %d environments" % num_envs)

iiwa_pose = gymapi.Transform()
table_pose = gymapi.Transform()
box_pose = gymapi.Transform()

iiwa_pose.p = gymapi.Vec3(0, 0, 0.3)
table_pose.p = gymapi.Vec3(0.5, 0.0, 0.5 * table_dims.z)

envs = []
box_idxs = []
hand_idxs = []
hand_tip_idxs = []
init_pos_list = []
init_rot_list = []
init_tip_pos_list = []
init_tip_rot_list = []

# add ground plane
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
gym.add_ground(sim, plane_params)

for i in range(num_envs):
    # create env
    env = gym.create_env(sim, env_lower, env_upper, num_per_row)
    envs.append(env)

    # add table
    table_handle = gym.create_actor(env, table_asset, table_pose, "table", i, 0)

    # add box
    box_pose.p.x = table_pose.p.x + np.random.uniform(-0.2, 0.1)
    box_pose.p.y = table_pose.p.y + np.random.uniform(-0.3, 0.3)
    box_pose.p.z = table_dims.z + 0.5 * box_size
    box_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.random.uniform(-math.pi, math.pi))
    box_handle = gym.create_actor(env, box_asset, box_pose, "box", i, 0)
    color = gymapi.Vec3(np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1))
    gym.set_rigid_body_color(env, box_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)

    # get global index of box in rigid body state tensor
    box_idx = gym.get_actor_rigid_body_index(env, box_handle, 0, gymapi.DOMAIN_SIM)
    box_idxs.append(box_idx)

    # add iiwa
    iiwa_handle = gym.create_actor(env, iiwa_asset, iiwa_pose, "iiwa", i, 2)

    # set dof properties
    gym.set_actor_dof_properties(env, iiwa_handle, iiwa_dof_props)

    # set initial dof states
    gym.set_actor_dof_states(env, iiwa_handle, default_dof_state, gymapi.STATE_ALL)

    # set initial position targets
    gym.set_actor_dof_position_targets(env, iiwa_handle, default_dof_pos)

    # get inital hand pose
    hand_handle = gym.find_actor_rigid_body_handle(env, iiwa_handle, "eef_root")
    hand_pose = gym.get_rigid_transform(env, hand_handle)
    init_pos_list.append([hand_pose.p.x, hand_pose.p.y, hand_pose.p.z])
    init_rot_list.append([hand_pose.r.x, hand_pose.r.y, hand_pose.r.z, hand_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_idx = gym.find_actor_rigid_body_index(env, iiwa_handle, "eef_root", gymapi.DOMAIN_SIM)
    hand_idxs.append(hand_idx)

    # get inital hand pose
    hand_tip_handle = gym.find_actor_rigid_body_handle(env, iiwa_handle, "eef_tip")
    hand_tip_pose = gym.get_rigid_transform(env, hand_tip_handle)
    init_tip_pos_list.append([hand_tip_pose.p.x, hand_tip_pose.p.y, hand_tip_pose.p.z])
    init_tip_rot_list.append([hand_tip_pose.r.x, hand_tip_pose.r.y, hand_tip_pose.r.z, hand_tip_pose.r.w])

    # get global index of hand in rigid body state tensor
    hand_tip_idx = gym.find_actor_rigid_body_index(env, iiwa_handle, "eef_tip", gymapi.DOMAIN_SIM)
    hand_tip_idxs.append(hand_tip_idx)

# point camera at middle env
cam_target = gymapi.Vec3(-1, 0, 0.5)
middle_env = envs[num_envs // 2 + num_per_row // 2]
gym.viewer_camera_look_at(viewer, env, cam_pos, cam_target)

# ==== prepare tensors =====
# from now on, we will use the tensor API that can run on CPU or GPU
gym.prepare_sim(sim)

# initial hand position and orientation tensors
init_pos = torch.Tensor(init_pos_list).view(num_envs, 3).to(device)
init_rot = torch.Tensor(init_rot_list).view(num_envs, 4).to(device)
init_tip_pos = torch.Tensor(init_tip_pos_list).view(num_envs, 3).to(device)
init_tip_rot = torch.Tensor(init_tip_rot_list).view(num_envs, 4).to(device)

# hand orientation for grasping
down_q = torch.stack(num_envs * [torch.tensor([1.0, 0.0, 0.0, 0.0])]).to(device).view((num_envs, 4))

# box corner coords, used to determine grasping yaw
box_half_size = 0.5 * box_size
corner_coord = torch.Tensor([box_half_size, box_half_size, box_half_size])
corners = torch.stack(num_envs * [corner_coord]).to(device)

# downard axis
down_dir = torch.Tensor([0, 0, -1]).to(device).view(1, 3)

# get jacobian tensor
# for fixed-base iiwa, tensor has shape (num envs, 10, 6, 9) = (num_env, num_link, 6, num_dof)
_jacobian = gym.acquire_jacobian_tensor(sim, "iiwa")
jacobian = gymtorch.wrap_tensor(_jacobian)

# jacobian entries corresponding to iiwa hand
j_eef = jacobian[:, iiwa_hand_index - 1, :, :7]
# iiwa_link_dict == {'base_link': 1, 'eef_root': 16, 'eef_tip': 17, 'gripper_tip_link': 12, 'hand_e_link': 13, 'hande_left_finger': 14, 'hande_right_finger': 15, 'iiwa_link_0': 2, 'iiwa_link_1': 3, 'iiwa_link_2': 4, 'iiwa_link_3': 5, 'iiwa_link_4': 6, 'iiwa_link_5': 7, 'iiwa_link_6': 8, 'iiwa_link_7': 9, 'iiwa_link_ee': 10, 'robotiq_coupler': 11, 'world': 0}

# get mass matrix tensor
_massmatrix = gym.acquire_mass_matrix_tensor(sim, "iiwa")
mm = gymtorch.wrap_tensor(_massmatrix)
mm = mm[:, :7, :7]  # only need elements corresponding to the iiwa arm

# get rigid body state tensor
# rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
_rb_states = gym.acquire_rigid_body_state_tensor(sim)
rb_states = gymtorch.wrap_tensor(_rb_states)

# get dof state tensor
# DOF state of the system [num. of instances, num. of dof, 2] where last index: pos, vel
_dof_states = gym.acquire_dof_state_tensor(sim)
dof_states = gymtorch.wrap_tensor(_dof_states)
dof_pos = dof_states[:, 0].view(num_envs, 9, 1)
dof_vel = dof_states[:, 1].view(num_envs, 9, 1)

# Create a tensor noting whether the hand should return to the initial position
hand_restart = torch.full([num_envs], False, dtype=torch.bool).to(device)

# Set action tensors
pos_action = torch.zeros_like(dof_pos).squeeze(-1)
effort_action = torch.zeros_like(pos_action)

# simulation loop
old_tip_pos = init_tip_pos.clone()
while not gym.query_viewer_has_closed(viewer):

    # step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # refresh tensors
    gym.refresh_rigid_body_state_tensor(sim)
    gym.refresh_dof_state_tensor(sim)
    gym.refresh_jacobian_tensors(sim)
    gym.refresh_mass_matrix_tensors(sim)

    box_pos = rb_states[box_idxs, :3]
    box_rot = rb_states[box_idxs, 3:7]  # quaternion

    hand_pos = rb_states[hand_idxs, :3]
    hand_rot = rb_states[hand_idxs, 3:7]  # quaternion
    hand_vel = rb_states[hand_idxs, 7:]  # v and omega

    hand_tip_pos = rb_states[hand_tip_idxs, :3]
    hand_tip_rot = rb_states[hand_tip_idxs, 3:7]  # quaternion
    hand_tip_vel = rb_states[hand_tip_idxs, 7:]  # v and omega

    to_box = box_pos - hand_pos
    box_dist = torch.norm(to_box, dim=-1).unsqueeze(-1)
    box_dir = to_box / box_dist
    box_dot = box_dir @ down_dir.view(3, 1)

    # how far the hand should be from box for grasping
    grasp_offset = 0.12 if controller == 'ik' else 0.125

    # determine if we're holding the box (grippers are closed and box is near)
    gripper_sep = 0.049 - (dof_pos[:, 6] + dof_pos[:, 7]) # weird, full open minus distance of each finger from its starting pose
    gripped = (gripper_sep < box_size) & (box_dist < grasp_offset + 0.5 * box_size)

    yaw_q = cube_grasping_yaw(box_rot, corners)
    box_yaw_dir = quat_axis(yaw_q, 0)
    hand_yaw_dir = quat_axis(hand_rot, 0)
    # torch.bmm performs a batch matrix-matrix product of matrices stored in input and mat2.
    yaw_dot = torch.bmm(box_yaw_dir.view(num_envs, 1, 3), hand_yaw_dir.view(num_envs, 3, 1)).squeeze(-1)

    # determine if we have reached the initial position; if so allow the hand to start moving to the box
    to_init = init_pos - hand_pos
    init_dist = torch.norm(to_init, dim=-1)
    hand_restart = (hand_restart & (init_dist > 0.02)).squeeze(-1)
    return_to_start = (hand_restart | gripped.squeeze(-1)).unsqueeze(-1)

    # if hand is above box, descend to grasp offset
    # otherwise, seek a position above the box
    above_box = ((box_dot >= 0.99) & (yaw_dot >= 0.95) & (box_dist < grasp_offset * 3)).squeeze(-1)
    grasp_pos = box_pos.clone()
    # https://pytorch.org/docs/stable/generated/torch.where.html
    grasp_pos[:, 2] = torch.where(above_box, box_pos[:, 2] + grasp_offset, box_pos[:, 2] + grasp_offset * 2.5)

    # compute goal position and orientation
    goal_pos = torch.where(return_to_start, init_pos, grasp_pos)
    goal_rot = torch.where(return_to_start, init_rot, quat_mul(down_q, quat_conjugate(yaw_q)))

    # compute position and orientation error
    pos_err = goal_pos - hand_pos
    orn_err = orientation_error(goal_rot, hand_rot)
    dpose = torch.cat([pos_err, orn_err], -1).unsqueeze(-1)

    # Deploy control based on type
    if controller == 'ik':
        gym.clear_lines(viewer)
        pos_action[:, :7] = dof_pos.squeeze(-1)[:, :7] + control_ik(dpose)
        for env in envs:
            gym.add_lines(
                viewer,
                env,
                1,
                [old_tip_pos[0][0],
                 old_tip_pos[0][1],
                 old_tip_pos[0][2],
                 hand_tip_pos[0][0],
                 hand_tip_pos[0][1],
                 hand_tip_pos[0][2]],
                [0.85, 0.1, 0.1])
        old_tip_pos = hand_tip_pos.clone()
    else:  # osc
        effort_action[:, :7] = control_osc(dpose)

    # gripper actions depend on distance between hand and box
    close_gripper = (box_dist < grasp_offset + 0.02) | gripped
    # always open the gripper above a certain height, dropping the box and restarting from the beginning
    hand_restart = hand_restart | (box_pos[:, 2] > table_dims.z+0.2)

    keep_going = torch.logical_not(hand_restart)
    close_gripper = close_gripper & keep_going.unsqueeze(-1)
    grip_acts = torch.where(close_gripper, torch.Tensor([[0.025, 0.025]] * num_envs).to(device), torch.Tensor([[0.0, 0.0]] * num_envs).to(device))
    pos_action[:, 7:9] = grip_acts

    # Deploy actions
    gym.set_dof_position_target_tensor(sim, gymtorch.unwrap_tensor(pos_action))
    gym.set_dof_actuation_force_tensor(sim, gymtorch.unwrap_tensor(effort_action))

    # update viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, False)
    gym.sync_frame_time(sim)

# cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
