#!/usr/bin/env python3

from isaacgym import gymapi, gymutil, gymtorch
from isaacgym.torch_utils import *

import os
import copy
import math
import time
import torch
import random
import pickle
import quaternion
import numpy as np
import networkx as nx
from operator import itemgetter


class RRT(object):

    def __init__(self, args, start_conf, goal_conf):

        self.ctrl_type = args.ctrl
        self.use_gpu = args.use_gpu
        self.use_gpu_pipeline = args.use_gpu_pipeline
        self.num_threads = args.num_threads
        self.compute_device_id = args.compute_device_id
        self.graphics_device_id = args.graphics_device_id
        self.physics_engine = args.physics_engine
        self.num_envs = args.num_envs
        self.device = args.sim_device
        self.damping = 0.  # 0.3
        self.use_sdf = args.sdf
        self.sdf_label = ''
        if args.sdf:
            self.sdf_label = '_sdf'
        self.start_conf = start_conf
        self.goal_conf = goal_conf
        self.roadmaps = [nx.Graph() for _ in range(self.num_envs)]

        self._initialize_sim()
        self._asset_root = "/dataset"
        self._load_obstacles()
        self._load_iiwa()
        self._configure_iiwa_dofs()
        self._initialize_envs()
        self._initialize_viewer()
        self._prepare_start_planning()

        # while not self.gym.query_viewer_has_closed(self.viewer):
        #     self._update_gym_sim(self.gym_wait_steps)

    def _unit_vector(self, vector, toggle_length=False):
        """ Returns a unit vector """
        length = np.linalg.norm(vector)
        if math.isclose(length, 0):
            if toggle_length:
                return 0.0, np.zeros_like(vector)
            else:
                return np.zeros_like(vector)
        if toggle_length:
            return length, vector / np.linalg.norm(vector)
        else:
            return vector / np.linalg.norm(vector)

    def _violates_limit(self, lims, value):
        """ Checks if the value violates the limit """
        lower, upper = lims
        if (value < lower) or (upper < value):
            return True
        return False

    def _violates_limits(self, limits, values):
        """ Checks limit violations """
        is_violated = any(self._violates_limit(lims, value)
                for lims, value in zip(limits, values))
        return is_violated

    def _cleanup_sim(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

    def _initialize_sim(self):
        # set random seed
        np.random.seed(42)
        torch.set_printoptions(precision=4, sci_mode=False)
        # acquire gym interface
        self.gym = gymapi.acquire_gym()
        # force GPU:
        if not self.use_gpu or self.use_gpu_pipeline:
            print("Forcing GPU sim - CPU sim not supported by SDF")
            self.use_gpu = True
            self.use_gpu_pipeline = True
        # configure sim
        sim_params = gymapi.SimParams()
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.8)
        sim_params.dt = 1.0 / 60.0  # 1.0 / 60.0
        sim_params.substeps = 2  # 2
        if self.ctrl_type == 'pos':
            sim_params.use_gpu_pipeline = self.use_gpu_pipeline
            # TODO if you set the gym_wait_steps more than 1
            # you need to get the contact force in the loop of _update_gym_sim()
            self.gym_wait_steps = 60
        elif self.ctrl_type == 'none':
            # Forcing CPU pipeline for DOF_MODE_NONE
            sim_params.use_gpu_pipeline = False
            self.gym_wait_steps = 1
        if self.physics_engine == gymapi.SIM_PHYSX:
            sim_params.physx.solver_type = 1
            sim_params.physx.num_position_iterations = 32  # 8
            sim_params.physx.num_velocity_iterations = 1
            sim_params.physx.friction_offset_threshold = 0.001
            sim_params.physx.friction_correlation_distance = 0.0005
            sim_params.physx.num_threads = self.num_threads
            sim_params.physx.use_gpu = self.use_gpu

            # contact-detection-related parameters
            # https://github.com/NVIDIA-Omniverse/IsaacGymEnvs/blob/main/docs/factory.md
            sim_params.physx.rest_offset = 0.0  # 0.0
            sim_params.physx.contact_offset = 0.005  # 0.005
            sim_params.physx.max_gpu_contact_pairs = 1024*1024  # 1024*1024
            sim_params.physx.default_buffer_size_multiplier = 8  # 8
        else:
            raise Exception("This example can only be used with PhysX")

        # create sim
        self.sim = self.gym.create_sim(
            self.compute_device_id,
            self.graphics_device_id,
            self.physics_engine,
            sim_params)
        if self.sim is None:
            raise Exception("Failed to create sim")

    def _initialize_viewer(self):
        # point camera at middle env
        self.viewer = self.gym.create_viewer(
            self.sim, gymapi.CameraProperties())
        if self.viewer is None:
            raise Exception("Failed to create viewer")
        cam_pos = gymapi.Vec3(1.5, 0, 1.5)
        cam_target = gymapi.Vec3(-1, 0, 0.5)
        for env in self.envs:
            self.gym.viewer_camera_look_at(
                self.viewer, env, cam_pos, cam_target)

        # visualize wireframe spheres at particle positions
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)
        self.point_pose = gymapi.Transform()
        self.point_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0)

    def _initialize_envs(self):
        # default dof states and position targets
        self.iiwa_num_dofs = self.gym.get_asset_dof_count(self.iiwa_asset)
        default_dof_pos = np.zeros(self.iiwa_num_dofs, dtype=np.float32)
        # default_dof_pos[:7] = iiwa_mids[:7]
        default_dof_pos[:7] = list(self.start_conf)
        # grippers open
        default_dof_pos[7:] = self.iiwa_upper_limits[7:]
        default_dof_state = np.zeros(self.iiwa_num_dofs, gymapi.DofState.dtype)
        default_dof_state["pos"] = default_dof_pos

        iiwa_link_dict = self.gym.get_asset_rigid_body_dict(self.iiwa_asset)
        self.iiwa_hand_index = iiwa_link_dict["eef_tip"]

        # configure env grid
        num_per_row = int(math.sqrt(self.num_envs))
        spacing = 1.0
        env_lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        env_upper = gymapi.Vec3(spacing, spacing, spacing)
        print("Creating %d environments" % self.num_envs)
        # initial pose
        iiwa_pose = gymapi.Transform()
        iiwa_pose.p = gymapi.Vec3(0, 0, 0)
        # add ground plane
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

        self.envs = []
        self.iiwa_handles = []
        self.obstacle_handles = [[] for _ in range(self.num_envs)]
        self.hand_tip_handles = []
        self.cf_target_handles = []
        self.ft_sensors = []
        for i in range(self.num_envs):
            # create env
            env = self.gym.create_env(
                self.sim, env_lower, env_upper, num_per_row)
            self.envs.append(env)

            # add obstacles and get global indeces
            for j, obs_asst in enumerate(self.obstacles_assets):
                obstacle_handle = self.gym.create_actor(
                    env, obs_asst[0], obs_asst[1], "obstacle"+str(j), i, 0)
                self.obstacle_handles[i].append(obstacle_handle)

            # add iiwa
            iiwa_handle = self.gym.create_actor(
                env, self.iiwa_asset, iiwa_pose, "iiwa", i, 2)
            self.iiwa_handles.append(iiwa_handle)
            # add sensors
            for sensor_idx in self.sensor_idxs:
                self.ft_sensors.append(self.gym.get_actor_force_sensor(
                    env, iiwa_handle, sensor_idx))
            # set dof properties
            self.gym.set_actor_dof_properties(
                env, iiwa_handle, self.iiwa_dof_props)
            # set initial dof states
            self.gym.set_actor_dof_states(
                env, iiwa_handle, default_dof_state, gymapi.STATE_ALL)
            # set initial position targets
            self.gym.set_actor_dof_position_targets(
                env, iiwa_handle, default_dof_pos)

            # get global index of hand in rigid body state tensor
            hand_tip_handle = self.gym.find_actor_rigid_body_index(
                env, iiwa_handle, "eef_tip", gymapi.DOMAIN_SIM)
            self.hand_tip_handles.append(hand_tip_handle)
            hand_handle = self.gym.find_actor_rigid_body_index(
                env, iiwa_handle, "rqhe", gymapi.DOMAIN_SIM)
            obj_handle = self.gym.find_actor_rigid_body_index(
                env, iiwa_handle, "obj", gymapi.DOMAIN_SIM)
            self.cf_target_handles.append(hand_handle)

    def _load_obstacles(self):

        self.obstacles_assets = []

        obstacle_size = 0.2
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True

        self.obstacle_positions = []

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.9
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.9
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.9
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.8
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.8
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.8
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 1.
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 1.
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 1.
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.7
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.7
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = 0.2
        obstacle_pose.p.y = 0.
        obstacle_pose.p.z = 0.7
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.1
        obstacle_pose.p.z = 1.
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.1
        obstacle_pose.p.z = 0.9
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.1
        obstacle_pose.p.z = 0.8
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

        obstacle_asset = self.gym.create_box(
            self.sim, obstacle_size, obstacle_size, obstacle_size, asset_options)
        obstacle_pose = gymapi.Transform()
        obstacle_pose.p.x = -0.2
        obstacle_pose.p.y = 0.1
        obstacle_pose.p.z = 0.7
        obstacle_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0)
        self.obstacles_assets.append([obstacle_asset, obstacle_pose])
        self.obstacle_positions.append(obstacle_pose)

    def _load_iiwa(self):

        iiwa_asset_file = os.path.join(
            'urdfs', 'iiwa14_rqhe' + self.sdf_label + '.urdf')

        # setting the asset option
        asset_options = gymapi.AssetOptions()
        asset_options.armature = 0.01
        asset_options.fix_base_link = True
        asset_options.disable_gravity = True
        asset_options.flip_visual_attachments = False
        asset_options.use_mesh_materials = True
        if not self.use_sdf:
            asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            asset_options.vhacd_enabled = True
            asset_options.convex_decomposition_from_submeshes = True
            asset_options.vhacd_params.resolution = 30000
            asset_options.vhacd_params.max_convex_hulls = 30000
            asset_options.vhacd_params.max_num_vertices_per_ch = 30000

        self.iiwa_asset = self.gym.load_asset(
            self.sim, self._asset_root, iiwa_asset_file, asset_options)

        # setting the force sensors
        self.num_bodies = self.gym.get_asset_rigid_body_count(self.iiwa_asset)
        self.iiwa_body_names = [
            self.gym.get_asset_rigid_body_name(self.iiwa_asset, i) for i in range(self.num_bodies)]
        print(self.iiwa_body_names)

        # create force sensors attached to the "feet"
        body_indices = [self.gym.find_asset_rigid_body_index(self.iiwa_asset, name) for name in self.iiwa_body_names]
        sensor_pose = gymapi.Transform()
        self.sensor_idxs = []
        for body_idx in body_indices:
            self.sensor_idxs.append(self.gym.create_asset_force_sensor(self.iiwa_asset, body_idx, sensor_pose))

    def _configure_iiwa_dofs(self):
        # configure iiwa dofs
        self.iiwa_dof_props = self.gym.get_asset_dof_properties(
            self.iiwa_asset)
        self.iiwa_lower_limits = self.iiwa_dof_props["lower"]
        self.iiwa_upper_limits = self.iiwa_dof_props["upper"]
        self.jlims = [(l, u) for l, u in zip(
            self.iiwa_lower_limits[:7], self.iiwa_upper_limits[:7])]

        if self.ctrl_type == 'pos':
            # use position drive for all dofs
            self.iiwa_dof_props["driveMode"][:7].fill(gymapi.DOF_MODE_POS)
            self.iiwa_dof_props["stiffness"][:7].fill(2000.0)
            self.iiwa_dof_props["damping"][:7].fill(200.0)
            # grippers
            self.iiwa_dof_props["driveMode"][7:].fill(gymapi.DOF_MODE_POS)
            self.iiwa_dof_props["stiffness"][7:].fill(2000.0)
            self.iiwa_dof_props["damping"][7:].fill(200.0)
        elif self.ctrl_type == 'none':
            # DOF_MODE_NONE : The DOF is free to move without any controls.
            self.iiwa_dof_props["driveMode"].fill(gymapi.DOF_MODE_NONE)
            self.iiwa_dof_props["stiffness"].fill(0.0)
            self.iiwa_dof_props["damping"].fill(0.0)
        else:
            print("Error: This controller", self.ctrl_type, "has not implemented yet.")
            exit(-1)

    def _deploy_gym_actions(self, conf, env_i, sleep_sec=0):
        if self.ctrl_type == 'pos':
            self.pos_action[env_i, :7] = torch.Tensor(conf)
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.pos_action))
        elif self.ctrl_type == 'none':
            iiwa_handle = self.iiwa_handles[env_i]
            env = self.envs[env_i]
            self.set_dof_pos[:7] = list(copy.deepcopy(conf))
            self.set_dof_state["pos"] = self.set_dof_pos
            self.gym.set_actor_dof_states(
                env, iiwa_handle, self.set_dof_state, gymapi.STATE_ALL)
            self.gym.set_actor_dof_position_targets(
                env, iiwa_handle, self.set_dof_pos)
        self._update_gym_sim(self.gym_wait_steps, sleep_sec=sleep_sec)

    def _deploy_gym_actions_envs(self, confs, env_ids):
        if self.ctrl_type == 'pos':
            for conf, ei in zip(confs, env_ids):
                self.pos_action[ei, :7] = torch.Tensor(conf)
            self.gym.set_dof_position_target_tensor(
                self.sim, gymtorch.unwrap_tensor(self.pos_action))
        elif self.ctrl_type == 'none':
            for conf, ei in zip(confs, env_ids):
                iiwa_handle = self.iiwa_handles[ei]
                env = self.envs[ei]
                self.set_dof_pos[:7] = list(copy.deepcopy(conf))
                self.set_dof_state["pos"] = self.set_dof_pos
                self.gym.set_actor_dof_states(
                    env, iiwa_handle, self.set_dof_state, gymapi.STATE_ALL)
                self.gym.set_actor_dof_position_targets(
                    env, iiwa_handle, self.set_dof_pos)
        self._update_gym_sim(self.gym_wait_steps)

    def _get_contact_forces(self):
        # collision checks
        self.gym.refresh_net_contact_force_tensor(self.sim)
        net_cf = gymtorch.wrap_tensor(
            self.gym.acquire_net_contact_force_tensor(self.sim))
        return net_cf

    def _is_collided(self, conf, coll_th, env_i):
        """
            Examines if joint values of the given conf are in ranges.
            Or else, will compute fk and carry out collision checking.
        """

        if not self._violates_limits(self.jlims, conf):
            self._deploy_gym_actions(conf, env_i)
            iiwa_body_colls = [
                [True if ((c > coll_th) or (c < -coll_th)) else False for c in cf]
                 for cf in self.vec_sensor_tensor[:][:3]]
            return np.any(iiwa_body_colls)  # collided
        else:
            return True

    def _sample_conf(self, rand_rates, default_conf, n_envs):
        flags = [random.randint(0, 99) < rand_rates[i] for i in range(n_envs)]
        return [np.array(tuple([np.random.uniform(low=self.jlims[i][0], high=self.jlims[i][1])
                for i in range(len(default_conf))])) if flag else default_conf for flag in flags]

    def _get_nearest_nid(self, roadmap, new_conf):
        """ Converts to numpy to accelerate access """

        nodes_dict = dict(roadmap.nodes(data='conf'))
        nodes_key_list = list(nodes_dict.keys())
        nodes_value_list = list(nodes_dict.values())
        conf_array = np.array(nodes_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        min_dist_nid = np.argmin(diff_conf_array)
        return nodes_key_list[min_dist_nid]

    def _extend_conf(self, conf1, conf2, ext_dist, exact_end=True):
        """ Extends the configuration """

        norm, vec = self._unit_vector(conf2 - conf1, toggle_length=True)
        if not exact_end:
            nval = math.ceil(norm / ext_dist)
            nval = 1 if nval == 0  else nval  # at least include itself
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
        else:
            nval = math.floor(norm / ext_dist)
            nval = 1 if nval == 0  else nval  # at least include itself
            conf_array = np.linspace(conf1, conf1 + nval * ext_dist * vec, nval)
            conf_array = np.vstack((conf_array, conf2))
        return list(conf_array)

    def _extend_roadmap(
            self,
            roadmap,
            conf,
            ext_dist,
            coll_th,
            goal_conf,
            env_i,
            exact_end=True):
        """ Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist, exact_end=exact_end)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf, coll_th, env_i):
                return -1
            else:
                new_nid = random.randint(0, 1e16)
                roadmap.add_node(new_nid, conf=new_conf)
                roadmap.add_edge(nearest_nid, new_nid)
                nearest_nid = new_nid
                # check goal
                if self._goal_test(
                        conf=roadmap.nodes[new_nid]['conf'],
                        goal_conf=goal_conf,
                        threshold=ext_dist):
                    print("Goal reached in env", env_i)
                    roadmap.add_node('connection', conf=goal_conf)
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
        else:
            return nearest_nid

    def _goal_test(self, conf, goal_conf, threshold):
        dist = np.linalg.norm(conf - goal_conf)
        if dist <= threshold:
            return True
        else:
            return False

    def _path_from_roadmap(self, roadmap, conf_keyname='conf'):
        try:
            nid_path = nx.shortest_path(roadmap, 'start', 'goal')
            return list(itemgetter(*nid_path)(roadmap.nodes(data=conf_keyname)))
        except nx.NetworkXNoPath:
            return []

    def _smooth_path(
            self,
            path,
            env_i,
            coll_th,
            granularity=2,
            iterations=50):

        smoothed_path = path
        for _ in range(iterations):
            if len(smoothed_path) <= 2:
                return smoothed_path
            i = random.randint(0, len(smoothed_path) - 1)
            j = random.randint(0, len(smoothed_path) - 1)
            if abs(i - j) <= 1:
                continue
            if j < i:
                i, j = j, i
            shortcut = self._extend_conf(smoothed_path[i], smoothed_path[j], granularity)
            if (len(shortcut) <= (j - i) + 1) and all(not self._is_collided(conf=conf, coll_th=coll_th, env_i=env_i) for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
        return smoothed_path

    def _update_gym_sim(self, gym_wait_steps, sleep_sec=0):
        cnt = 0
        # TODO if you set the gym_wait_steps more than 1, you need to get the contact force in this loop
        while cnt < gym_wait_steps:
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            # refresh tensors
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.gym.refresh_mass_matrix_tensors(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            # update viewer
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, False)
            cnt += 1
        time.sleep(sleep_sec)

    def _update_gym_sim_sec(self, wait_sec):
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)
            toc = time.time()
            if wait_sec > 0.0:
                if toc - tic > wait_sec:
                    break

    def _prepare_start_planning(self):
        # from now on, we will use the tensor API that can run on CPU or GPU
        self.gym.prepare_sim(self.sim)

        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "iiwa")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        # jacobian entries corresponding to iiwa hand
        self.j_eef = jacobian[:, self.iiwa_hand_index - 1, :, :7]

        # get mass matrix tensor
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "iiwa")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self.mm = mm[:, :7, :7]  # only need elements corresponding to the iiwa arm

        # get rigid body state tensor
        # rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
        _rb_states = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rb_states = gymtorch.wrap_tensor(_rb_states)

        # get sensor tensor
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor)

        # DOF state of the system [num. of instances, num. of dof, 2]
        # where last index: pos, vel
        _dof_states = self.gym.acquire_dof_state_tensor(self.sim)
        dof_states = gymtorch.wrap_tensor(_dof_states)
        dof_pos = dof_states[:, 0].view(self.num_envs, int(dof_states.size()[0]/self.num_envs), 1)
        # set action tensors
        if self.ctrl_type == 'pos':
            self.pos_action = torch.zeros_like(dof_pos).squeeze(-1)
        elif self.ctrl_type == 'none':
            self.set_dof_pos = np.zeros(
                self.iiwa_num_dofs, dtype=np.float32)
            self.set_dof_state = np.zeros(
                self.iiwa_num_dofs, gymapi.DofState.dtype)

        # visualize wireframe spheres at start and goal positions
        sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * math.pi, 0, 0)
        sphere_pose = gymapi.Transform(r=sphere_rot)

        # deploy actions to move the robot to start configuration
        self._deploy_gym_actions_envs(
            [self.start_conf for _ in range(self.num_envs)],
            [ei for ei in range(self.num_envs)])
        self.start_position = self.rb_states[self.hand_tip_handles, :3][0]
        start_sphere_geom = gymutil.WireframeSphereGeometry(
            0.03, 12, 12, sphere_pose, color=(0, 1, 0))
        start_point_pose = gymapi.Transform()
        start_point_pose.p.x = self.start_position[0]
        start_point_pose.p.y = self.start_position[1]
        start_point_pose.p.z = self.start_position[2]
        start_point_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0)
        for env in self.envs:
            gymutil.draw_lines(
                start_sphere_geom,
                self.gym,
                self.viewer,
                env,
                start_point_pose)

        wait_sec = 3.
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._deploy_gym_actions_envs(
                [self.start_conf for _ in range(self.num_envs)],
                [ei for ei in range(self.num_envs)])
            self._update_gym_sim(self.gym_wait_steps)
            toc = time.time()
            if wait_sec > 0.0:
                if toc - tic > wait_sec:
                    break

        # deploy actions to move the robot to goal configuration
        self._deploy_gym_actions_envs(
            [self.goal_conf for _ in range(self.num_envs)],
            [ei for ei in range(self.num_envs)])
        self.goal_position = self.rb_states[self.hand_tip_handles, :3][0]
        goal_sphere_geom = gymutil.WireframeSphereGeometry(
            0.03, 12, 12, sphere_pose, color=(0, 0, 1))
        goal_point_pose = gymapi.Transform()
        goal_point_pose.p.x = self.goal_position[0]
        goal_point_pose.p.y = self.goal_position[1]
        goal_point_pose.p.z = self.goal_position[2]
        goal_point_pose.r = gymapi.Quat.from_axis_angle(
            gymapi.Vec3(0, 0, 1), 0)
        for env in self.envs:
            gymutil.draw_lines(
                goal_sphere_geom,
                self.gym,
                self.viewer,
                env,
                goal_point_pose)

        wait_sec = 3.
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._deploy_gym_actions_envs(
                [self.goal_conf for _ in range(self.num_envs)],
                [ei for ei in range(self.num_envs)])
            self._update_gym_sim(self.gym_wait_steps)
            toc = time.time()
            if wait_sec > 0.0:
                if toc - tic > wait_sec:
                    break

    def _is_invalid_input(self, start_conf, goal_conf, ext_dist, coll_th):
        # check joint values and goal conf
        if self._is_collided(start_conf, coll_th, 0):
            print("The start robot configuration is in collision!")
            return None
        if self._is_collided(goal_conf, coll_th, 0):
            print("The goal robot configuration is in collision!")
            return None
        if self._goal_test(
                conf=start_conf,
                goal_conf=goal_conf,
                threshold=ext_dist):
            return None

    def plan(
            self,
            ext_dist=0.1,
            coll_th=0.5,
            rand_rate=90,
            annealing_rate=None,
            min_rand_rate=10,
            max_iter=10000,
            max_time=30.0,
            smoothing_iterations=10,  # 50
            visualize=True):

        if self._is_invalid_input(
                self.start_conf,
                self.goal_conf,
                ext_dist,
                coll_th):
            return None

        for roadmap in self.roadmaps:
            roadmap.clear()
            roadmap.add_node('start', conf=self.start_conf)

        iter = 0
        on_env_ids = [ei for ei in range(self.num_envs)]
        on_env_ids_copy = on_env_ids.copy()
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rates = [rand_rate - int(self.roadmaps[ei].number_of_nodes() * annealing_rate) for ei in on_env_ids]
                rand_rates = [rr if rr > min_rand_rate else min_rand_rate for rr in rand_rates]
            else:
                rand_rates = [rand_rate for _ in range(len(on_env_ids))]
            rand_conf = self._sample_conf(
                rand_rates=rand_rates,
                default_conf=self.goal_conf,
                n_envs=len(on_env_ids))

            for i, ei in enumerate(on_env_ids):
                # extend roadmap while checking collsiion
                last_nid = self._extend_roadmap(
                    roadmap=self.roadmaps[ei],
                    conf=rand_conf[i],
                    ext_dist=ext_dist,
                    coll_th=coll_th,
                    goal_conf=self.goal_conf,
                    env_i=ei)
                if last_nid == 'connection':
                    mapping = {'connection': 'goal'}
                    self.roadmaps = [
                        nx.relabel_nodes(rm, mapping)
                        for rm in self.roadmaps]
                    on_env_ids.remove(ei)

            if len(on_env_ids) == 0:
                break

            iter += 1
            if iter > max_iter:
                print("Reach to maximum iteration! Failed to find a path.")
                return None

        print("\nParameter setting:")
        print("\tExtention distance =", ext_dist)
        print("\tRandom generation rate =", rand_rate)
        print("\tMaximum iteration =", max_iter)
        print("\tMaximum time =", max_time)
        print("\tSmoothing iterations =", smoothing_iterations)
        print("\tCollision threshold =", coll_th)

        path_confs = [self._path_from_roadmap(rm) for rm in self.roadmaps]
        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = [
            self._smooth_path(
                path=path_conf,
                env_i=i,
                coll_th=coll_th,
                granularity=ext_dist,
                iterations=smoothing_iterations)
            for i, path_conf in enumerate(path_confs)]
        self.path_confs = smoothed_path_confs

        mean_num_nodes = np.mean([rm.number_of_nodes() for ei, rm in enumerate(self.roadmaps)])
        mean_len_paths = np.mean([len(pc) for ei, pc in enumerate(path_confs)])
        mean_err_confs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(path_confs)])
        mean_len_spaths = np.mean([len(pc) for ei, pc in enumerate(self.path_confs)])
        mean_err_sconfs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(self.path_confs)])

        print("\nGenerated trajectory (mean over the environments):")
        print("\tNumber of nodes =", mean_num_nodes)
        print("\tLength of original paths =", mean_len_paths)
        print("\tError of original configurations =", mean_err_confs)
        print("\tLength of smoothed paths =", mean_len_spaths)
        print("\tError of smoothed configurations =", mean_err_sconfs)

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'gym_rrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path(on_env_ids_copy)
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs

    def _replay_planned_path(self, env_ids):

        pre_pos = [copy.deepcopy(self.start_position) for _ in range(self.num_envs)]
        on_env_ids = copy.deepcopy(env_ids)
        is_drawn = [False for _ in range(self.num_envs)]
        pt_i = [0 for _ in range(self.num_envs)]
        lenpts = [len(self.path_confs[ei]) for ei in range(self.num_envs)]

        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            self._deploy_gym_actions_envs(
                [self.path_confs[oei][pt_i[oei]] for oei in on_env_ids],
                on_env_ids)
            cur_pos = self.rb_states[self.hand_tip_handles, :3][on_env_ids]
            for i, ei in enumerate(on_env_ids):
                # trace the generate path and highlight the path
                if not is_drawn[ei]:
                    self.gym.add_lines(
                        self.viewer,
                        self.envs[ei],
                        1,
                        [pre_pos[ei][0],
                         pre_pos[ei][1],
                         pre_pos[ei][2],
                         cur_pos[i][0],
                         cur_pos[i][1],
                         cur_pos[i][2]],
                        [1., 0., 0.])
                    pre_pos[ei] = copy.deepcopy(cur_pos[i])

                pt_i[ei] += 1
                if pt_i[ei] == lenpts[ei]:
                    is_drawn[ei] = True
                    pt_i[ei] = 0

    def visualize_path(self, env_ids, conf_keyname='conf'):

        near_poslist = [
            {'start': self.start_position,
             'goal': self.goal_position}
            for _ in range(self.num_envs)]
        eg_i = 0
        on_env_ids = copy.deepcopy(env_ids)
        edges = [list(rm.edges) for rm in self.roadmaps] 
        lenegs = [len(e) for e in edges]

        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            self._deploy_gym_actions_envs(
                [self.roadmaps[oei].nodes[edges[oei][eg_i][1]][conf_keyname] for oei in on_env_ids],
                on_env_ids)
            cur_pos = self.rb_states[self.hand_tip_handles, :3][on_env_ids]
            for i, ei in enumerate(on_env_ids):
                near_pos = copy.deepcopy(
                    near_poslist[ei][edges[ei][eg_i][0]])
                self.gym.add_lines(
                    self.viewer,
                    self.envs[ei],
                    1,
                    [near_pos[0],
                     near_pos[1],
                     near_pos[2],
                     cur_pos[i][0],
                     cur_pos[i][1],
                     cur_pos[i][2]],
                    [0., 0., 0.])
                near_poslist[ei][edges[ei][eg_i][1]] = copy.deepcopy(cur_pos[i])

            eg_i += 1
            on_env_ids = [oei for oei in on_env_ids if eg_i < lenegs[oei]]
            if len(on_env_ids) == 0:
                break

        self._replay_planned_path(env_ids)
        self._cleanup_sim()


class naiveRRT(RRT):

    def __init__(self, args, start_conf, goal_conf):
        super().__init__(args, start_conf, goal_conf)

    def _move_node(self, conf1, conf2, ext_dist):
        diff_conf = conf2 - conf1
        new_conf = conf1 + (ext_dist / np.linalg.norm(diff_conf)) * diff_conf
        return new_conf

    def _find_nn(self, tree_nodes, conf):
        min_dist = 100000000000
        final_node = None
        for tree_node in tree_nodes:
            dist = np.linalg.norm(conf - np.array(tree_node))
            if dist < min_dist:
                final_node = np.array(tree_node)
                min_dist = dist
        return final_node

    def _build_adjmat(self, edges, vertices):
        adj_mat = np.zeros((len(vertices), len(vertices)))
        for edge in edges:
            adj_mat[vertices.index(edge[0]), vertices.index(edge[1])] = 1
            adj_mat[vertices.index(edge[1]), vertices.index(edge[0])] = 1
        return adj_mat

    def _find_path(self, s, d, vertex, adj_list, visited, curr_path, ei):
        if len(self.final_path[ei]) > 0:
            return 
        idx = vertex.index(s)
        visited[idx] = 1
        curr_path.append(s)
        if d == s:
            self.final_path[ei] = copy.deepcopy(curr_path)
            return
        else:
            for x in range(len(vertex)):
                if adj_list[idx,x] == 1 and visited[x] == 0:
                    self._find_path(
                        vertex[x], d, vertex, adj_list, visited, curr_path, ei)
        curr_path.pop()
        visited[idx] = 0

    # mehod override
    def plan(
            self,
            ext_dist=0.1,
            coll_th=0.5,
            rand_rate=90,
            annealing_rate=None,
            min_rand_rate=10,
            max_iter=10000,
            max_time=30.0,
            smoothing_iterations=10,  # 50
            visualize=True):

        qstart = np.array(self.start_conf)
        qgoal = np.array(self.goal_conf)
        if self._is_invalid_input(
                qstart,
                qgoal,
                ext_dist,
                coll_th):
            return None

        self.final_path = [[] for _ in range(self.num_envs)]
        self.path = [[tuple(start_conf)] for _ in range(self.num_envs)]
        self.edges = [[] for _ in range(self.num_envs)]
        path_confs = [[] for _ in range(self.num_envs)]

        iter = 0
        on_env_ids = [ei for ei in range(self.num_envs)]
        on_env_ids_copy = on_env_ids.copy()
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # a compute random configuration (annealing random sampling)
            if annealing_rate is not None:
                rand_rates = [rand_rate - int(len(self.path[ei]) * annealing_rate) for ei in on_env_ids]
                rand_rates = [rr if rr > min_rand_rate else min_rand_rate for rr in rand_rates]
            else:
                rand_rates = [rand_rate for _ in range(len(on_env_ids))]
            qrand = self._sample_conf(
                rand_rates=rand_rates,
                default_conf=self.goal_conf,
                n_envs=len(on_env_ids))

            for i, ei in enumerate(on_env_ids):
                # b find nearest vertex to qrand that is already in g
                qnear = self._find_nn(self.path[ei], qrand[i])
                if np.linalg.norm(qnear - qrand[i]) == 0.0:
                    continue
                # c move Dq from qnear to qrand
                qnew = self._move_node(qnear, qrand[i], ext_dist)

                # e if edge is collision-free
                if not self._is_collided(qnew, coll_th, ei):
                    self.path[ei].append(tuple(qnew.reshape(1, -1)[0]))
                    self.edges[ei].append((
                        tuple(qnear.reshape(1, -1)[0]),
                        tuple(qnew.reshape(1, -1)[0])))
                    # d judge that it reach the goal (connect goal)
                    if self._goal_test(qnew, qgoal, ext_dist):
                        print("Goal reached in env", ei)
                        adj_mat = self._build_adjmat(self.edges[ei], self.path[ei])
                        visited = np.zeros(len(self.path[ei]))
                        curr_path = []
                        self._find_path(
                            tuple(self.start_conf),
                            tuple(qnew.reshape(1, -1)[0]),
                            self.path[ei],
                            adj_mat,
                            visited,
                            curr_path,
                            ei)
                        path_confs[ei] = self.final_path[ei]
                        on_env_ids.remove(ei)

            if len(on_env_ids) == 0:
                break

            iter += 1
            if iter > max_iter:
                print("Reach to maximum iteration! Failed to find a path.")
                return None

        print("\nParameter setting:")
        print("\tExtention distance =", ext_dist)
        print("\tRandom generation rate =", rand_rate)
        print("\tMaximum iteration =", max_iter)
        print("\tMaximum time =", max_time)
        print("\tCollision threshold =", coll_th)

        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        path_confs = [[np.array(path_conf) for path_conf in path_confs[0]]]
        self.path_confs = path_confs
        smoothed_path_confs = [
            self._smooth_path(
                path=path_conf,
                env_i=i,
                coll_th=coll_th,
                granularity=ext_dist,
                iterations=smoothing_iterations)
            for i, path_conf in enumerate(path_confs)]
        self.path_confs = smoothed_path_confs

        mean_num_nodes = np.mean([len(p) for ei, p in enumerate(self.path)])
        mean_len_paths = np.mean([len(pc) for ei, pc in enumerate(path_confs)])
        mean_err_confs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(path_confs)])
        mean_len_spaths = np.mean([len(pc) for ei, pc in enumerate(self.path_confs)])
        mean_err_sconfs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(self.path_confs)])

        print("\nGenerated trajectory (mean over the environments):")
        print("\tNumber of nodes =", mean_num_nodes)
        print("\tLength of original paths =", mean_len_paths)
        print("\tError of original configurations =", mean_err_confs)
        print("\tLength of smoothed paths =", mean_len_spaths)
        print("\tError of smoothed configurations =", mean_err_sconfs)

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'gym_naiverrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path(on_env_ids_copy)
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs

    # mehod override
    def visualize_path(self, env_ids):

        near_poslist = [{0: self.start_position} for _ in range(self.num_envs)]
        eg_i = 0
        on_env_ids = copy.deepcopy(env_ids)
        lenegs = [len(self.edges[ei]) for ei in range(self.num_envs)]

        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            self._deploy_gym_actions_envs(
                [self.edges[oei][eg_i][1] for oei in on_env_ids],
                on_env_ids)
            cur_pos = self.rb_states[self.hand_tip_handles, :3][on_env_ids]
            for i, ei in enumerate(on_env_ids):
                near_pos = copy.deepcopy(
                    near_poslist[ei][self.path[ei].index(self.edges[ei][eg_i][0])])
                self.gym.add_lines(
                    self.viewer,
                    self.envs[ei],
                    1,
                    [near_pos[0],
                     near_pos[1],
                     near_pos[2],
                     cur_pos[i][0],
                     cur_pos[i][1],
                     cur_pos[i][2]],
                    [0., 0., 0.])
                near_poslist[ei][self.path[ei].index(self.edges[ei][eg_i][1])] = \
                    copy.deepcopy(cur_pos[i])

            eg_i += 1
            on_env_ids = [oei for oei in on_env_ids if eg_i < lenegs[oei]]
            if len(on_env_ids) == 0:
                break

        self._replay_planned_path(env_ids)
        self._cleanup_sim()


class BiRRT(RRT):

    def __init__(self, args, start_conf, goal_conf):
        super().__init__(args, start_conf, goal_conf)
        self.roadmaps_start = [nx.Graph() for _ in range(self.num_envs)]
        self.roadmaps_goal = [nx.Graph() for _ in range(self.num_envs)]

    # mehod override
    def plan(
            self,
            ext_dist=2,
            coll_th=0.5,
            rand_rate=100,
            annealing_rate=None,
            min_rand_rate=10,
            max_iter=300,
            max_time=15.0,
            smoothing_iterations=10,  # 50
            visualize=True):

        if self._is_invalid_input(
                self.start_conf,
                self.goal_conf,
                ext_dist,
                coll_th):
            return None

        for rm, rm_s, rm_g in zip(self.roadmaps, self.roadmaps_start, self.roadmaps_goal):
            rm.clear()
            rm_s.clear()
            rm_g.clear()
            rm_s.add_node('start', conf=self.start_conf)
            rm_g.add_node('goal', conf=self.goal_conf)

        tree_a_goal_conf = [rm_g.nodes['goal']['conf'] for rm_g in self.roadmaps_goal]
        tree_b_goal_conf = [rm_s.nodes['start']['conf'] for rm_s in self.roadmaps_start]
        tree_a = self.roadmaps_start
        tree_b = self.roadmaps_goal

        iter = 0
        on_env_ids = [ei for ei in range(self.num_envs)]
        on_env_ids_copy = on_env_ids.copy()
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rates = [rand_rate - int(self.roadmaps[ei].number_of_nodes() * annealing_rate) for ei in on_env_ids]
                rand_rates = [rr if rr > min_rand_rate else min_rand_rate for rr in rand_rates]
            else:
                rand_rates = [rand_rate for _ in range(len(on_env_ids))]
            rand_conf = self._sample_conf(
                rand_rates=rand_rates,
                default_conf=self.goal_conf,
                n_envs=len(on_env_ids))

            for i, ei in enumerate(on_env_ids):
                last_nid = self._extend_roadmap(
                    roadmap=tree_a[ei],
                    conf=rand_conf[i],
                    ext_dist=ext_dist,
                    coll_th=coll_th,
                    goal_conf=tree_a_goal_conf[ei],
                    env_i=ei,
                    exact_end=False)
                if last_nid != -1:  # not trapped:
                    goal_nid = last_nid
                    tree_b_goal_conf[ei] = tree_a[ei].nodes[goal_nid]['conf']
                    last_nid = self._extend_roadmap(
                        roadmap=tree_b[ei],
                        conf=tree_a[ei].nodes[last_nid]['conf'],
                        ext_dist=ext_dist,
                        coll_th=coll_th,
                        goal_conf=tree_b_goal_conf[ei],
                        env_i=ei,
                        exact_end=False)
                    if last_nid == 'connection':
                        self.roadmaps[ei] = nx.compose(tree_a[ei], tree_b[ei])
                        self.roadmaps[ei].add_edge(last_nid, goal_nid)
                        on_env_ids.remove(ei)
                    elif last_nid != -1:
                        goal_nid = last_nid
                        tree_a_goal_conf[ei] = tree_b[ei].nodes[goal_nid]['conf']
                if tree_a[ei].number_of_nodes() > tree_b[ei].number_of_nodes():
                    tree_a[ei], tree_b[ei] = tree_b[ei], tree_a[ei]
                    tree_a_goal_conf[ei], tree_b_goal_conf[ei] = tree_b_goal_conf[ei], tree_a_goal_conf[ei]

            if len(on_env_ids) == 0:
                break

            iter += 1
            if iter > max_iter:
                print("Reach to maximum iteration! Failed to find a path.")
                return None

        print("\nParameter setting:")
        print("\tExtention distance =", ext_dist)
        print("\tRandom generation rate =", rand_rate)
        print("\tMaximum iteration =", max_iter)
        print("\tMaximum time =", max_time)
        print("\tSmoothing iterations =", smoothing_iterations)
        print("\tCollision threshold =", coll_th)

        path_confs = [self._path_from_roadmap(rm) for rm in self.roadmaps]
        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = [
            self._smooth_path(
                path=path_conf,
                env_i=i,
                coll_th=coll_th,
                granularity=ext_dist,
                iterations=smoothing_iterations)
            for i, path_conf in enumerate(path_confs)]
        self.path_confs = smoothed_path_confs

        mean_num_nodes = np.mean([rm.number_of_nodes() for ei, rm in enumerate(self.roadmaps)])
        mean_len_paths = np.mean([len(pc) for ei, pc in enumerate(path_confs)])
        mean_err_confs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(path_confs)])
        mean_len_spaths = np.mean([len(pc) for ei, pc in enumerate(self.path_confs)])
        mean_err_sconfs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(self.path_confs)])

        print("\nGenerated trajectory (mean over the environments):")
        print("\tNumber of nodes =", mean_num_nodes)
        print("\tLength of original paths =", mean_len_paths)
        print("\tError of original configurations =", mean_err_confs)
        print("\tLength of smoothed paths =", mean_len_spaths)
        print("\tError of smoothed configurations =", mean_err_sconfs)

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'gym_birrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path(on_env_ids_copy)
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


class RRTStar(RRT):

    def __init__(self, args, start_conf, goal_conf, nearby_ratio=2):
        super().__init__(args, start_conf, goal_conf)
        self.roadmaps = [nx.DiGraph() for _ in range(self.num_envs)]
        self.nearby_ratio = nearby_ratio

    def _get_nearby_nid_with_min_cost(self, roadmap, new_conf, ext_dist):
        nodes_conf_dict = dict(roadmap.nodes(data='conf'))
        nodes_conf_key_list = list(nodes_conf_dict.keys())
        nodes_conf_value_list = list(nodes_conf_dict.values())
        conf_array = np.array(nodes_conf_value_list)
        diff_conf_array = np.linalg.norm(conf_array - new_conf, axis=1)
        # warninng: assumes no collision
        candidate_mask = diff_conf_array < ext_dist * self.nearby_ratio
        nodes_conf_key_array = np.array(nodes_conf_key_list, dtype=object)
        nearby_nid_list = list(nodes_conf_key_array[candidate_mask])
        return nearby_nid_list

    def _extend_one_conf(self, conf1, conf2, ext_dist):
        norm, vec = self._unit_vector(conf2 - conf1, toggle_length=True)
        return conf1 + ext_dist * vec if norm > 1e-6 else None

    def _extend_roadmap(
            self,
            roadmap,
            conf,
            ext_dist,
            coll_th,
            goal_conf,
            env_i):
        """
            Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf = self._extend_one_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        if new_conf is not None:
            if self._is_collided(new_conf, coll_th, env_i):
                return -1
            else:
                new_nid = random.randint(0, 1e16)
                # find nearby_nid_list
                nearby_nid_list = self._get_nearby_nid_with_min_cost(
                    roadmap, new_conf, ext_dist)
                # costs
                nodes_cost_dict = dict(roadmap.nodes(data='cost'))
                nearby_cost_list = itemgetter(*nearby_nid_list)(nodes_cost_dict)
                if type(nearby_cost_list) == np.ndarray:
                    nearby_cost_list = [nearby_cost_list]
                nearby_min_cost_nid = \
                    nearby_nid_list[np.argmin(np.array(nearby_cost_list))]
                roadmap.add_node(new_nid, conf=new_conf, cost=0)  # add new nid
                roadmap.add_edge(nearby_min_cost_nid, new_nid)  # add new edge
                if roadmap.nodes[nearby_min_cost_nid].get('cost') is None:
                    roadmap.nodes[nearby_min_cost_nid]['cost'] = 0
                roadmap.nodes[new_nid]['cost'] = \
                    roadmap.nodes[nearby_min_cost_nid]['cost'] + 1  # update cost
                # rewire
                for nearby_nid in nearby_nid_list:
                    if nearby_nid != nearby_min_cost_nid:
                        if roadmap.nodes[new_nid]['cost'] + 1 < roadmap.nodes[nearby_nid]['cost']:
                            nearby_parent_nid = next(roadmap.predecessors(nearby_nid))
                            roadmap.remove_edge(nearby_parent_nid, nearby_nid)
                            roadmap.add_edge(new_nid, nearby_nid)
                            roadmap.nodes[nearby_nid]['cost'] = roadmap.nodes[new_nid]['cost'] + 1
                            cost_counter = 0
                            for nid in roadmap.successors(nearby_nid):
                                cost_counter += 1
                                roadmap.nodes[nid]['cost'] = roadmap.nodes[nearby_nid]['cost'] + cost_counter
                # check goal
                if self._goal_test(
                        conf=roadmap.nodes[new_nid]['conf'],
                        goal_conf=goal_conf,
                        threshold=ext_dist):
                    print("Goal reached in env", env_i)
                    roadmap.add_node('connection', conf=goal_conf)
                    if roadmap.nodes['connection'].get('cost') is None:
                        roadmap.nodes['connection']['cost'] = 0
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
                return nearby_min_cost_nid

    # mehod override
    def plan(
            self,
            ext_dist=2,
            coll_th=0.5,
            rand_rate=70,
            annealing_rate=None,
            min_rand_rate=10,
            max_iter=1000,
            max_time=15.0,
            smoothing_iterations=0, # 17
            visualize=True):

        if self._is_invalid_input(
                self.start_conf,
                self.goal_conf,
                ext_dist,
                coll_th):
            return None

        for roadmap in self.roadmaps:
            roadmap.clear()
            roadmap.add_node('start', conf=self.start_conf)
            roadmap.nodes['start']['cost'] = 0

        iter = 0
        on_env_ids = [ei for ei in range(self.num_envs)]
        on_env_ids_copy = on_env_ids.copy()
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rates = [rand_rate - int(self.roadmaps[ei].number_of_nodes() * annealing_rate) for ei in on_env_ids]
                rand_rates = [rr if rr > min_rand_rate else min_rand_rate for rr in rand_rates]
            else:
                rand_rates = [rand_rate for _ in range(len(on_env_ids))]
            rand_conf = self._sample_conf(
                rand_rates=rand_rates,
                default_conf=self.goal_conf,
                n_envs=len(on_env_ids))

            for i, ei in enumerate(on_env_ids):
                # extend_roadmap_while_checking_collsiion
                last_nid = self._extend_roadmap(
                    roadmap=self.roadmaps[ei],
                    conf=rand_conf[i],
                    ext_dist=ext_dist,
                    coll_th=coll_th,
                    goal_conf=self.goal_conf,
                    env_i=ei)
                if last_nid == 'connection':
                    mapping = {'connection': 'goal'}
                    self.roadmaps = [
                        nx.relabel_nodes(rm, mapping)
                        for rm in self.roadmaps]
                    on_env_ids.remove(ei)

            if len(on_env_ids) == 0:
                break

            iter += 1
            if iter > max_iter:
                print("Reach to maximum iteration! Failed to find a path.")
                return None

        print("\nParameter setting:")
        print("\tExtention distance =", ext_dist)
        print("\tRandom generation rate =", rand_rate)
        print("\tMaximum iteration =", max_iter)
        print("\tMaximum time =", max_time)
        print("\tSmoothing iterations =", smoothing_iterations)
        print("\tCollision threshold =", coll_th)

        path_confs = [self._path_from_roadmap(rm) for rm in self.roadmaps]
        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = [
            self._smooth_path(
                path=path_conf,
                env_i=i,
                coll_th=coll_th,
                granularity=ext_dist,
                iterations=smoothing_iterations)
            for i, path_conf in enumerate(path_confs)]
        self.path_confs = smoothed_path_confs

        mean_num_nodes = np.mean([rm.number_of_nodes() for ei, rm in enumerate(self.roadmaps)])
        mean_len_paths = np.mean([len(pc) for ei, pc in enumerate(path_confs)])
        mean_err_confs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(path_confs)])
        mean_len_spaths = np.mean([len(pc) for ei, pc in enumerate(self.path_confs)])
        mean_err_sconfs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(self.path_confs)])

        print("\nGenerated trajectory (mean over the environments):")
        print("\tNumber of nodes =", mean_num_nodes)
        print("\tLength of original paths =", mean_len_paths)
        print("\tError of original configurations =", mean_err_confs)
        print("\tLength of smoothed paths =", mean_len_spaths)
        print("\tError of smoothed configurations =", mean_err_sconfs)

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'gym_rrtstar.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path(on_env_ids_copy)
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


class BiRRTStar(RRTStar):

    def __init__(self, args, start_conf, goal_conf, nearby_ratio=2):
        super().__init__(args, start_conf, goal_conf)
        self.nearby_ratio = nearby_ratio

        self.roadmaps_start = [nx.Graph() for _ in range(self.num_envs)]
        self.roadmaps_goal = [nx.Graph() for _ in range(self.num_envs)]

    def _extend_roadmap(
            self,
            roadmap,
            conf,
            ext_dist,
            coll_th,
            goal_conf,
            env_i):
        """
            Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf = self._extend_one_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        if new_conf is not None:
            if self._is_collided(new_conf, coll_th, env_i):
                return -1
            else:
                new_nid = random.randint(0, 1e16)
                # find nearby_nid_list
                nearby_nid_list = self._get_nearby_nid_with_min_cost(
                    roadmap, new_conf, ext_dist)
                # costs
                nodes_cost_dict = dict(roadmap.nodes(data='cost'))
                nearby_cost_list = itemgetter(*nearby_nid_list)(nodes_cost_dict)
                if type(nearby_cost_list) == np.ndarray:
                    nearby_cost_list = [nearby_cost_list]
                nearby_min_cost_nid = \
                    nearby_nid_list[np.argmin(np.array(nearby_cost_list))]
                roadmap.add_node(new_nid, conf=new_conf, cost=0)  # add new nid
                roadmap.add_edge(nearby_min_cost_nid, new_nid)  # add new edge
                if roadmap.nodes[nearby_min_cost_nid].get('cost') is None:
                    roadmap.nodes[nearby_min_cost_nid]['cost'] = 0
                roadmap.nodes[new_nid]['cost'] = \
                    roadmap.nodes[nearby_min_cost_nid]['cost'] + 1  # update cost
                # rewire
                for nearby_nid in nearby_nid_list:
                    if nearby_nid != nearby_min_cost_nid:
                        if roadmap.nodes[nearby_min_cost_nid]['cost'] + 1 < roadmap.nodes[nearby_nid]['cost']:
                            nearby_e_nid = list(roadmap.neighbors(nearby_nid))[0]
                            roadmap.remove_edge(nearby_nid, nearby_e_nid)
                            roadmap.add_edge(nearby_nid, nearby_min_cost_nid)
                            roadmap.nodes[nearby_nid]['cost'] = roadmap.nodes[nearby_min_cost_nid]['cost'] + 1
                # check goal
                if self._goal_test(
                        conf=roadmap.nodes[new_nid]['conf'],
                        goal_conf=goal_conf,
                        threshold=ext_dist):
                    print("Goal reached in env", env_i)
                    roadmap.add_node('connection', conf=goal_conf)
                    if roadmap.nodes['connection'].get('cost') is None:
                        roadmap.nodes['connection']['cost'] = 0
                    roadmap.add_edge(new_nid, 'connection')
                    return 'connection'
                return new_nid
        return nearest_nid

    # mehod override
    def plan(
            self,
            ext_dist=2,
            coll_th=0.5,
            rand_rate=70,
            annealing_rate=None,
            min_rand_rate=10,
            max_iter=1000,
            max_time=15.0,
            smoothing_iterations=0,  # 17
            visualize=True):

        if self._is_invalid_input(
                self.start_conf,
                self.goal_conf,
                ext_dist,
                coll_th):
            return None

        for rm, rm_s, rm_g in zip(self.roadmaps, self.roadmaps_start, self.roadmaps_goal):
            rm.clear()
            rm_s.clear()
            rm_g.clear()
            rm_s.add_node('start', conf=self.start_conf)
            rm_g.add_node('goal', conf=self.goal_conf)
            rm.add_node('start', conf=self.start_conf)
            rm.nodes['start']['cost'] = 0

        tree_a_goal_conf = [rm_g.nodes['goal']['conf'] for rm_g in self.roadmaps_goal]
        tree_b_goal_conf = [rm_s.nodes['start']['conf'] for rm_s in self.roadmaps_start]
        tree_a = self.roadmaps_start
        tree_b = self.roadmaps_goal

        iter = 0
        on_env_ids = [ei for ei in range(self.num_envs)]
        on_env_ids_copy = on_env_ids.copy()
        tic = time.time()
        while not self.gym.query_viewer_has_closed(self.viewer):
            self._update_gym_sim(self.gym_wait_steps)

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # anealing random sampling
            if annealing_rate is not None:
                rand_rates = [rand_rate - int(self.roadmaps[ei].number_of_nodes() * annealing_rate) for ei in on_env_ids]
                rand_rates = [rr if rr > min_rand_rate else min_rand_rate for rr in rand_rates]
            else:
                rand_rates = [rand_rate for _ in range(len(on_env_ids))]
            rand_conf = self._sample_conf(
                rand_rates=rand_rates,
                default_conf=self.goal_conf,
                n_envs=len(on_env_ids))

            for i, ei in enumerate(on_env_ids):
                last_nid = self._extend_roadmap(
                    roadmap=tree_a[ei],
                    conf=rand_conf[i],
                    ext_dist=ext_dist,
                    coll_th=coll_th,
                    goal_conf=tree_a_goal_conf[ei],
                    env_i=ei)
                if last_nid != -1:  # not trapped:
                    goal_nid = last_nid
                    tree_b_goal_conf[ei] = tree_a[ei].nodes[goal_nid]['conf']
                    last_nid = self._extend_roadmap(
                        roadmap=tree_b[ei],
                        conf=tree_a[ei].nodes[last_nid]['conf'],
                        ext_dist=ext_dist,
                        coll_th=coll_th,
                        goal_conf=tree_b_goal_conf[ei],
                        env_i=ei)
                    if last_nid == 'connection':
                        self.roadmaps[ei] = nx.compose(tree_a[ei], tree_b[ei])
                        self.roadmaps[ei].add_edge(last_nid, goal_nid)
                        on_env_ids.remove(ei)
                    elif last_nid != -1:
                        goal_nid = last_nid
                        tree_a_goal_conf[ei] = tree_b[ei].nodes[goal_nid]['conf']
                if tree_a[ei].number_of_nodes() > tree_b[ei].number_of_nodes():
                    tree_a[ei], tree_b[ei] = tree_b[ei], tree_a[ei]
                    tree_a_goal_conf[ei], tree_b_goal_conf[ei] = tree_b_goal_conf[ei], tree_a_goal_conf[ei]

            if len(on_env_ids) == 0:
                break

            iter += 1
            if iter > max_iter:
                print("Reach to maximum iteration! Failed to find a path.")
                return None

        print("\nParameter setting:")
        print("\tExtention distance =", ext_dist)
        print("\tRandom generation rate =", rand_rate)
        print("\tMaximum iteration =", max_iter)
        print("\tMaximum time =", max_time)
        print("\tSmoothing iterations =", smoothing_iterations)
        print("\tCollision threshold =", coll_th)

        path_confs = [self._path_from_roadmap(rm) for rm in self.roadmaps]
        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = [
            self._smooth_path(
                path=path_conf,
                env_i=i,
                coll_th=coll_th,
                granularity=ext_dist,
                iterations=smoothing_iterations)
            for i, path_conf in enumerate(path_confs)]
        self.path_confs = smoothed_path_confs

        mean_num_nodes = np.mean([rm.number_of_nodes() for ei, rm in enumerate(self.roadmaps)])
        mean_len_paths = np.mean([len(pc) for ei, pc in enumerate(path_confs)])
        mean_err_confs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(path_confs)])
        mean_len_spaths = np.mean([len(pc) for ei, pc in enumerate(self.path_confs)])
        mean_err_sconfs = np.mean([sum([abs(pc[i+1] - pc[i]) for i in range(len(pc)-1)]) for ei, pc in enumerate(self.path_confs)])

        print("\nGenerated trajectory (mean over the environments):")
        print("\tNumber of nodes =", mean_num_nodes)
        print("\tLength of original paths =", mean_len_paths)
        print("\tError of original configurations =", mean_err_confs)
        print("\tLength of smoothed paths =", mean_len_spaths)
        print("\tError of smoothed configurations =", mean_err_sconfs)

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'gym_birrtstar.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path(on_env_ids_copy)
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


if __name__ == "__main__":

    custom_parameters = [
        {"name": '--num_envs',
         "type": int,
         "default": 1,
         "help": "Number of environments to create"},
        {"name": '--sdf',
         "action": 'store_true',
         "help": "Use SDF-based collision check"},
        {"name": '--ctrl',
         "type": str,
         "choices": ['none', 'pos'],
         "default": 'none',
         "help": "Select a controller from ['none', 'pos']"},
        {"name": '--alg',
         "type": str,
         "choices": ['naiveRRT', 'RRT', 'BiRRT', 'RRTStar', 'BiRRTStar', 'naiveRRT'],
         "default": 'RRT',
         "help": "Select a planning algorithm from ['naiveRRT', 'RRT', 'BiRRT', 'RRTStar', 'BiRRTStar', 'naiveRRT']"}]

    args = gymutil.parse_arguments(
        description="Assembly System",
        custom_parameters=custom_parameters)

    # start and goal position and configuration
    start_conf = np.array([-1.6, -1.3, 0., -1.0, 0., 1.2, -1.4])
    goal_conf = np.array([1.6, -1.4, 0., -1.4, 0., 0.8, -1.4])

    if args.alg == 'naiveRRT':
        rrt = naiveRRT(args, start_conf, goal_conf)
        ext_dist = 0.1
        coll_th = 0.1
    elif args.alg == 'RRT':
        rrt = RRT(args, start_conf, goal_conf)
        ext_dist = 0.1
        coll_th = 0.1
    elif args.alg == 'BiRRT':
        rrt = BiRRT(args, start_conf, goal_conf)
        ext_dist = 0.1
        coll_th = 0.1
    elif args.alg == 'RRTStar':
        rrt = RRTStar(args, start_conf, goal_conf)
        ext_dist = 0.1
        coll_th = 0.1
    elif args.alg == 'BiRRTStar':
        rrt = BiRRTStar(args, start_conf, goal_conf)
        ext_dist = 0.1
        coll_th = 0.1
    else:
        print("Error: The algorithm", args.alg, "is not implemented.")
        exit(-1)

    path_confs = rrt.plan(
        ext_dist=ext_dist,
        coll_th=coll_th,
        rand_rate=50,
        annealing_rate=0.01,  # the percentage of decrease of random rate per increase of one node
        max_time=10000,
        visualize=True)
