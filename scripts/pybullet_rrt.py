#!/usr/bin/env python3

import os
import time
import copy
import math
import pickle
import argparse
import quaternion
import numpy as np
import networkx as nx
from operator import itemgetter

import pybullet as p
import pybullet_data

from pybullet_planning.pybullet_tools.utils import *


class RRT(object):

    def __init__(self, args, start_conf, goal_conf):

        self.start_conf = start_conf
        self.goal_conf = goal_conf
        self.roadmap = nx.Graph()

        self._initialize_sim()
        self._asset_root = "/dataset"
        self._load_obstacles()
        self._load_iiwa()
        self._prepare_start_planning()

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

    def _cleanup_sim(self):
        p.disconnect()

    def _initialize_sim(self):
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

    def _load_obstacles(self):
        plane = p.loadURDF("plane.urdf")

        obstacle1 = p.loadURDF(
            os.path.join(
                self._asset_root,
                'urdfs',
                'block.urdf'),
            basePosition=[-0.2, -0.05, 0.8],
            useFixedBase=True)
        p.stepSimulation()
        obstacle2 = p.loadURDF(
            os.path.join(
                self._asset_root,
                'urdfs',
                'block.urdf'),
            basePosition=[0., -0.1, 0.8],
            useFixedBase=True)
        p.stepSimulation()
        obstacle3 = p.loadURDF(
            os.path.join(
                self._asset_root,
                'urdfs',
                'block.urdf'),
            basePosition=[0.2, -0.05, 0.9],
            useFixedBase=True)
        p.stepSimulation()
        self.obstacles = [plane, obstacle1, obstacle2, obstacle3]

    def _set_joint_positions(self, body, joints, values):
        assert len(joints) == len(values)
        for joint, value in zip(joints, values):
            p.resetJointState(body, joint, value)

    def _load_iiwa(self):
        iiwa_urdfname = 'iiwa14_rq140.urdf'
        self.iiwa = p.loadURDF(
            os.path.join(
                self._asset_root,
                'urdfs',
                iiwa_urdfname),
            basePosition=[0, 0, 0.02],
            useFixedBase=True)
        p.stepSimulation()

        # extract link information
        self._link_name_to_index = {p.getBodyInfo(self.iiwa)[0].decode('UTF-8'):-1,}        
        for _id in range(p.getNumJoints(self.iiwa)):
            _name = p.getJointInfo(self.iiwa, _id)[12].decode('UTF-8')
            self._link_name_to_index[_name] = _id

        self.iiwa_joint_indices = [
            self._link_name_to_index['iiwa_link_1'],
            self._link_name_to_index['iiwa_link_2'],
            self._link_name_to_index['iiwa_link_3'],
            self._link_name_to_index['iiwa_link_4'],
            self._link_name_to_index['iiwa_link_5'],
            self._link_name_to_index['iiwa_link_6'],
            self._link_name_to_index['iiwa_link_7']]

        self.jlims = [get_joint_limits(self.iiwa, j) for j in self.iiwa_joint_indices]
        # 1: (-4.71238898038469, 4.71238898038469)
        # 2: (-2.007128639793479, 2.007128639793479)
        # 3: (-4.71238898038469, 4.71238898038469)
        # 4: (-0.03490658503988659, 2.652900463031381)
        # 5: (-4.71238898038469, 4.71238898038469)
        # 6: (-2.181661564992912, 2.181661564992912)
        # 7: (-3.141592653589793, 3.141592653589793)

        # _link_name_to_index
        # {'world': -1, 'iiwa_link_0': 0, 'iiwa_link_1': 1, 'iiwa_link_2': 2, 'iiwa_link_3': 3, 'iiwa_link_4': 4, 'iiwa_link_5': 5, 'iiwa_link_6': 6, 'iiwa_link_7': 7, 'iiwa_link_ee': 8, 'robotiq_arg2f_base_link': 9, 'left_outer_knuckle': 10, 'left_outer_finger': 11, 'left_inner_finger': 12, 'left_finger_tip': 13, 'left_inner_knuckle': 14, 'right_outer_knuckle': 15, 'right_outer_finger': 16, 'right_inner_finger': 17, 'right_finger_tip': 18, 'right_inner_knuckle': 19, 'tool': 20, 'base_link': 21}

        # get the collision checking function
        disa_col_link_pairs = set([
            (-1, 0),
            (-1, 21),
            (21, 0),
            (-1, 1),
            (0, 1),
            (0, 2),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (5, 7),
            (5, 8),
            (6, 7),
            (6, 8),
            (7, 8),
            (7, 9),
            (8, 9),
            (8, 20),
            (9, 20),
            (9, 10),
            (9, 11),
            (9, 12),
            (9, 13),
            (9, 14),
            (9, 15),
            (9, 16),
            (9, 17),
            (9, 18),
            (9, 19),
            (10, 11),
            (10, 12),
            (10, 13),
            (10, 14),
            (11, 12),
            (11, 13),
            (11, 14),
            (12, 13),
            (12, 14),
            (13, 14),
            (15, 16),
            (15, 17),
            (15, 18),
            (15, 19),
            (16, 17),
            (16, 18),
            (16, 19),
            (17, 18),
            (17, 19),
            (18, 19)])

        self.collision_fn = get_collision_fn(
            self.iiwa,
            self.iiwa_joint_indices,
            obstacles=self.obstacles,
            attachments=[],
            self_collisions=False,
            disabled_collisions=disa_col_link_pairs)

    def _is_collided(self, conf):
        if not violates_limits(self.iiwa, self.iiwa_joint_indices, conf):
            return self.collision_fn(conf)
        else:
            return True

    def _sample_conf(self, rand_rate, default_conf):
        flag = random.randint(0, 99) < rand_rate
        return np.array(tuple(
            [np.random.uniform(low=self.jlims[i][0], high=self.jlims[i][1]) \
             for i in range(len(default_conf))])) if flag else default_conf

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
            goal_conf,
            exact_end=True):
        """ Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf_list = self._extend_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist, exact_end=exact_end)[1:]
        for new_conf in new_conf_list:
            if self._is_collided(new_conf):
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
                    print("Goal reached")
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
            if (len(shortcut) <= (j - i) + 1) and all(not self._is_collided(conf=conf) for conf in shortcut):
                smoothed_path = smoothed_path[:i] + shortcut + smoothed_path[j + 1:]
        return smoothed_path

    def _draw_sphere_marker(self, position, radius, color):
        vs_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color)
        marker_id = p.createMultiBody(
            basePosition=position,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id)
        return marker_id

    def _update_bullet_sim_sec(self, wait_sec):
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()
            toc = time.time()
            if wait_sec > 0.0:
                if toc - tic > wait_sec:
                    break

    def _prepare_start_planning(self):
        set_joint_positions(
            self.iiwa, self.iiwa_joint_indices, self.start_conf)
        p.stepSimulation()
        self.start_position = p.getLinkState(
            self.iiwa,
            self._link_name_to_index['iiwa_link_ee'],
            computeForwardKinematics=True)[0]
        start_marker = self._draw_sphere_marker(
            position=self.start_position, radius=0.01, color=[0, 1, 0, 1])
        time.sleep(3.0)

        set_joint_positions(
            self.iiwa, self.iiwa_joint_indices, self.goal_conf)
        p.stepSimulation()
        self.goal_position = p.getLinkState(
            self.iiwa,
            self._link_name_to_index['iiwa_link_ee'],
            computeForwardKinematics=True)[0]
        goal_marker = self._draw_sphere_marker(
            position=self.goal_position, radius=0.01, color=[1, 0, 0, 1])
        time.sleep(3.0)

    def _is_invalid_input(self, start_conf, goal_conf, ext_dist):
        # check joint values and goal conf
        if self._is_collided(start_conf):
            print("The start robot configuration is in collision!")
            return None
        if self._is_collided(goal_conf):
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
                ext_dist):
            return None

        self.roadmap.clear()
        self.roadmap.add_node('start', conf=self.start_conf)

        iter = 0
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rate_on = rand_rate - int(self.roadmap.number_of_nodes() * annealing_rate)
                rand_rate_on = rand_rate_on if rand_rate_on > min_rand_rate else min_rand_rate
            rand_conf = self._sample_conf(
                rand_rate=rand_rate_on,
                default_conf=self.goal_conf)

            # extend roadmap while checking collsiion
            last_nid = self._extend_roadmap(
                roadmap=self.roadmap,
                conf=rand_conf,
                ext_dist=ext_dist,
                goal_conf=self.goal_conf)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
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

        path_confs = self._path_from_roadmap(self.roadmap)
        if len(path_confs) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = self._smooth_path(
            path=path_confs,
            granularity=ext_dist,
            iterations=smoothing_iterations)
        self.path_confs = smoothed_path_confs

        num_nodes = self.roadmap.number_of_nodes()
        len_paths = len(path_confs)
        err_confs = sum([abs(path_confs[i+1] - path_confs[i]) for i in range(len(path_confs)-1)])
        len_spaths = len(self.path_confs)
        err_sconfs = sum([abs(self.path_confs[i+1] - self.path_confs[i]) for i in range(len(self.path_confs)-1)])

        print("\nGenerated trajectory:")
        print("\tNumber of nodes =", num_nodes)
        print("\tLength of original paths =", len_paths)
        print("\tError of original configurations =", [float('{:.2f}'.format(err)) for err in err_confs])
        print("\tLength of smoothed paths =", len_spaths)
        print("\tError of smoothed configurations =", [float('{:.2f}'.format(err)) for err in err_sconfs])

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'bullet_rrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path()
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs

    def _replay_planned_path(self):
        while True:
            for q in self.path_confs:
                self._set_joint_positions(
                    self.iiwa, self.iiwa_joint_indices, q)
                p.stepSimulation()
                eefp = p.getLinkState(
                    self.iiwa, 
                    self._link_name_to_index['iiwa_link_ee'],
                    computeForwardKinematics=True)[0]
                set_point(create_capsule(0.005, 0.005, color=BLUE), [eefp[0], eefp[1], eefp[2]])
                p.stepSimulation()
                time.sleep(0.05)

    def visualize_path(self, conf_keyname='conf'):

        ep_nears = {
            'start': self.start_position,
            'goal': self.goal_position}
        eg_i = 0
        edge = list(self.roadmap.edges)
        leneg = len(edge)
        ep_near = copy.deepcopy(self.start_position)

        while p.isConnected():
            self._set_joint_positions(
                self.iiwa,
                self.iiwa_joint_indices,
                self.roadmap.nodes[edge[eg_i][1]][conf_keyname])
            p.stepSimulation()
            ep_st = p.getLinkState(
                self.iiwa,
                self._link_name_to_index['iiwa_link_ee'],
                computeForwardKinematics=True)
            ep_new = ep_st[0]

            w2e_rotmat = quaternion.as_rotation_matrix(
                np.quaternion(ep_st[1][3], ep_st[1][0], ep_st[1][1], ep_st[1][2]))
            eelink_x_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0.02, 0, 0])) 
            eelink_y_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0, 0.02, 0]))
            eelink_z_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0, 0, 0.02]))
            ep_near = copy.deepcopy(ep_nears[edge[eg_i][0]])
            # visualize the tree nodes
            p.addUserDebugLine(
                ep_near,
                ep_new,
                lineColorRGB=[0, 0, 0],
                lineWidth=1,
                lifeTime=0)
            # visualize the x axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_x_vec,
                lineColorRGB=[1, 0, 0],
                lineWidth=2,
                lifeTime=3)
            # visualize the y axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_y_vec,
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=3)
            # visualize the z axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_z_vec,
                lineColorRGB=[0, 0, 1],
                lineWidth=2,
                lifeTime=3)
            p.stepSimulation()
            time.sleep(0.01)
            ep_nears[edge[eg_i][1]] = copy.deepcopy(ep_new)

            eg_i += 1
            if eg_i >= leneg:
                break

        self._replay_planned_path()
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

    def _find_path(self, s, d, vertex, adj_list, visited, curr_path):
        if len(self.final_path) > 0:
            return 
        idx = vertex.index(s)
        visited[idx] = 1
        curr_path.append(s)
        if d == s:
            self.final_path = copy.deepcopy(curr_path)
            return
        else:
            for x in range(len(vertex)):
                if adj_list[idx,x] == 1 and visited[x] == 0:
                    self._find_path(
                        vertex[x], d, vertex, adj_list, visited, curr_path)
        curr_path.pop()
        visited[idx] = 0

    # method override
    def plan(
            self,
            ext_dist=0.1,
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
                ext_dist):
            return None

        self.final_path = []
        self.path = [tuple(self.start_conf)]
        self.edges = []
        path_confs = []

        iter = 0
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # a compute random configuration (annealing random sampling)
            if annealing_rate is not None:
                rand_rate_on = rand_rate - int(len(self.path) * annealing_rate)
                rand_rate_on = rand_rate_on if rand_rate_on > min_rand_rate else min_rand_rate
            qrand = self._sample_conf(
                rand_rate=rand_rate_on,
                default_conf=self.goal_conf)

            # b find nearest vertex to qrand that is already in g
            qnear = self._find_nn(self.path, qrand)
            if np.linalg.norm(qnear - qrand) == 0.0:
                continue
            # c move Dq from qnear to qrand
            qnew = self._move_node(qnear, qrand, ext_dist)

            # e if edge is collision-free
            if not self._is_collided(qnew):
                self.path.append(tuple(qnew.reshape(1, -1)[0]))
                self.edges.append((
                    tuple(qnear.reshape(1, -1)[0]),
                    tuple(qnew.reshape(1, -1)[0])))
                # d judge that it reach the goal (connect goal)
                if self._goal_test(qnew, qgoal, ext_dist):
                    print("Goal reached")
                    adj_mat = self._build_adjmat(self.edges, self.path)
                    visited = np.zeros(len(self.path))
                    curr_path = []
                    self._find_path(
                        tuple(self.start_conf),
                        tuple(qnew.reshape(1, -1)[0]),
                        self.path,
                        adj_mat,
                        visited,
                        curr_path)
                    path_confs = self.final_path
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

        if sum([len(pcs) for pcs in path_confs]) == 0:
            print("The path between start and goal was not found.")
            return None

        path_confs = [np.array(path_conf) for path_conf in path_confs]
        self.path_confs = path_confs
        smoothed_path_confs = self._smooth_path(
            path=path_confs,
            granularity=ext_dist,
            iterations=smoothing_iterations)
        self.path_confs = smoothed_path_confs

        num_nodes = len(self.path)
        len_paths = len(path_confs)
        err_confs = sum([abs(path_confs[i+1] - path_confs[i]) for i in range(len(path_confs)-1)])
        len_spaths = len(self.path_confs)
        err_sconfs = sum([abs(self.path_confs[i+1] - self.path_confs[i]) for i in range(len(self.path_confs)-1)])

        print("\nGenerated trajectory:")
        print("\tNumber of nodes =", num_nodes)
        print("\tLength of original paths =", len_paths)
        print("\tError of original configurations =", [float('{:.2f}'.format(err)) for err in err_confs])
        print("\tLength of smoothed paths =", len_spaths)
        print("\tError of smoothed configurations =", [float('{:.2f}'.format(err)) for err in err_sconfs])

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'bullet_naiverrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path()
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs

    # method override
    def visualize_path(self):

        ep_nears = {0: self.start_position}
        for i, eg in enumerate(self.edges):
            self._set_joint_positions(self.iiwa, self.iiwa_joint_indices, eg[1])
            p.stepSimulation()
            ep_st = p.getLinkState(
                self.iiwa,
                self._link_name_to_index['iiwa_link_ee'],
                computeForwardKinematics=True)
            ep_new = ep_st[0]
            w2e_rotmat = quaternion.as_rotation_matrix(
                np.quaternion(ep_st[1][3], ep_st[1][0], ep_st[1][1], ep_st[1][2]))
            eelink_x_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0.02, 0, 0])) 
            eelink_y_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0, 0.02, 0]))
            eelink_z_vec = ep_new + np.dot(
                np.transpose(w2e_rotmat), np.array([0, 0, 0.02]))
            if i == 0:
                ep_near = copy.deepcopy(self.start_position)
            else:
                ep_near = copy.deepcopy(ep_nears[self.path.index(eg[0])])
            # visualize the tree nodes
            p.addUserDebugLine(
                ep_near,
                ep_new,
                lineColorRGB=[0, 0, 0],
                lineWidth=1,
                lifeTime=0)
            # visualize the x axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_x_vec,
                lineColorRGB=[1, 0, 0],
                lineWidth=2,
                lifeTime=3)
            # visualize the y axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_y_vec,
                lineColorRGB=[0, 1, 0],
                lineWidth=2,
                lifeTime=3)
            # visualize the z axis at the eef coordinate
            p.addUserDebugLine(
                ep_new,
                eelink_z_vec,
                lineColorRGB=[0, 0, 1],
                lineWidth=2,
                lifeTime=3)
            p.stepSimulation()
            time.sleep(0.01)
            ep_nears[self.path.index(eg[1])] = copy.deepcopy(ep_new)

        self._replay_planned_path()
        self._cleanup_sim()


class BiRRT(RRT):

    def __init__(self, args, start_conf, goal_conf):
        super().__init__(args, start_conf, goal_conf)
        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

    # mehod override
    def plan(
            self,
            ext_dist=2,
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
                ext_dist):
            return None
    
        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.roadmap_start.add_node('start', conf=self.start_conf)
        self.roadmap_goal.add_node('goal', conf=self.goal_conf)

        tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
        tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal

        iter = 0
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rate_on = rand_rate - int(self.roadmap.number_of_nodes() * annealing_rate)
                rand_rate_on = rand_rate_on if rand_rate_on > min_rand_rate else min_rand_rate
            rand_conf = self._sample_conf(
                rand_rate=rand_rate_on,
                default_conf=self.goal_conf)

            last_nid = self._extend_roadmap(
                roadmap=tree_a,
                conf=rand_conf,
                ext_dist=ext_dist,
                goal_conf=tree_a_goal_conf,
                exact_end=False)
            if last_nid != -1:  # not trapped:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
                last_nid = self._extend_roadmap(
                    roadmap=tree_b,
                    conf=tree_a.nodes[last_nid]['conf'],
                    ext_dist=ext_dist,
                    goal_conf=tree_b_goal_conf,
                    exact_end=False)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf

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

        path_confs = self._path_from_roadmap(self.roadmap)
        if len(path_confs) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = self._smooth_path(
            path=path_confs,
            granularity=ext_dist,
            iterations=smoothing_iterations)
        self.path_confs = smoothed_path_confs

        num_nodes = self.roadmap.number_of_nodes()
        len_paths = len(path_confs)
        err_confs = sum([abs(path_confs[i+1] - path_confs[i]) for i in range(len(path_confs)-1)])
        len_spaths = len(self.path_confs)
        err_sconfs = sum([abs(self.path_confs[i+1] - self.path_confs[i]) for i in range(len(self.path_confs)-1)])

        print("\nGenerated trajectory:")
        print("\tNumber of nodes =", num_nodes)
        print("\tLength of original paths =", len_paths)
        print("\tError of original configurations =", [float('{:.2f}'.format(err)) for err in err_confs])
        print("\tLength of smoothed paths =", len_spaths)
        print("\tError of smoothed configurations =", [float('{:.2f}'.format(err)) for err in err_sconfs])

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'bullet_birrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path()
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


class RRTStar(RRT):

    def __init__(self, args, start_conf, goal_conf, nearby_ratio=2):
        super().__init__(args, start_conf, goal_conf)
        self.roadmaps = nx.DiGraph()
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
            goal_conf):
        """
            Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf = self._extend_one_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        if new_conf is not None:
            if self._is_collided(new_conf):
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
                    print("Goal reached",)
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
                ext_dist):
            return None

        self.roadmap.clear()
        self.roadmap.add_node('start', conf=self.start_conf)
        self.roadmap.nodes['start']['cost'] = 0

        iter = 0
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # annealing random sampling
            if annealing_rate is not None:
                rand_rate_on = rand_rate - int(self.roadmap.number_of_nodes() * annealing_rate)
                rand_rate_on = rand_rate_on if rand_rate > min_rand_rate else min_rand_rate
            rand_conf = self._sample_conf(
                rand_rate=rand_rate_on,
                default_conf=self.goal_conf)

            # extend_roadmap_while_checking_collsiion
            last_nid = self._extend_roadmap(
                roadmap=self.roadmap,
                conf=rand_conf,
                ext_dist=ext_dist,
                goal_conf=self.goal_conf)
            if last_nid == 'connection':
                mapping = {'connection': 'goal'}
                self.roadmap = nx.relabel_nodes(self.roadmap, mapping)
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

        path_confs = self._path_from_roadmap(self.roadmap)
        if len(path_confs) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = self._smooth_path(
            path=path_confs,
            granularity=ext_dist,
            iterations=smoothing_iterations)
        self.path_confs = smoothed_path_confs

        num_nodes = self.roadmap.number_of_nodes()
        len_paths = len(path_confs)
        err_confs = sum([abs(path_confs[i+1] - path_confs[i]) for i in range(len(path_confs)-1)])
        len_spaths = len(self.path_confs)
        err_sconfs = sum([abs(self.path_confs[i+1] - self.path_confs[i]) for i in range(len(self.path_confs)-1)])

        print("\nGenerated trajectory:")
        print("\tNumber of nodes =", num_nodes)
        print("\tLength of original paths =", len_paths)
        print("\tError of original configurations =", [float('{:.2f}'.format(err)) for err in err_confs])
        print("\tLength of smoothed paths =", len_spaths)
        print("\tError of smoothed configurations =", [float('{:.2f}'.format(err)) for err in err_sconfs])

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'bullet_rrtstar.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path()
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


class BiRRTStar(RRTStar):

    def __init__(self, args, start_conf, goal_conf, nearby_ratio=2):
        super().__init__(args, start_conf, goal_conf)
        self.nearby_ratio = nearby_ratio

        self.roadmap_start = nx.Graph()
        self.roadmap_goal = nx.Graph()

    def _extend_roadmap(
            self,
            roadmap,
            conf,
            ext_dist,
            goal_conf):
        """
            Finds the nearest point between the given roadmap and the configuration.
            Then, this extends towards the configuration.
        """

        nearest_nid = self._get_nearest_nid(roadmap, conf)
        new_conf = self._extend_one_conf(
            roadmap.nodes[nearest_nid]['conf'], conf, ext_dist)
        if new_conf is not None:
            if self._is_collided(new_conf):
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
                    print("Goal reached")
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
                ext_dist):
            return None

        self.roadmap.clear()
        self.roadmap_start.clear()
        self.roadmap_goal.clear()
        self.roadmap_start.add_node('start', conf=self.start_conf)
        self.roadmap_goal.add_node('goal', conf=self.goal_conf)
        self.roadmap.add_node('start', conf=self.start_conf)
        self.roadmap.nodes['start']['cost'] = 0

        tree_a_goal_conf = self.roadmap_goal.nodes['goal']['conf']
        tree_b_goal_conf = self.roadmap_start.nodes['start']['conf']
        tree_a = self.roadmap_start
        tree_b = self.roadmap_goal

        iter = 0
        tic = time.time()
        while p.isConnected():
            p.stepSimulation()

            toc = time.time()
            if max_time > 0.0:
                if toc - tic > max_time:
                    print("Too much motion time! Failed to find a path.")
                    return None

            # anealing random sampling
            if annealing_rate is not None:
                rand_rate_on = rand_rate - int(self.roadmap.number_of_nodes() * annealing_rate)
                rand_rate_on = rand_rate_on if rand_rate_on > min_rand_rate else min_rand_rate
            rand_conf = self._sample_conf(
                rand_rate=rand_rate_on,
                default_conf=self.goal_conf)

            last_nid = self._extend_roadmap(
                roadmap=tree_a,
                conf=rand_conf,
                ext_dist=ext_dist,
                goal_conf=tree_a_goal_conf)
            if last_nid != -1:  # not trapped:
                goal_nid = last_nid
                tree_b_goal_conf = tree_a.nodes[goal_nid]['conf']
                last_nid = self._extend_roadmap(
                    roadmap=tree_b,
                    conf=tree_a.nodes[last_nid]['conf'],
                    ext_dist=ext_dist,
                    goal_conf=tree_b_goal_conf)
                if last_nid == 'connection':
                    self.roadmap = nx.compose(tree_a, tree_b)
                    self.roadmap.add_edge(last_nid, goal_nid)
                    break
                elif last_nid != -1:
                    goal_nid = last_nid
                    tree_a_goal_conf = tree_b.nodes[goal_nid]['conf']
            if tree_a.number_of_nodes() > tree_b.number_of_nodes():
                tree_a, tree_b = tree_b, tree_a
                tree_a_goal_conf, tree_b_goal_conf = tree_b_goal_conf, tree_a_goal_conf

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

        path_confs = self._path_from_roadmap(self.roadmap)
        if len(path_confs) == 0:
            print("The path between start and goal was not found.")
            return None

        self.path_confs = path_confs
        smoothed_path_confs = self._smooth_path(
            path=path_confs,
            granularity=ext_dist,
            iterations=smoothing_iterations)
        self.path_confs = smoothed_path_confs

        num_nodes = self.roadmap.number_of_nodes()
        len_paths = len(path_confs)
        err_confs = sum([abs(path_confs[i+1] - path_confs[i]) for i in range(len(path_confs)-1)])
        len_spaths = len(self.path_confs)
        err_sconfs = sum([abs(self.path_confs[i+1] - self.path_confs[i]) for i in range(len(self.path_confs)-1)])

        print("\nGenerated trajectory:")
        print("\tNumber of nodes =", num_nodes)
        print("\tLength of original paths =", len_paths)
        print("\tError of original configurations =", [float('{:.2f}'.format(err)) for err in err_confs])
        print("\tLength of smoothed paths =", len_spaths)
        print("\tError of smoothed configurations =", [float('{:.2f}'.format(err)) for err in err_sconfs])

        # save the trajectory datasets
        save_dirpath = '/dataset/pickle'
        os.makedirs(save_dirpath, exist_ok=True)
        save_pklpath = os.path.join(
            save_dirpath,
            'bullet_birrt.pkl')
        with open(save_pklpath, 'wb') as f:
            pickle.dump(self.path_confs, f, protocol=4)

        # visualize the progress of planning and generated trajectory
        if visualize:
            self.visualize_path()
            return self.path_confs
        else:
            self._cleanup_sim()
            return self.path_confs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--alg',
        type=str,
        choices=['naiveRRT', 'RRT', 'BiRRT', 'RRTStar', 'BiRRTStar'],
        default='naiveRRT',
        help="Select a planning algorithm from ['naiveRRT', 'RRT', 'BiRRT', 'RRTStar', 'BiRRTStar']")
    args = parser.parse_args()

    # start and goal position and configuration
    start_conf = np.array([-1.6, -1.3, 0., -1.0, 0., 1.2, -1.4])
    goal_conf = np.array([1.6, -1.4, 0., -1.4, 0., 0.8, -1.4])

    if args.alg == 'naiveRRT':
        rrt = naiveRRT(args, start_conf, goal_conf)
        ext_dist = 0.1
    elif args.alg == 'RRT':
        rrt = RRT(args, start_conf, goal_conf)
        ext_dist = 0.1
    elif args.alg == 'BiRRT':
        rrt = BiRRT(args, start_conf, goal_conf)
        ext_dist = 0.1
    elif args.alg == 'RRTStar':
        rrt = RRTStar(args, start_conf, goal_conf)
        ext_dist = 0.1
    elif args.alg == 'BiRRTStar':
        rrt = BiRRTStar(args, start_conf, goal_conf)
        ext_dist = 0.1
    else:
        print("Error: The algorithm", args.alg, "is not implemented.")
        exit(-1)

    path_confs = rrt.plan(
        ext_dist=ext_dist,
        rand_rate=50,
        annealing_rate=0.01,  # the percentage of decrease of random rate per increase of one node
        max_time=10000,
        visualize=True)
