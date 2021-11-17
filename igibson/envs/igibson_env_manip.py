import argparse
import logging
import time
from collections import OrderedDict

import gym
# from gym.utils import EzPickle
import numpy as np
import pybullet as p
from transforms3d.euler import euler2quat

from igibson.envs.env_base import BaseEnv
from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from igibson.robots.robot_base import BaseRobot
from igibson.sensors.bump_sensor import BumpSensor
from igibson.sensors.scan_sensor import ScanSensor
from igibson.sensors.vision_sensor import VisionSensor
from igibson.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from igibson.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.tasks.reaching_random_task import ReachingRandomTask
from igibson.tasks.dynamic_reaching_random_task import DynamicReachingRandomTask
from igibson.tasks.room_rearrangement_task import RoomRearrangementTask
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT
from igibson.utils.utils import quatToXYZW

from igibson.utils.utils import l2_distance
from igibson.external.pybullet_tools.utils import pairwise_collision, set_base_values, get_collision_fn
from igibson.external.pybullet_tools.utils import (
    # plan_base_motion_2d,
    # plan_base_motion,
    # plan_base_motion_ref,
    # plan_base_motion_ref_stat_dynamic,
    # plan_base_motion_update_dynamic_rrg,
    # plan_joint_motion,
    # plan_joint_motion_ref_static_dynamic,
    # plan_joint_motion_update_dynamic_rrg,
    set_base_values_with_z,
    get_base_distance_fn,
    set_joint_positions,
    link_from_name,
    joints_from_names,
    get_max_limits, get_min_limits
)

from igibson.external.pybullet_tools.utils import (
    control_joints,
    get_base_values,
    get_joint_positions,
    get_sample_fn,
    is_collision_free,
)

class iGibsonEnv(BaseEnv):#, EzPickle
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(iGibsonEnv, self).__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
        )
        # EzPickle.__init__(
        #     self,
        #     config_file=config_file,
        #     # scene_id=scene_id,
        #     # mode=mode,
        #     # action_timestep=action_timestep,
        #     # physics_timestep=physics_timestep,
        #     # device_idx=device_idx,
        #     # render_to_tensor=render_to_tensor,
        #     )
        self.automatic_reset = automatic_reset
        self.robot_id = self.robots[0].robot_ids[0]
        self.arm_joint_ids = joints_from_names(
                self.robots[0].robot_ids[0],
                [
                    "torso_lift_joint",
                    "shoulder_pan_joint",
                    "shoulder_lift_joint",
                    "upperarm_roll_joint",
                    "elbow_flex_joint",
                    "forearm_roll_joint",
                    "wrist_flex_joint",
                    "wrist_roll_joint",
                ],
            )

    def load_task_setup(self):
        """
        Load task setup
        """
        self.initial_pos_z_offset = self.config.get("initial_pos_z_offset", 0.1)
        # s = 0.5 * G * (t ** 2)
        drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
        assert drop_distance < self.initial_pos_z_offset, "initial_pos_z_offset is too small for collision checking"

        # ignore the agent's collision with these body ids
        self.collision_ignore_body_b_ids = set(self.config.get("collision_ignore_body_b_ids", []))
        # ignore the agent's collision with these link ids of itself
        self.collision_ignore_link_a_ids = set(self.config.get("collision_ignore_link_a_ids", []))

        # discount factor
        self.discount_factor = self.config.get("discount_factor", 0.99)

        # domain randomization frequency
        self.texture_randomization_freq = self.config.get("texture_randomization_freq", None)
        self.object_randomization_freq = self.config.get("object_randomization_freq", None)

        # task
        if self.config["task"] == "point_nav_fixed":
            self.task = PointNavFixedTask(self)
        elif self.config["task"] == "point_nav_random":
            self.task = PointNavRandomTask(self)
        elif self.config["task"] == "interactive_nav_random":
            self.task = InteractiveNavRandomTask(self)
        elif self.config["task"] == "dynamic_nav_random":
            self.task = DynamicNavRandomTask(self)
        elif self.config["task"] == "reaching_random":
            self.task = ReachingRandomTask(self)
        elif self.config["task"] == "dynamic_reaching_random":
            self.task = DynamicReachingRandomTask(self)
        elif self.config["task"] == "room_rearrangement":
            self.task = RoomRearrangementTask(self)
        else:
            self.task = None

    def build_obs_space(self, shape, low, high):
        """
        Helper function that builds individual observation spaces
        """
        return gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float32)

    def load_observation_space(self):
        """
        Load observation space
        """
        self.output = self.config["output"]
        self.image_width = self.config.get("image_width", 128)
        self.image_height = self.config.get("image_height", 128)
        observation_space = OrderedDict()
        sensors = OrderedDict()
        vision_modalities = []
        scan_modalities = []

        if "task_obs" in self.output:
            observation_space["task_obs"] = self.build_obs_space(
                shape=(self.task.task_obs_dim,), low=-np.inf, high=np.inf
            )
        if "rgb" in self.output:
            observation_space["rgb"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb")
        if "depth" in self.output:
            observation_space["depth"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=1.0
            )
            vision_modalities.append("depth")
        if "pc" in self.output:
            observation_space["pc"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("pc")
        if "optical_flow" in self.output:
            observation_space["optical_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 2), low=-np.inf, high=np.inf
            )
            vision_modalities.append("optical_flow")
        if "scene_flow" in self.output:
            observation_space["scene_flow"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("scene_flow")
        if "normal" in self.output:
            observation_space["normal"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=-np.inf, high=np.inf
            )
            vision_modalities.append("normal")
        if "seg" in self.output:
            observation_space["seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_CLASS_COUNT
            )
            vision_modalities.append("seg")
        if "ins_seg" in self.output:
            observation_space["ins_seg"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 1), low=0.0, high=MAX_INSTANCE_COUNT
            )
            vision_modalities.append("ins_seg")
        if "rgb_filled" in self.output:  # use filler
            observation_space["rgb_filled"] = self.build_obs_space(
                shape=(self.image_height, self.image_width, 3), low=0.0, high=1.0
            )
            vision_modalities.append("rgb_filled")
        if "scan" in self.output:
            self.n_horizontal_rays = self.config.get("n_horizontal_rays", 128)
            self.n_vertical_beams = self.config.get("n_vertical_beams", 1)
            assert self.n_vertical_beams == 1, "scan can only handle one vertical beam for now"
            observation_space["scan"] = self.build_obs_space(
                shape=(self.n_horizontal_rays * self.n_vertical_beams, 1), low=0.0, high=1.0
            )
            scan_modalities.append("scan")
        if "occupancy_grid" in self.output:
            self.grid_resolution = self.config.get("grid_resolution", 128)
            self.occupancy_grid_space = gym.spaces.Box(
                low=0.0, high=1.0, shape=(self.grid_resolution, self.grid_resolution, 1)
            )
            observation_space["occupancy_grid"] = self.occupancy_grid_space
            scan_modalities.append("occupancy_grid")

        if "bump" in self.output:
            observation_space["bump"] = gym.spaces.Box(low=0.0, high=1.0, shape=(1,))
            sensors["bump"] = BumpSensor(self)

        if len(vision_modalities) > 0:
            sensors["vision"] = VisionSensor(self, vision_modalities)

        if len(scan_modalities) > 0:
            sensors["scan_occ"] = ScanSensor(self, scan_modalities)

        self.observation_space = gym.spaces.Dict(observation_space)
        self.sensors = sensors

    def load_action_space(self):
        """
        Load action space
        """
        self.action_space = self.robots[0].action_space

    def load_miscellaneous_variables(self):
        """
        Load miscellaneous variables for book keeping
        """
        self.current_step = 0
        self.collision_step = 0
        self.current_episode = 0
        self.collision_links = []

    def load(self):
        """
        Load environment
        """
        super(iGibsonEnv, self).load()
        self.load_task_setup()
        self.load_observation_space()
        self.load_action_space()
        self.load_miscellaneous_variables()

    def get_state(self, collision_links=[]):
        """
        Get the current observation

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """
        state = OrderedDict()
        if "task_obs" in self.output:
            state["task_obs"] = self.task.get_task_obs(self)
        if "vision" in self.sensors:
            vision_obs = self.sensors["vision"].get_obs(self)
            for modality in vision_obs:
                state[modality] = vision_obs[modality]
        if "scan_occ" in self.sensors:
            scan_obs = self.sensors["scan_occ"].get_obs(self)
            for modality in scan_obs:
                state[modality] = scan_obs[modality]
        if "bump" in self.sensors:
            state["bump"] = self.sensors["bump"].get_obs(self)

        return state

    def run_simulation(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links(collision_links)

    def filter_collision_links(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            new_collision_links.append(item)
        return new_collision_links


    def run_simulation_manip(self):
        """
        Run simulation for one action timestep (same as one render timestep in Simulator class)

        :return: collision_links: collisions from last physics timestep
        """
        self.simulator_step()
        collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
        return self.filter_collision_links_manip(collision_links)

    def filter_collision_links_manip(self, collision_links):
        """
        Filter out collisions that should be ignored

        :param collision_links: original collisions, a list of collisions
        :return: filtered collisions
        """
        new_collision_links = []
        for item in collision_links:
            # print("item in filter coll links", self.robots[0].robot_ids[0], item[0], item[1],item[2],item[3],item[4])
            # ignore collision with body b
            if item[2] in self.collision_ignore_body_b_ids:
                continue

            # ignore collision with robot link a
            if item[3] in self.collision_ignore_link_a_ids:
                continue

            # ignore self collision with robot link a (body b is also robot itself)
            if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
                continue
            
            # this is for the disabled collisions for the fetch robot, and its combination
            if item[2] == self.robots[0].robot_ids[0] and item[3]==3 and item[4] in {13, 14, 15, 16, 22}:
                # print("disabled_collisions working in dry run step, item", item)
                continue
            if item[2] == self.robots[0].robot_ids[0] and item[4]==3 and item[3] in {13, 14, 15, 16, 22}:
                # print("disabled_collisions working in dry run step, item", item)
                continue
            new_collision_links.append(item)
        return new_collision_links
    
    def populate_info(self, info):
        """
        Populate info dictionary with any useful information
        """
        info["episode_length"] = self.current_step
        info["collision_step"] = self.collision_step

    def step(self, action):
        """
        Apply robot's action.
        Returns the next state, reward, done and info,
        following OpenAI Gym's convention

        :param action: robot actions
        :return: state: next observation
        :return: reward: reward of this time step
        :return: done: whether the episode is terminated
        :return: info: info dictionary with any useful information
        """
        self.current_step += 1
        if action is not None:
            self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        state = self.get_state(collision_links)
        info = {}
        reward, info = self.task.get_reward(self, collision_links, action, info)
        done, info = self.task.get_termination(self, collision_links, action, info)
        self.task.step(self)
        self.populate_info(info)

        if done and self.automatic_reset:
            info["last_observation"] = state
            state = self.reset()

        return state, reward, done, info
    

    def step_dry_run(self):
        """
        Apply robot's action. updated by sandeep
        :return: info: info dictionary with any useful information
        """
        self.max_collisions_allowed = self.config.get("max_collisions_allowed", 5)

        self.current_step += 1
        # if action is not None:
        #     self.robots[0].apply_action(action)
        collision_links = self.run_simulation()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        # state = self.get_state(collision_links)
        # info = {}
        # reward, info = self.task.get_reward(self, collision_links, action, info)
        # done, info = self.task.get_termination(self, collision_links, action, info)
        # self.task.step(self)
        # self.populate_info(info)
        
        done = self.collision_step > self.max_collisions_allowed
        if done:
            print("done in the dry_run_step function, and coll_step", done, self.collision_step)

        # if done and self.automatic_reset:
        #     info["last_observation"] = state
        #     state = self.reset()

        # return state, reward, done, info
        return done

    def step_dry_run_manip(self):
        """
        Apply robot's action. updated by sandeep
        :return: info: info dictionary with any useful information
        """
        self.max_collisions_allowed = self.config.get("max_collisions_allowed", 5)

        self.current_step += 1
        # if action is not None:
        #     self.robots[0].apply_action(action)
        collision_links = self.run_simulation_manip()
        self.collision_links = collision_links
        self.collision_step += int(len(collision_links) > 0)

        # state = self.get_state(collision_links)
        # info = {}
        # reward, info = self.task.get_reward(self, collision_links, action, info)
        # done, info = self.task.get_termination(self, collision_links, action, info)
        # self.task.step(self)
        # self.populate_info(info)
        
        done = self.collision_step > self.max_collisions_allowed
        if done:
            print("done in the dry_run_step function, and coll_step", done, self.collision_step)

        # if done and self.automatic_reset:
        #     info["last_observation"] = state
        #     state = self.reset()

        # return state, reward, done, info
        return done

    def check_collision(self, body_id):
        """
        Check with the given body_id has any collision after one simulator step

        :param body_id: pybullet body id
        :return: whether the given body_id has no collision
        """
        self.simulator_step()
        collisions = list(p.getContactPoints(bodyA=body_id))

        # if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
        # for item in collisions:
        #     # logging.debug("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))
        #     print("bodyA:{}, bodyB:{}, linkA:{}, linkB:{}".format(item[1], item[2], item[3], item[4]))

        return len(collisions) == 0

    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        """
        Reset position and orientation for the robot or the object

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :param offset: z offset
        """
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = self.initial_pos_z_offset

        is_robot = isinstance(obj, BaseRobot)
        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), "wxyz"))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])

    def test_valid_position(self, obj, pos, orn=None):
        """
        Test if the robot or the object can be placed with no collision

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        :return: validity
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id
        has_collision = self.check_collision(body_id)
        return has_collision

    def collision_function(self, body, q, orn=None):
        obstacles = []
        obstacles.append(self.scene.mesh_body_id)
        # print('mp_obstacles in collision function in igibson_env', obstacles)
        max_distance = 0
        set_base_values(body.robot_ids[0], q)
        # self.set_pos_orn_with_z_offset(body, q, orn)#set base values function from pybullet utils
        return not(any(pairwise_collision(body.robot_ids[0], obs, max_distance=max_distance) for obs in obstacles))

    def get_ik_parameters(self):
        """
        Get IK parameters such as joint limits, joint damping, reset position, etc

        :return: IK parameters
        """
        max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
        # if self.robot_type == "Fetch":
        max_limits = [0.0, 0.0] + get_max_limits(self.robot_id, self.arm_joint_ids)
        min_limits = [0.0, 0.0] + get_min_limits(self.robot_id, self.arm_joint_ids)
        # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
        min_limits[2] += 0.02
        rest_position = [0.0, 0.0] + list(get_joint_positions(self.robot_id, self.arm_joint_ids))
        joint_range = list(np.array(max_limits) - np.array(min_limits))
        joint_range = [item + 1 for item in joint_range]
        joint_damping = [0.01 for _ in joint_range]

        # elif self.robot_type == "Movo":
        #     max_limits = get_max_limits(self.robot_id, self.robot.all_joints)
        #     min_limits = get_min_limits(self.robot_id, self.robot.all_joints)
        #     rest_position = list(get_joint_positions(self.robot_id, self.robot.all_joints))
        #     joint_range = list(np.array(max_limits) - np.array(min_limits))
        #     joint_range = [item + 1 for item in joint_range]
        #     joint_damping = [0.1 for _ in joint_range]

        return (max_limits, min_limits, rest_position, joint_range, joint_damping)

    def get_arm_joint_positions(self, arm_ik_goal, threshold = 1):
        """
        Attempt to find arm_joint_positions that satisfies arm_subgoal
        If failed, return None

        :param arm_ik_goal: [x, y, z] in the world frame
        :return: arm joint positions
        """
        # ik_start = time()

        max_limits, min_limits, rest_position, joint_range, joint_damping = self.get_ik_parameters()

        n_attempt = 0
        max_attempt = 75
        sample_fn = get_sample_fn(self.robot_id, self.arm_joint_ids)
        base_pose = get_base_values(self.robot_id)
        state_id = p.saveState()
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        # find collision-free IK solution for arm_subgoal
        min_dist = 5
        while n_attempt < max_attempt:
            # if self.robot_type == "Movo":
            #     self.robot.tuck()

            set_joint_positions(self.robot_id, self.arm_joint_ids, sample_fn())
            arm_joint_positions = p.calculateInverseKinematics(
                self.robot_id,
                self.robots[0].end_effector_part_index(),
                targetPosition=arm_ik_goal,
                # targetOrientation=self.robots[0].get_orientation(),
                lowerLimits=min_limits,
                upperLimits=max_limits,
                jointRanges=joint_range,
                restPoses=rest_position,
                jointDamping=joint_damping,
                solver=p.IK_SDLS,
                maxNumIterations=100,
            )

            # if self.robot_type == "Fetch":
            arm_joint_positions = arm_joint_positions[2:10]
            # elif self.robot_type == "Movo":
            #     arm_joint_positions = arm_joint_positions[:8]

            set_joint_positions(self.robot_id, self.arm_joint_ids, arm_joint_positions)

            dist = l2_distance(self.robots[0].get_end_effector_position(), arm_ik_goal)
            # print('dist', dist)
            if dist < min_dist:
                min_dist = dist
            arm_ik_threshold = threshold
            if dist > arm_ik_threshold:
                n_attempt += 1
                continue

            # need to simulator_step to get the latest collision
            self.simulator_step()

            # simulator_step will slightly move the robot base and the objects
            set_base_values_with_z(self.robot_id, base_pose, z=self.initial_pos_z_offset)
            # self.reset_object_states()
            # TODO: have a princpled way for stashing and resetting object states

            # arm should not have any collision
            # if self.robot_type == "Movo":
            #     collision_free = is_collision_free(body_a=self.robot_id, link_a_list=self.arm_joint_ids_all)
                # ignore linear link
            # else:
            #     collision_free = is_collision_free(body_a=self.robot_id, link_a_list=self.arm_joint_ids)

            # if not collision_free:
            #     n_attempt += 1
            #     # print('arm has collision')
            #     continue

            # # gripper should not have any self-collision
            # collision_free = is_collision_free(
            #     body_a=self.robot_id, link_a_list=[self.robot.end_effector_part_index()], body_b=self.robot_id
            # )
            # if not collision_free:
            #     n_attempt += 1
            #     print("gripper has collision")
            #     continue

            # self.episode_metrics['arm_ik_time'] += time() - ik_start
            # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            p.restoreState(state_id)
            p.removeState(state_id)
            return arm_joint_positions, min_dist

        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        # self.episode_metrics['arm_ik_time'] += time() - ik_start
        return None, min_dist


    def collision_function_manip(self, body, q):
        arm_joint_positions, min_dist = self.get_arm_joint_positions(q, threshold=1)
        if arm_joint_positions is None:
            arm_joint_positions, min_dist = self.get_arm_joint_positions(q, threshold=min_dist+0.2)
        joints = self.arm_joint_ids
        mp_obstacles_d = []
        mp_obstacles_d.append(self.scene.mesh_body_id)# for the static scenes
        mp_obstacles_d.append(self.task.dynamic_objects[0].robot_ids[0])
        self_collisions = True
        attachments = []
        disabled_collisions = {(3, 13), (3, 14),(3, 15),(3, 16),(3, 22)}
        cf = get_collision_fn(body, joints, mp_obstacles_d, attachments, self_collisions, disabled_collisions,
                                    custom_limits={}, allow_collision_links=[19])
        
        return not(cf(arm_joint_positions))


    def land(self, obj, pos, orn):
        """
        Land the robot or the object onto the floor, given a valid position and orientation

        :param obj: an instance of robot or object
        :param pos: position
        :param orn: orientation
        """
        is_robot = isinstance(obj, BaseRobot)

        self.set_pos_orn_with_z_offset(obj, pos, orn)

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

        body_id = obj.robot_ids[0] if is_robot else obj.body_id

        land_success = False
        # land for maximum 1 second, should fall down ~5 meters
        max_simulator_step = int(1.0 / self.action_timestep)
        for _ in range(max_simulator_step):
            self.simulator_step()
            if len(p.getContactPoints(bodyA=body_id)) > 0:
                land_success = True
                break

        if not land_success:
            print("WARNING: Failed to land")

        if is_robot:
            obj.robot_specific_reset()
            obj.keep_still()

    def reset_variables(self):
        """
        Reset bookkeeping variables for the next new episode
        """
        self.current_episode += 1
        self.current_step = 0
        self.collision_step = 0
        self.collision_links = []

    def randomize_domain(self):
        """
        Domain randomization
        Object randomization loads new object models with the same poses
        Texture randomization loads new materials and textures for the same object models
        """
        if self.object_randomization_freq is not None:
            if self.current_episode % self.object_randomization_freq == 0:
                self.reload_model_object_randomization()
        if self.texture_randomization_freq is not None:
            if self.current_episode % self.texture_randomization_freq == 0:
                self.simulator.scene.randomize_texture()

    def reset(self):
        """
        Reset episode
        """
        self.randomize_domain()
        # move robot away from the scene
        self.robots[0].set_position([100.0, 100.0, 100.0])
        self.task.reset_scene(self)
        self.task.reset_agent(self)
        self.simulator.sync()
        state = self.get_state()
        self.reset_variables()

        return state


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="which config file to use [default: use yaml files in examples/configs]")
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui"],
        default="headless",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = iGibsonEnv(config_file=args.config, mode=args.mode, action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    step_time_list = []
    for episode in range(100):
        print("Episode: {}".format(episode))
        start = time.time()
        env.reset()
        for _ in range(100):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            print("reward", reward)
            if done:
                break
        print("Episode finished after {} timesteps, took {} seconds.".format(env.current_step, time.time() - start))
    env.close()
