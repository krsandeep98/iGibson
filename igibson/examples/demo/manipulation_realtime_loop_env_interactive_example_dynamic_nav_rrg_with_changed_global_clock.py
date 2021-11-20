import logging
import os
# import time

import sys
logging.disable(sys.maxsize)

import numpy as np
import pybullet as p
# import matplotlib.pyplot as plt
from time import time

from igibson.external.pybullet_tools.utils import (
    plan_base_motion_2d,
    plan_base_motion,
    plan_base_motion_ref,
    plan_base_motion_ref_stat_dynamic,
    plan_base_motion_update_dynamic_rrg,
    plan_joint_motion,
    plan_joint_motion_ref_static_dynamic,
    plan_joint_motion_update_dynamic_rrg,
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

import igibson
from igibson.envs.igibson_env_manip import iGibsonEnv
from igibson.utils.utils import rotate_vector_2d, l2_distance
# from igibson.render.profiler import Profiler

# arm_default_joint_positions = (
#                 0.10322468280792236,
#                 -1.414019864768982,
#                 1.5178184935241699,
#                 0.8189625336474915,
#                 2.200358942909668,
#                 2.9631312579803466,
#                 -1.2862852996643066,
#                 0.0008453550418615341,
#             )

def plan_base_motions(env, ref_rrg, updated_agent_path, goal, robot_number = 0, maxtime=10):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        # state = env.get_state()
        print("Start FETCH in update one ", env.robots[robot_number].get_position())
        print("Goal Fetch", goal)
        print("current position of turtlebot", env.task.dynamic_objects[robot_number].get_position())
        # print("robot_ids for the 0 robot in plan base motion function", env.robots[robot_number].robot_ids)
        x, y, theta = goal
        # grid = state["occupancy_grid"]


        # yaw = env.robots[robot_number].get_rpy()[2]
        # occupancy_range = env.sensors["scan_occ"].occupancy_range
        # half_occupancy_range = occupancy_range / 2.0
        # robot_position_xy = env.robots[robot_number].get_position()[:2]
        # corners = [
        #     robot_position_xy + rotate_vector_2d(local_corner, -yaw)
        #     for local_corner in [
        #         np.array([half_occupancy_range, half_occupancy_range]),
        #         np.array([half_occupancy_range, -half_occupancy_range]),
        #         np.array([-half_occupancy_range, half_occupancy_range]),
        #         np.array([-half_occupancy_range, -half_occupancy_range]),
        #     ]
        # ]

        mp_obstacles = []
        mp_obstacles.append(env.scene.mesh_body_id)# for the static scenes
        mp_obstacles.append(env.task.dynamic_objects[0].robot_ids[0])
        # mp_obstacles.extend(env.scene._load())# for the static scenes
        # mp_obstacles.extend(env.scene.get_body_ids()) #for the interactive ones
        # print('mp_obstacles in plan base motions function', mp_obstacles)

        path, coll_or_not = plan_base_motion_update_dynamic_rrg(
            # env.robots[robot_number].robot_ids[0],
            ref_rrg,
            env.robots[robot_number],
            updated_agent_path,
            # env,
            [x, y, theta],
            # (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # map_2d=grid,
            # occupancy_range= occupancy_range,
            # grid_resolution=env.grid_resolution,
            # robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
            # resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=mp_obstacles,
            # algorithm="rrg",
            max_time=maxtime
            # optimize_iter=10,
        )

        return path, coll_or_not

def get_ik_parameters(env):
    """
    Get IK parameters such as joint limits, joint damping, reset position, etc

    :return: IK parameters
    """
    max_limits, min_limits, rest_position, joint_range, joint_damping = None, None, None, None, None
    # if self.robot_type == "Fetch":
    max_limits = [0.0, 0.0] + get_max_limits(env.robots[0].robot_ids[0], env.arm_joint_ids)
    min_limits = [0.0, 0.0] + get_min_limits(env.robots[0].robot_ids[0], env.arm_joint_ids)
    # increase torso_lift_joint lower limit to 0.02 to avoid self-collision
    min_limits[2] += 0.02
    rest_position = [0.0, 0.0] + list(get_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids))
    joint_range = list(np.array(max_limits) - np.array(min_limits))
    joint_range = [item + 1 for item in joint_range]
    joint_damping = [0.01 for _ in joint_range]#0.1


    return (max_limits, min_limits, rest_position, joint_range, joint_damping)

def get_arm_joint_positions(env, arm_ik_goal, max_att = 100, max_iters=100, threshold=0.5):
    """
    Attempt to find arm_joint_positions that satisfies arm_subgoal
    If failed, return None

    :param arm_ik_goal: [x, y, z] in the world frame
    :return: arm joint positions
    """
    # ik_start = time()

    max_limits, min_limits, rest_position, joint_range, joint_damping = get_ik_parameters(env)

    # print("arm_joint_ids in the get_arm_joint_positions fn ", env.arm_joint_ids)
    # print("max_limits, min_limits, rest_position, joint_range, joint_damping", max_limits, min_limits, rest_position, joint_range, joint_damping)
    n_attempt = 0
    max_attempt = max_att#100#75
    sample_fn = get_sample_fn(env.robots[0].robot_ids[0], env.arm_joint_ids)
    base_pose = get_base_values(env.robots[0].robot_ids[0])
    state_id = p.saveState()
    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
    # find collision-free IK solution for arm_subgoal
    min_dist = 5
    while n_attempt < max_attempt:
        # if self.robot_type == "Movo":
        #     self.robot.tuck()

        set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, sample_fn())
        arm_joint_positions = p.calculateInverseKinematics(
            env.robots[0].robot_ids[0],
            # self.robot.end_effector_part_index(),
            env.robots[0].end_effector_part_index(),
            targetPosition=arm_ik_goal,
            # targetOrientation=self.robots[0].get_orientation(),
            lowerLimits=min_limits,
            upperLimits=max_limits,
            jointRanges=joint_range,
            restPoses=rest_position,
            jointDamping=joint_damping,
            solver=p.IK_SDLS,#IK_DLS was amother option, not as good as this one though
            maxNumIterations=max_iters,#100
            residualThreshold = 0.01
        )

        # if self.robot_type == "Fetch":
        arm_joint_positions = arm_joint_positions[2:10]
        # print("arm_joint_positions in the get arm joint function{}, n attempt{}".format(arm_joint_positions, n_attempt))
        # elif self.robot_type == "Movo":
        #     arm_joint_positions = arm_joint_positions[:8]

        set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, arm_joint_positions)

        # dist = l2_distance(self.robot.get_end_effector_position(), arm_ik_goal)
        dist = l2_distance(env.robots[0].get_end_effector_position(), arm_ik_goal)

        if dist < min_dist:
            min_dist = dist
        # print("end effector position {}, arm_ik_goal{}, distance {:.4f}, min dist {:.4f}".format(env.robots[0].get_end_effector_position(), arm_ik_goal, dist, min_dist))
        # print("end effector position {}, distance {:.4f}, min dist {:.4f}".format(env.robots[0].get_end_effector_position(),dist, min_dist))
        # print('dist', dist)
        arm_ik_threshold = threshold#1.5#0.05
        if dist > arm_ik_threshold:
            # print("threshold is too small and need better arm_joint positions, dist", dist)
            n_attempt += 1
            continue

        # need to simulator_step to get the latest collision
        env.simulator_step()

        # simulator_step will slightly move the robot base and the objects
        set_base_values_with_z(env.robots[0].robot_ids[0], base_pose, z=env.initial_pos_z_offset)
        # self.reset_object_states()
        # TODO: have a princpled way for stashing and resetting object states

        # arm should not have any collision
        
        # TODO the following two collision free calls check for arm and gripper colls, ignoring those for now, cause don't know how to deal with it
        # collision_free = is_collision_free(body_a=env.robots[0].robot_ids[0], link_a_list=env.arm_joint_ids)
        # if not collision_free:
        #     n_attempt += 1
        #     print('arm has collision')
        #     continue

        # gripper should not have any self-collision
        # collision_free = is_collision_free(
        #     body_a=env.robots[0].robot_ids[0], link_a_list=[env.robots[0].end_effector_part_index()], body_b=env.robots[0].robot_ids[0]
        # )
        # if not collision_free:
        #     n_attempt += 1
        #     print("gripper has collision")
        #     continue

        # self.episode_metrics['arm_ik_time'] += time() - ik_start
        # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        print("yes!! EE position {}, distance {:.4f}, n_attempt {}".format(env.robots[0].get_end_effector_position(),dist, n_attempt))
        return arm_joint_positions, min_dist

    # p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
    p.restoreState(state_id)
    p.removeState(state_id)
    print("oh, no! EE position {}, distance {:.4f}, n_attempt {}".format(env.robots[0].get_end_effector_position(),dist, n_attempt))
    # self.episode_metrics['arm_ik_time'] += time() - ik_start
    return None, min_dist

def plan_arm_motion(env, arm_joint_positions):
    """
    Attempt to reach arm arm_joint_positions and return arm trajectory
    If failed, reset the arm to its original pose and return None

    :param arm_joint_positions: final arm joint position to reach
    :return: arm trajectory or None if no plan can be found
    """
    disabled_collisions = {}
    # if self.robot_type == "Fetch":
    disabled_collisions = {
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "torso_fixed_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "shoulder_lift_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "upperarm_roll_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "forearm_roll_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "elbow_flex_link")),
    }
    # elif self.robot_type == "Movo":

    # if self.fine_motion_plan:
    #     self_collisions = True
    #     mp_obstacles = self.mp_obstacles
    # else:
    self_collisions = False
    mp_obstacles = []

    plan_arm_start = time()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
    state_id = p.saveState()

    allow_collision_links = []
    # if self.robot_type == "Fetch":
    allow_collision_links = [19]
    # elif self.robot_type == "Movo":
    #     allow_collision_links = [23, 24]

    arm_path = plan_joint_motion(
        env.robots[0].robot_ids[0],
        env.arm_joint_ids,
        arm_joint_positions,
        disabled_collisions=disabled_collisions,
        self_collisions=self_collisions,
        obstacles=mp_obstacles,
        # algorithm=self.arm_mp_algo,
        algorithm="birrt",
        allow_collision_links=allow_collision_links,
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
    p.restoreState(state_id)
    p.removeState(state_id)
    return arm_path

def dry_run_arm_plan(env, arm_path):
    """
    Dry run arm motion plan by setting the arm joint position without physics simulation

    :param arm_path: arm trajectory or None if no plan can be found
    """
    base_pose = get_base_values(env.robots[0].robot_ids[0])
    if arm_path is not None:
        # if self.mode in ["gui", "iggui", "pbgui"]:
        for joint_way_point in arm_path:
            set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, joint_way_point)
            # set_base_values_with_z(env.robots[0].robot_ids[0], base_pose, z=self.initial_height)
            set_base_values_with_z(env.robots[0].robot_ids[0], base_pose, z=env.initial_pos_z_offset)
            # self.simulator_sync()
            env.step_dry_run()
            # sleep(0.02)  # animation
        # else:
        #     set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, arm_path[-1])
    else:
        # print('arm mp fails')
        # if self.robot_type == "Movo":
        #     self.robot.tuck()
        set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, arm_default_joint_positions)

def plan_arm_motion_rrg(env, arm_joint_positions, maxtime = 10):
    """
    Attempt to reach arm arm_joint_positions and return arm trajectory
    If failed, reset the arm to its original pose and return None

    :param arm_joint_positions: final arm joint position to reach
    :return: arm trajectory or None if no plan can be found
    """
    disabled_collisions = {}
    # if self.robot_type == "Fetch":
    disabled_collisions = {
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "torso_fixed_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "shoulder_lift_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "upperarm_roll_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "forearm_roll_link")),
        (link_from_name(env.robots[0].robot_ids[0], "torso_lift_link"), link_from_name(env.robots[0].robot_ids[0], "elbow_flex_link")),
    }

    # print("disabled collisions in planning fn", disabled_collisions)
    # if self.fine_motion_plan:
    #     self_collisions = True
    #     mp_obstacles = self.mp_obstacles
    # else:
    self_collisions = True#False
    # mp_obstacles = []
    # print("Start FETCH in rrg plan one", env.robots[0].get_end_effector_position())
    mp_obstacles = []
    mp_obstacles.append(env.scene.mesh_body_id)# for the static scenes
    # mp_obstacles.append(env.task.dynamic_objects[0].robot_ids[0])

    mp_obstacles_d = []
    mp_obstacles_d.append(env.scene.mesh_body_id)# for the static scenes
    mp_obstacles_d.append(env.task.dynamic_objects[0].robot_ids[0])

    # plan_arm_start = time()
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
    state_id = p.saveState()

    allow_collision_links = []
    # if self.robot_type == "Fetch":
    allow_collision_links = [19]
    # elif self.robot_type == "Movo":
    #     allow_collision_links = [23, 24]

    # arm_path
    ref_rrg, path, coll_or_not = plan_joint_motion_ref_static_dynamic(
        env.robots[0].robot_ids[0],
        env.arm_joint_ids,
        arm_joint_positions,
        disabled_collisions=disabled_collisions,
        self_collisions=self_collisions,
        obstacles=mp_obstacles,
        dynamic_obstacles = mp_obstacles_d,
        # algorithm=self.arm_mp_algo,
        algorithm="rrg",
        allow_collision_links=allow_collision_links,
        max_time = maxtime
    )
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
    p.restoreState(state_id)
    p.removeState(state_id)
    #the one above with restore state and remove state is probably a good way to go about it 
    return ref_rrg, path, coll_or_not #arm_path


def plan_joint_motions_update(env, ref_rrg, updated_agent_path, maxtime=10):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        
        # start_in_update = env.robots[robot_number].get_position() # we would need the get end effector position or get joint positions
        # print("Start FETCH in update one ", start_in_update)
        # print("Goal Fetch", goal)
        # print("current position of turtlebot", env.task.dynamic_objects[robot_number].get_position())
        
        # x, y, theta = goal
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()   


        path, next_path, coll_or_not = plan_joint_motion_update_dynamic_rrg(
            # env.robots[robot_number].robot_ids[0],
            ref_rrg,
            env.robots[0].robot_ids[0],
            env.arm_joint_ids,
            updated_agent_path,
            # [x, y, theta],
            # goal,
            # obstacles=mp_obstacles,
            # algorithm="rrg",
            max_time=maxtime
        )

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)
        # have to get the reset info done properly, but just landing it on the start position might not be the best idea
        # env.land(env.robots[0], start_in_update, env.task.initial_orn) # need this to ensure that we reset the robot back to its original position, 

        return path, next_path, coll_or_not

def plan_joint_motions_update_pred(env, ref_rrg, updated_agent_path, p1, pt, maxtime=10):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """        
        # x, y, theta = goal
        # print("current position of turtlebot before pred", env.task.dynamic_objects[0].get_position())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
        state_id = p.saveState()
        # print("p1 and pt in update pred", p1, pt, maxtime)

        t0 = time()
        waypoint_counter_turtlebot = 0
        for way_point1 in p1:
            set_base_values_with_z(
                env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
            )
            # waypoint_counter_fetch+=1
            waypoint_counter_turtlebot+=1
            env.simulator.step()#or sync
            # collision_in_exec = env.step_dry_run()#we can return as soon as this is true and that would help speed up things
            # collision_in_exec = env.step_dry_run_manip()
            
            if time()- t0 > pt:
                # print("time out for the dry run for the obstacle in the pred part", time()- t0, waypoint_counter_turtlebot)
                break
                # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                # return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec

        path, next_path, coll_or_not = plan_joint_motion_update_dynamic_rrg(
            # env.robots[robot_number].robot_ids[0],
            ref_rrg,
            env.robots[0].robot_ids[0],
            env.arm_joint_ids,
            updated_agent_path,
            max_time=maxtime
        )

        # print("current position of turtlebot before resetting", env.task.dynamic_objects[0].get_position())
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
        p.restoreState(state_id)
        p.removeState(state_id)

        # print("current position of turtlebot after resetting", env.task.dynamic_objects[0].get_position())
        # have to get the reset info done properly, but just landing it on the start position might not be the best idea
        # env.land(env.robots[0], start_in_update, env.task.initial_orn) # need this to ensure that we reset the robot back to its original position, 

        return path, next_path, coll_or_not



def dry_run_arm_plan_coll_break(env, path, path1, time_to_apply_the_path):
        t0 = time()
        
        # this is for updating both of the robots at the same time and is much better
        waypoint_counter_fetch = 0
        collision_in_exec = False
        base_pose = get_base_values(env.robots[0].robot_ids[0])
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            # print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
            waypoint_counter_fetch = 0
            waypoint_counter_turtlebot = 0
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot and obstacle position with waypoints", waypoint_counter_fetch)
                set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, way_point)
                set_base_values_with_z(env.robots[0].robot_ids[0], base_pose, z=env.initial_pos_z_offset)#TODO this can also be a thing which causes collisions


                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                collision_in_exec = env.step_dry_run_manip()
                if collision_in_exec:
                    print("breaking out due to collision_in_exec in dry run plan function")
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                
                if time()- t0 > time_to_apply_the_path:
                    # print("time out for the dry run together", time()- t0, waypoint_counter_fetch)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                    # yield way_point
                    # break
            
            # the next if else part is to accommodate the case where the number of waypoints are different in both paths,
            # and run the longer one after both ran together for the length of the shorter path
            if len(path) <= len(path1):
                # print("setting  obstacle position with waypoints", waypoint_counter_turtlebot)
                for i in range (len(path), len(path1)):
                    set_base_values_with_z( env.task.dynamic_objects[0].robot_ids[0], [path1[i][0], path1[i][1], path1[i][2]], z=env.initial_pos_z_offset
                    )
                    # env.simulator.step()
                    # collision_in_exec = env.step_dry_run()
                    collision_in_exec = env.step_dry_run_manip()
                    waypoint_counter_turtlebot+=1

                    if collision_in_exec:
                        print("breaking out due to collision_in_exec in dry run plan function")
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec

                    # maybe this part should be before the setbasevalues
                    if time()- t0 > time_to_apply_the_path:
                        # print("time out for the dry run of obstacle", time()- t0, waypoint_counter_turtlebot)
                        # print("waypoint for the obstacle, waypoint counter", path1[i], waypoint_counter_turtlebot)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            else:
                # print("setting robot with waypoints", waypoint_counter_fetch)
                for j in range (len(path1), len(path)):
                    # set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    # )
                    # set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, [path[j][0], path[j][1], path[j][2]])
                    # print('path[j] in dry run after path1 is done, j', path[j], j)
                    set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, path[j])
                    set_base_values_with_z(env.robots[0].robot_ids[0], base_pose, z=env.initial_pos_z_offset)

                    # env.simulator.step()
                    # collision_in_exec = env.step_dry_run()
                    collision_in_exec = env.step_dry_run_manip()
                    waypoint_counter_fetch+=1

                    if collision_in_exec:
                        print("breaking out due to collision_in_exec in dry run plan function")
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec

                    if time()- t0 > time_to_apply_the_path:
                        # print("time out for the dry run of agent alone", time()- t0, waypoint_counter_fetch)
                        # print("waypoint for the agent, waypoint counter", path[j], waypoint_counter_fetch)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
        elif path1 is not None:
            # print("length of paths, number of waypoints 2 when agent is none beginning",  len(path1))
            
            # env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            # print("setting fetch to default arm joint pos because there is no path")
            # set_joint_positions(env.robots[0].robot_ids[0], env.arm_joint_ids, arm_default_joint_positions)
            
            waypoint_counter_turtlebot = 0
            # waypoint_counter_fetch = 0
            for way_point1 in path1:
                # print("setting robot position with waypoints")
                # set_base_values_with_z(
                #     env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                # )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                # waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                # collision_in_exec = env.step_dry_run()#we can return as soon as this is true and that would help speed up things
                collision_in_exec = env.step_dry_run_manip()

                if collision_in_exec:
                    print("breaking out due to collision_in_exec in dry run plan function")
                    return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                
                if time()- t0 > time_to_apply_the_path:
                    # print("time out for the dry run for the obstacle separately", time()- t0, waypoint_counter_turtlebot)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            # print("path/path1 is not found, and setting the agent to its initial position")
            # if path is None:
            return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec #need to add this to keep things consistent with this function returning the path left to execute for all robots
        
        else:
            print("i dont know why it would get printed" )
            return path, path1, waypoint_counter_fetch, collision_in_exec



def plan_base_motions_update(env, ref_rrg, updated_agent_path, goal, robot_number = 0, maxtime=10):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        
        start_in_update = env.robots[robot_number].get_position()
        # print("Start FETCH in update one ", start_in_update)
        # print("Goal Fetch", goal)
        # print("current position of turtlebot", env.task.dynamic_objects[robot_number].get_position())
        
        x, y, theta = goal
        
        mp_obstacles = []
        # need to comment the following line for stadium
        mp_obstacles.append(env.scene.mesh_body_id)# for the static scenes
        mp_obstacles.append(env.task.dynamic_objects[0].robot_ids[0])
        # mp_obstacles.extend(env.scene._load())# for the static scenes
        # mp_obstacles.extend(env.scene.get_body_ids()) #for the interactive ones
        # print('mp_obstacles in plan base motions function', mp_obstacles)

        path, next_path, coll_or_not = plan_base_motion_update_dynamic_rrg(
            # env.robots[robot_number].robot_ids[0],
            ref_rrg,
            env.robots[robot_number],
            updated_agent_path,
            [x, y, theta],
            obstacles=mp_obstacles,
            algorithm="rrg",
            max_time=maxtime
        )
        env.land(env.robots[0], start_in_update, env.task.initial_orn)

        return path, next_path, coll_or_not

def plan_base_motion_with_dynamic_rrg(env, goal, robot_number = 0, maxtime=50):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        # state = env.get_state()
        start_in_plan_fn = env.robots[robot_number].get_position()
        print("Start FETCH ", start_in_plan_fn)
        print("Goal Fetch", goal)
        # print("current position of turtlebot before ref plan motion", env.task.dynamic_objects[robot_number].get_position())
        x, y, theta = goal
        
        yaw = env.robots[robot_number].get_rpy()[2]
        occupancy_range = env.sensors["scan_occ"].occupancy_range
        half_occupancy_range = occupancy_range / 2.0
        robot_position_xy = env.robots[robot_number].get_position()[:2]
        corners = [
            robot_position_xy + rotate_vector_2d(local_corner, -yaw)
            for local_corner in [
                np.array([half_occupancy_range, half_occupancy_range]),
                np.array([half_occupancy_range, -half_occupancy_range]),
                np.array([-half_occupancy_range, half_occupancy_range]),
                np.array([-half_occupancy_range, -half_occupancy_range]),
            ]
        ]

        mp_obstacles = []
        mp_obstacles.append(env.scene.mesh_body_id)# for the static scenes
        # mp_obstacles.append(env.task.dynamic_objects[0].robot_ids[0])

        mp_obstacles_d = []
        mp_obstacles_d.append(env.scene.mesh_body_id)# for the static scenes
        mp_obstacles_d.append(env.task.dynamic_objects[0].robot_ids[0])
        
        ref_rrg, path, coll_or_not = plan_base_motion_ref_stat_dynamic(
            env.robots[robot_number],
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=mp_obstacles,
            dynamic_obstacles = mp_obstacles_d,
            algorithm="rrg",
            max_time=maxtime
        )
        env.land(env.robots[0], start_in_plan_fn, env.task.initial_orn)
        return ref_rrg, path, coll_or_not

def plan_base_motion_dynamic(env, goal, robot_number = 0):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        state = env.get_state()
        print("start dynamic obstacle", env.task.dynamic_objects[robot_number].get_position())
        print("goal dynamic obstacle", goal)
        
        x, y, theta = goal
        # grid = state["occupancy_grid"]
        # print("grid in the plan base motion function", grid.shape)#(120, 120, 1)

        yaw = env.task.dynamic_objects[robot_number].get_rpy()[2]
        occupancy_range = env.sensors["scan_occ"].occupancy_range
        half_occupancy_range = occupancy_range / 2.0
        robot_position_xy = env.task.dynamic_objects[robot_number].get_position()[:2]
        corners = [
            robot_position_xy + rotate_vector_2d(local_corner, -yaw)
            for local_corner in [
                np.array([half_occupancy_range, half_occupancy_range]),
                np.array([half_occupancy_range, -half_occupancy_range]),
                np.array([-half_occupancy_range, half_occupancy_range]),
                np.array([-half_occupancy_range, -half_occupancy_range]),
            ]
        ]
        
        mp_obstacles = []
        mp_obstacles.append(env.scene.mesh_body_id)# for the static scenes
        mp_obstacles.append(env.robots[0].robot_ids[0])
        # mp_obstacles.extend(env.scene.get_body_ids()) #for the interactive ones

        # env.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        path, coll_or_not = plan_base_motion(
            # env.task.dynamic_objects[robot_number].robot_ids[0],
            env.task.dynamic_objects[robot_number],
            # env,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # map_2d=grid,
            # occupancy_range= occupancy_range,
            # grid_resolution=env.grid_resolution,
            # robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
            resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=mp_obstacles,
            algorithm="rrg",#somehow the rrt version sets the goal to be (0.55, 0.95), no idea why this happens, let's go with the rrg
            # optimize_iter=10,
        )

        return path, coll_or_not


def dry_run_base_plan(env, path, path1, time_to_apply_the_path):
        t0 = time()
        
        # this is for updating both of the robots at the same time and is much better
        waypoint_counter_fetch = 0
        collision_in_exec = False
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
            waypoint_counter_fetch = 0
            waypoint_counter_turtlebot = 0
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot and obstacle position with waypoints", waypoint_counter_fetch)
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                collision_in_exec = env.step_dry_run()
                
                if time()- t0 > time_to_apply_the_path:
                    print("time out for the dry run together", time()- t0, waypoint_counter_fetch)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                    # yield way_point
                    # break
            
            # the next if else part is to accommodate the case where the number of waypoints are different in both paths,
            # and run the longer one after both ran together for the length of the shorter path
            if len(path) <= len(path1):
                # print("setting  obstacle position with waypoints", waypoint_counter_turtlebot)
                for i in range (len(path), len(path1)):
                    set_base_values_with_z( env.task.dynamic_objects[0].robot_ids[0], [path1[i][0], path1[i][1], path1[i][2]], z=env.initial_pos_z_offset
                    )
                    # env.simulator.step()
                    collision_in_exec = env.step_dry_run()
                    waypoint_counter_turtlebot+=1
                    # maybe this part should be before the setbasevalues
                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of obstacle", time()- t0, waypoint_counter_turtlebot)
                        # print("waypoint for the obstacle, waypoint counter", path1[i], waypoint_counter_turtlebot)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            else:
                # print("setting robot with waypoints", waypoint_counter_fetch)
                for j in range (len(path1), len(path)):
                    set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    )
                    # env.simulator.step()
                    collision_in_exec = env.step_dry_run()
                    waypoint_counter_fetch+=1

                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of agent alone", time()- t0, waypoint_counter_fetch)
                        # print("waypoint for the agent, waypoint counter", path[j], waypoint_counter_fetch)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
        elif path1 is not None:
            print("length of paths, number of waypoints 2 when agent is none beginning",  len(path1))
            
            #this is because in the process of calculating the path for the agent, the collision checking was 
            #actually setting the agent to be at a different position, is an issue if the goal config is in collision, then it's already at goal
            # and the later distance check with target would say that it's a success but it really is not
            
            # env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            
            waypoint_counter_turtlebot = 0
            # waypoint_counter_fetch = 0
            for way_point1 in path1:
                # print("setting robot position with waypoints")
                # set_base_values_with_z(
                #     env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                # )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                # waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                collision_in_exec = env.step_dry_run()#we can return as soon as this is true and that would help speed up things
                
                if time()- t0 > time_to_apply_the_path:
                    # print("time out for the dry run for the obstacle separately", time()- t0, waypoint_counter_turtlebot)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            # print("path/path1 is not found, and setting the agent to its initial position")
            # if path is None:
            return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec #need to add this to keep things consistent with this function returning the path left to execute for all robots
        
        else:
            print("i dont know why it would get printed" )
            return path, path1, waypoint_counter_fetch, collision_in_exec

def dry_run_base_plan_coll_break(env, path, path1, time_to_apply_the_path):
        t0 = time()
        
        # this is for updating both of the robots at the same time and is much better
        waypoint_counter_fetch = 0
        collision_in_exec = False
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            # print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
            waypoint_counter_fetch = 0
            waypoint_counter_turtlebot = 0
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot and obstacle position with waypoints", waypoint_counter_fetch)
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                collision_in_exec = env.step_dry_run()
                if collision_in_exec:
                    print("breaking out due to collision_in_exec in dry run plan function")
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                
                if time()- t0 > time_to_apply_the_path:
                    # print("time out for the dry run together", time()- t0, waypoint_counter_fetch)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                    # yield way_point
                    # break
            
            # the next if else part is to accommodate the case where the number of waypoints are different in both paths,
            # and run the longer one after both ran together for the length of the shorter path
            if len(path) <= len(path1):
                # print("setting  obstacle position with waypoints", waypoint_counter_turtlebot)
                for i in range (len(path), len(path1)):
                    set_base_values_with_z( env.task.dynamic_objects[0].robot_ids[0], [path1[i][0], path1[i][1], path1[i][2]], z=env.initial_pos_z_offset
                    )
                    # env.simulator.step()
                    collision_in_exec = env.step_dry_run()
                    waypoint_counter_turtlebot+=1

                    if collision_in_exec:
                        print("breaking out due to collision_in_exec in dry run plan function")
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec

                    # maybe this part should be before the setbasevalues
                    if time()- t0 > time_to_apply_the_path:
                        # print("time out for the dry run of obstacle", time()- t0, waypoint_counter_turtlebot)
                        # print("waypoint for the obstacle, waypoint counter", path1[i], waypoint_counter_turtlebot)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            else:
                # print("setting robot with waypoints", waypoint_counter_fetch)
                for j in range (len(path1), len(path)):
                    set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    )
                    # env.simulator.step()
                    collision_in_exec = env.step_dry_run()
                    waypoint_counter_fetch+=1

                    if collision_in_exec:
                        print("breaking out due to collision_in_exec in dry run plan function")
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec

                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of agent alone", time()- t0, waypoint_counter_fetch)
                        # print("waypoint for the agent, waypoint counter", path[j], waypoint_counter_fetch)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
        elif path1 is not None:
            # print("length of paths, number of waypoints 2 when agent is none beginning",  len(path1))
            
            # env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            
            waypoint_counter_turtlebot = 0
            # waypoint_counter_fetch = 0
            for way_point1 in path1:
                # print("setting robot position with waypoints")
                # set_base_values_with_z(
                #     env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                # )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                # waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                # env.simulator.step()#or sync
                collision_in_exec = env.step_dry_run()#we can return as soon as this is true and that would help speed up things

                if collision_in_exec:
                    print("breaking out due to collision_in_exec in dry run plan function")
                    return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
                
                if time()- t0 > time_to_apply_the_path:
                    # print("time out for the dry run for the obstacle separately", time()- t0, waypoint_counter_turtlebot)
                    # print("waypoint for the agent, waypoint_obs, waypoint counter", way_point, way_point1, waypoint_counter_fetch)
                    return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec
            # print("path/path1 is not found, and setting the agent to its initial position")
            # if path is None:
            return path, path1[waypoint_counter_turtlebot:], waypoint_counter_fetch, collision_in_exec #need to add this to keep things consistent with this function returning the path left to execute for all robots
        
        else:
            print("i dont know why it would get printed" )
            return path, path1, waypoint_counter_fetch, collision_in_exec

def dry_run_base_plan_timed(env, path, path1, time_to_apply_the_path):
        """
        Dry run base motion plan by setting the base positions without physics simulation
        :param path: base waypoints or None if no plan can be found
        """
        t0 = time()
        
        # this is for updating both of the robots at the same time and is much better
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
            waypoint_counter_fetch = 0
            waypoint_counter_turtlebot = 0
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot position with waypoints")
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                waypoint_counter_fetch+=1
                waypoint_counter_turtlebot+=1
                env.simulator.step()#or sync
                
                if time()- t0 > time_to_apply_the_path:
                    print("time out for the dry run together", time()- t0)
                    print("waypoint for the agent, waypoint counter", way_point, waypoint_counter_fetch)
                    return
                    # yield way_point
                    # break
            
            # the next if else part is to accommodate the case where the number of waypoints are different in both paths,
            # and run the longer one after both ran together for the length of the shorter path
            if len(path) <= len(path1):
                for i in range (len(path), len(path1)):
                    set_base_values_with_z( env.task.dynamic_objects[0].robot_ids[0], [path1[i][0], path1[i][1], path1[i][2]], z=env.initial_pos_z_offset
                    )
                    env.simulator.step()
                    waypoint_counter_turtlebot+=1
                    # maybe this part should be before the setbasevalues
                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of obstacle", time()- t0)
                        print("waypoint for the obstacle, waypoint counter", path1[i], waypoint_counter_turtlebot)
                        return
            else:
                for j in range (len(path1), len(path)):
                    set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    )
                    env.simulator.step()
                    waypoint_counter_fetch+=1

                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of agent alone", time()- t0)
                        print("waypoint for the agent, waypoint counter", path[j], waypoint_counter_fetch)
                        return
        else:
            env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            print("path/path1 is not found, and setting the agent to its initial position")


def main():
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav.yaml")
    config_filename = os.path.join(igibson.example_config_path, "fetch_turtlebot_room_rearrangement_sandeep.yaml")
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav_sandeep_manip.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="gui")
    # env = iGibsonEnv(config_file=config_filename, mode="headless")
    
    
    env.reset()
    
    env.arm_joint_ids = joints_from_names(
                env.robots[0].robot_ids[0],
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

    # print("arm_joint_ids in the main fn ", env.arm_joint_ids)
    # print("default arm joint posn in the main fn ", arm_default_joint_positions)
    # this is for gibson RS
    goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
    path_for_base_obs, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
    reverse_path_from_b2a, coll_again_for_dynamic = plan_base_motion_dynamic(env, [-1.5, 0.75, 0], 0)
    # env.task.target_pos
    # path_for_base_obs, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
    # reverse_path_from_b2a, coll_again_for_dynamic = plan_base_motion_dynamic(env, [-1, -1, 0], 0)
    # reverse_path_from_b2a = path_for_base1[::-1]

    path_for_base_original = path_for_base_obs + reverse_path_from_b2a
    path_for_base_original = path_for_base_original + path_for_base_original + path_for_base_original + path_for_base_original
    # print("path for base1 last", path_for_base1)
    # print("path for base1 last reversed", reverse_path_from_b2a)
    if path_for_base_original is None:
        print("Oh, No! not able to find the path for the dynamic turtlebot")
    
    # still need this reset as the dynamic obstacle would stay at it's goal and not the start, for the path planning for agent
    env.reset()#added this to handle the case where the turtlebot was at 0.55, 0.95, but that was probably an rrt issue

    # distance_fn = get_base_distance_fn([1, 1, 0])
    counter = 0
    collision_counter = 0
    # short_interval_success_counter = 0
    # new_path_for_agent = None
    total_time_taken = 0
    for j in range(5):
        # collision2 = False
        collision2 = 100
        collision_in_exec = False
        collision1 = False
        path_for_base1 = path_for_base_original

        tau_l = 5
        tau_s = 1
        lamb = 5/10.0
        t_cutoff = 25#* (1+lamb)
        exec_time = tau_s #(1-lamb)*
        pred_time = (1-lamb)*tau_s
        plan_time = lamb*tau_s 
        number_of_short_intervals = (t_cutoff - tau_l)/(2*tau_s)
        # print("number_of_short_intervals", number_of_short_intervals)
        print("TIMEEEEEEEEEEEEEE, exec, pred, plan, tau_s, tau_l, cutoff",exec_time, pred_time, plan_time, tau_s, tau_l, t_cutoff)
        # env.reset()
        # for i in range(100):
        if j != 0:
            env.reset()
        
        # start_time = time()
        # print("Start Fetch before everything", env.robots[0].get_end_effector_position())
        # goal = env.task.target_pos #[-1, 1, 0]#
        # start = env.task.initial_pos
        # print("start in the main function", start)
        # path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
        # ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)#tau_l
        arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos)
        # print("First arm_joint_positions in the main function, arm_ik_goal", env.task.target_pos)
        print("arm_ik_goal {}, min_dist1 tolerance {:.4f}".format(env.task.target_pos, min_dist))
        # print("arm_joint_positions in main function", arm_joint_positions)
        if arm_joint_positions is None:
            arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos, 250, 250, min_dist+0.20)
            print("arm_ik_goal2 {}, min_dist2 tolerance {:.4f}".format(env.task.target_pos, min_dist))
            # print("arm_joint_positions in main function", arm_joint_positions)
        # path_for_base = plan_arm_motion(env, env.task.target_pos)#tau_l
        # path_for_base = plan_arm_motion(env, arm_joint_positions)#tau_l
        # print("end effector position before ref plan", env.robots[0].get_position())
        start_time = time()
        ref_rrg, path_for_base, collision1 = plan_arm_motion_rrg(env, arm_joint_positions, maxtime=tau_l)
        # print("end effector position after ref plan", env.robots[0].get_position())
        # print("path for base1", path_for_base[:-5])

        
        if collision1:
            print("let's reset the environment in hope of a new and better start and goal")
            env.reset()

            arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos)
            
            print("arm_ik_goal new after 1 reset {}, min_dist1 tolerance {:.4f}".format(env.task.target_pos, min_dist))
            if arm_joint_positions is None:
                arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos, 250, 250, min_dist+0.2)
                print("arm_ik_goal2 new after 1 reset  {}, min_dist2 tolerance {:.4f}".format(env.task.target_pos, min_dist))
            
            start_time = time()
            ref_rrg, path_for_base, collision1 = plan_arm_motion_rrg(env, arm_joint_positions, maxtime=tau_l)
            
            # ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)
        
            if path_for_base is None:
                print("Oh, No! not able to find the path for the fetch agent even after a reset, what can i do man!")
            if collision1:
                print("no point of reset, in collision again")
                env.reset()

                arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos)
                
                print("arm_ik_goal new after 2 resets {}, min_dist1 tolerance {:.4f}".format(env.task.target_pos, min_dist))
                if arm_joint_positions is None:
                    arm_joint_positions, min_dist = get_arm_joint_positions(env, env.task.target_pos, 250, 250, min_dist+0.2)
                    print("arm_ik_goal2 new after 2 resets  {}, min_dist2 tolerance {:.4f}".format(env.task.target_pos, min_dist))
                
                start_time = time()
                ref_rrg, path_for_base, collision1 = plan_arm_motion_rrg(env, arm_joint_positions, maxtime=tau_l)
                
                # ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)
                if collision1:
                    print("this is it, no more reset, in collision again")

        if path_for_base is None:
            print("Oh, No! not able to find the path for the fetch agent")
        else:
            print("the length of reference path generated", len(path_for_base))
            # print("path for base1 start", path_for_base[:5])
            # print("path for base1 end", path_for_base[-5:])
        # dry_run_base_plan(env, path_for_base, path_for_base1)
        # dry_run_base_plan(env, path_for_base, None)
        # dry_run_base_plan_timed(env, path_for_base, path_for_base1, 1.5)

        # updated_agent_path, path_for_base1 =  dry_run_base_plan(env, path_for_base, path_for_base1, tau_s/2)#tau_s/2
        
        # TODO need to do this in the plan_base_motion functions
        # env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)#since at the end of planning, if robot gets to goal, it would stay there even without executing the plan at all
        ref_path_time = time()
        time_taken_for_ref_path_calc = time()-start_time
        # proper_cutoff_time = (t_cutoff - time_taken_for_ref_path_calc)* (1+lamb)
        proper_cutoff_time = (t_cutoff - time_taken_for_ref_path_calc)* (2)
        # l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos)
        # print("distance between current and goal before reference run {}, and ref time {}".format(distance_fn(env.robots[0].get_position(), env.task.target_pos), time_taken_for_ref_path_calc))
        print("distance between current and goal before reference run {:.4f}, and ref time {:.4f}".format(l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos), time_taken_for_ref_path_calc))
        # print("proper cutoff time {:.4f}, time {}".format(proper_cutoff_time, time()-ref_path_time))
        print("proper cutoff time {:.4f}".format(proper_cutoff_time))

        # dry_run_arm_plan(env, path_for_base)
        # print("current position of turtlebot after reference traj applied ", env.task.dynamic_objects[0].get_position())
        while_loop_counter = 0
        # while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and while_loop_counter < 2:
        
        while l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos) >= min_dist and time()-ref_path_time < proper_cutoff_time and ref_rrg is not None:
        # while l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos) >= 0.01 and time()-ref_path_time < proper_cutoff_time and ref_rrg is not None:
        # while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and time()-start_time < t_cutoff and ref_rrg is not None:
            # print("in the while loop number", while_loop_counter+1)
            
            # env.task.initial_pos = env.robots[0].get_position()
            # print("end effector position before plan update joint", env.robots[0].get_position())
            # reference_trajectory_after_update_call, next_traj, collision2 = plan_joint_motions_update(env, ref_rrg, path_for_base, env.task.target_pos, 0, maxtime=plan_time)
            reference_trajectory_after_update_call, next_traj, collision2 = plan_joint_motions_update_pred(env, ref_rrg, path_for_base, maxtime=plan_time, p1=path_for_base1, pt=pred_time)
            # print("end effector position after plan update joint", env.robots[0].get_position())
            if collision2 == 0:
                print("breaking out of the while loop since there was collision with start involved")
                break
            
            # print("end effector position before plan exec joint", env.robots[0].get_end_effector_position())
            reference_traj_left_to_execute, path_for_base1, execution_counter_fetch, collision_in_exec = dry_run_arm_plan_coll_break(env, reference_trajectory_after_update_call, path_for_base1, exec_time)#tau_s/2
            # print("end effector position after plan exec joint", env.robots[0].get_position())

            if l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos) < min_dist:#0.01
                print("breaking after the first execution in the while loop, distance", l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos))
                break
            
            # print("exec counter fetch after the dry run call", execution_counter_fetch)
            if next_traj is not None:
                path_for_base = next_traj[execution_counter_fetch:]
                # print("length of the next_traj", len(path_for_base))
            else:
                # print("next traj is none")
                path_for_base = None
            
            if collision_in_exec:
                print("collision in execution and with the help of dry_run_step function")
                break


            # reference_trajectory_after_update_call2, next_traj_2, collision2 = plan_joint_motions_update(env, ref_rrg, path_for_base, env.task.target_pos, 0, maxtime=plan_time)
            reference_trajectory_after_update_call2, next_traj_2, collision2 = plan_joint_motions_update_pred(env, ref_rrg, path_for_base, maxtime=plan_time, p1=path_for_base1, pt=pred_time)
            if collision2 == 0:
                print("breaking out of the while loop since there was collision with start involved second try")
                break
            # reference_traj_left_to_execute2, path_for_base1, execution_counter_fetch2 = dry_run_base_plan(env, path_for_base, path_for_base1, exec_time)#tau_s/2
            reference_traj_left_to_execute2, path_for_base1, execution_counter_fetch2, collision_in_exec = dry_run_arm_plan_coll_break(env, path_for_base, path_for_base1, exec_time)#tau_s/2
            # reference_traj_left_to_execute, path_for_base1, execution_counter_fetch, collision_in_exec = dry_run_base_plan(env, reference_trajectory_after_update_call, path_for_base1, exec_time)#tau_s/2
            
            # need to have a condition on the counter being less than the length
            # print("exec counter fetch after the second dry run call", execution_counter_fetch2)
            if next_traj_2 is not None:
                path_for_base = next_traj_2[execution_counter_fetch2:]
                # print("length of the next_traj2", len(path_for_base))#print the next line
            else:
                # print("next traj2 is none")
                path_for_base = None
            
            if collision_in_exec:
                print("collision in execution and with the help of dry_run_step function second try")
                break

            # this one is to avoid cases where we don't have anymore waypoints left for the dynamic obstacle exec
            # 20 is just an arbitrary number, can make it smaller as well
            if len(path_for_base1) < 20:
                # print("resetting the path for turtlebot since it almost got finished")
                path_for_base1 = path_for_base_original
            while_loop_counter+=1
        
        # if new_path_for_agent is not None or path_for_base is not None:
        # if collision1 or collision2:
        if collision2 ==0 or collision2 ==1 or collision1 or collision_in_exec:
            print("collision1 {} when collision2 is {}".format( collision1, collision2))
            print("collision_in_exec", collision_in_exec)
            collision_counter += 1
        
        else:
            if l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos) < min_dist:#0.01
                counter += 1
                print("increasing the counter and distance", l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos))
                # if while_loop_counter > number_of_short_intervals:
                #     short_interval_success_counter+=1
            else:
                print("the final distance between agent and goal", l2_distance(env.robots[0].get_end_effector_position(), env.task.target_pos))
        time_elapsed = time()-start_time
        total_time_taken += time_elapsed

        # print('counter, time elapsed, collision_counter, while_loop_counter, short_interval counter, total in the dynamic nav file', counter,
        #                                     time_elapsed, collision_counter, while_loop_counter, short_interval_success_counter, j+1)
        print('avg time {:.4f}, collision_counter {}, while_loop number {}, ITERATION {}, time elapsed {:.4f}, COUNTER {}'
                .format(total_time_taken/(j+1), collision_counter, while_loop_counter, j+1, time_elapsed, counter))

    print("total time taken for everything", total_time_taken)
    env.close()





if __name__ == "__main__":
    main()
