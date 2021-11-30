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
    # plan_joint_motion,
    set_base_values_with_z,
    get_base_distance_fn
    # set_joint_positions,
)

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import rotate_vector_2d
# from igibson.render.profiler import Profiler



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

def main():
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav.yaml")
    config_filename = os.path.join(igibson.example_config_path, "fetch_turtlebot_room_rearrangement.yaml")
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="headless")
    # env = iGibsonEnv(config_file=config_filename, mode="gui")
    
    
    env.reset()
    
    # this is for gibson RS
    goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
    path_for_base_obs, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
    reverse_path_from_b2a, coll_again_for_dynamic = plan_base_motion_dynamic(env, [-1.5, 0.75, 0], 0)

    #2, -1 is a good goal, starting from -1, -1 for ACKERMANVILLE
    # goal1 = [2, -1, 0]#-2.5, -2.5 is a good goal, starting from -1, -1 for ALOHA
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

    distance_fn = get_base_distance_fn([1, 1, 0])
    counter = 0
    collision_counter = 0
    short_interval_success_counter = 0
    new_path_for_agent = None
    total_time_taken = 0
    
    for j in range(5):#number of iterations to run
        # collision2 = False
        collision2 = 100
        collision_in_exec = False
        path_for_base1 = path_for_base_original

        tau_l = 10
        tau_s = 5
        lamb = 9/10.0
        exec_time = tau_s #(1-lamb)*
        plan_time = lamb*tau_s 
        t_cutoff = 25#* (1+lamb)
        number_of_short_intervals = (t_cutoff - tau_l)/(2*tau_s)
        # print("number_of_short_intervals", number_of_short_intervals)
        print("TIMEEEEEEEEEEEEEE, exec, plan, tau_s, tau_l, cutoff",exec_time, plan_time, tau_s, tau_l, t_cutoff)
        # env.reset()
        # for i in range(100):
        if j != 0:
            env.reset()
        
        start_time = time()
        # goal = env.task.target_pos #[-1, 1, 0]#
        # start = env.task.initial_pos
        # print("start in the main function", start)
        # path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
        ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)#tau_l
        # print("path for base1", path_for_base)

        
        if collision1:
            print("let's reset the environment in hope of a new and better start and goal")
            env.reset()
            # path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
            ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)
        
            if path_for_base is None:
                print("Oh, No! not able to find the path for the fetch agent even after a reset, what can i do man!")
            if collision1:
                print("no point of reset, in collision again")
                env.reset()
                # path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
                ref_rrg, path_for_base, collision1 = plan_base_motion_with_dynamic_rrg(env, env.task.target_pos, 0, maxtime=tau_l)
                if collision1:
                    print("this is it, no more reset, in collision again")

        if path_for_base is None:
            print("Oh, No! not able to find the path for the fetch agent")
        else:
            print("the length of reference path generated", len(path_for_base))
        # dry_run_base_plan(env, path_for_base, path_for_base1)
        # dry_run_base_plan(env, path_for_base, None)
        # dry_run_base_plan_timed(env, path_for_base, path_for_base1, 1.5)

        # updated_agent_path, path_for_base1 =  dry_run_base_plan(env, path_for_base, path_for_base1, tau_s/2)#tau_s/2
        
        # TODO need to do this in the plan_base_motion functions
        # env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)#since at the end of planning, if robot gets to goal, it would stay there even without executing the plan at all
        ref_path_time = time()
        time_taken_for_ref_path_calc = time()-start_time
        proper_cutoff_time = (t_cutoff - time_taken_for_ref_path_calc)* (1+lamb)
        print("distance between current and goal before reference run {}, and ref time {}".format(distance_fn(env.robots[0].get_position(), env.task.target_pos), time_taken_for_ref_path_calc))
        print("proper cutoff time {:.4f}, time {}".format(proper_cutoff_time, time()-ref_path_time))
        # print("current position of turtlebot after reference traj applied ", env.task.dynamic_objects[0].get_position())
        while_loop_counter = 0
        # while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and while_loop_counter < 2:
        while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and time()- ref_path_time < proper_cutoff_time and ref_rrg is not None:
        # while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and time()-start_time < t_cutoff and ref_rrg is not None:
            # print("in the while loop number", while_loop_counter+1)
            # print("distance between current and goal in the while loop number", distance_fn(env.robots[0].get_position(), env.task.target_pos),
                                                                                #  while_loop_counter+1)
            # env.task.initial_pos = env.robots[0].get_position()

            # path_for_base is the reference trajectory
            reference_trajectory_after_update_call, next_traj, collision2 = plan_base_motions_update(env, ref_rrg, path_for_base, env.task.target_pos, 0, maxtime=plan_time)
            if collision2 == 0:
                print("breaking out of the while loop since there was collision with start involved")
                break

            # if new_path_for_agent is None:
            #     print("Oh, No! not able to find the path for the fetch agent in the second try")
            
            # dry_run_base_plan_timed(env, new_path_for_agent, path_for_base1, 1)#shorten this 
            # updated_agent_path, path_for_base1 =  dry_run_base_plan(env, new_path_for_agent, path_for_base1, tau_s/2)#tau_s/2
            # path_for_base, path_for_base1 =  dry_run_base_plan(env, new_path_for_agent, path_for_base1, exec_time)#tau_s/2
            # reference_traj_left_to_execute, path_for_base1, execution_counter_fetch = dry_run_base_plan(env, reference_trajectory_after_update_call, path_for_base1, exec_time)#tau_s/2
            reference_traj_left_to_execute, path_for_base1, execution_counter_fetch, collision_in_exec = dry_run_base_plan_coll_break(env, reference_trajectory_after_update_call, path_for_base1, exec_time)#tau_s/2

            if distance_fn(env.robots[0].get_position(), env.task.target_pos) < 0.01 :
                print("breaking after the first execution in the while loop")
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


            reference_trajectory_after_update_call2, next_traj_2, collision2 = plan_base_motions_update(env, ref_rrg, path_for_base, env.task.target_pos, 0, maxtime=plan_time)
            if collision2 == 0:
                print("breaking out of the while loop since there was collision with start involved second try")
                break
            # reference_traj_left_to_execute2, path_for_base1, execution_counter_fetch2 = dry_run_base_plan(env, path_for_base, path_for_base1, exec_time)#tau_s/2
            reference_traj_left_to_execute2, path_for_base1, execution_counter_fetch2, collision_in_exec = dry_run_base_plan_coll_break(env, path_for_base, path_for_base1, exec_time)#tau_s/2
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
            if distance_fn(env.robots[0].get_position(), env.task.target_pos) < 0.01:
                counter += 1
                # if while_loop_counter > number_of_short_intervals:
                #     short_interval_success_counter+=1
            else:
                print("the final distance between agent and goal", distance_fn(env.robots[0].get_position(), env.task.target_pos))
        time_elapsed = time()-start_time
        total_time_taken += time_elapsed

        # print('counter, time elapsed, collision_counter, while_loop_counter, short_interval counter, total in the dynamic nav file', counter,
        #                                     time_elapsed, collision_counter, while_loop_counter, short_interval_success_counter, j+1)
        print('time elapsed {:.4f}, collision_counter {}, while_loop number {}, ITERATION {}, avg time {:.4f}, COUNTER {}'
                .format(time_elapsed, collision_counter, while_loop_counter, j+1, total_time_taken/(j+1), counter))

    print("total time taken for everything", total_time_taken)
    env.close()


if __name__ == "__main__":
    main()
