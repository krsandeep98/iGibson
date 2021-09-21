import logging
import os
import time

import sys
logging.disable(sys.maxsize)

import numpy as np
import pybullet as p
# import matplotlib.pyplot as plt
# from time import time
import multiprocessing
from multiprocessing import Pool

from igibson.external.pybullet_tools.utils import (
    plan_base_motion_2d,
    plan_base_motion,
    # plan_joint_motion,
    set_base_values_with_z,
    get_base_distance_fn
    # set_joint_positions,
)

import igibson
from igibson.envs.igibson_env_parallel_ray import iGibsonEnv
from igibson.utils.utils import rotate_vector_2d

import ray
ray.init(num_cpus = 4)

@ray.remote
class iGibsonRayEnv(iGibsonEnv):
    # def sample_action_space(self):
    #     return self.action_space.sample()
    def multiproc_dry_run_obstacle_ray(self, path):
        return self.multiproc_dry_run_obstacle(path)
    def multiproc_dry_run_agent_ray(self, path):
        return self.multiproc_dry_run_agent(path)
# from igibson.render.profiler import Profiler



def plan_base_motions(env, goal, robot_number = 0, maxtime=10):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        # state = env.get_state()
        print("Start FETCH ", env.robots[robot_number].get_position())
        print("Goal Fetch", goal)
        print("current position of turtlebot in agent plan basemotion ", env.task.dynamic_objects[robot_number].get_position())
        # print("robot_ids for the 0 robot in plan base motion function", env.robots[robot_number].robot_ids)
        x, y, theta = goal
        # grid = state["occupancy_grid"]


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
        mp_obstacles.append(env.task.dynamic_objects[0].robot_ids[0])
        # mp_obstacles.extend(env.scene._load())# for the static scenes
        # mp_obstacles.extend(env.scene.get_body_ids()) #for the interactive ones
        # print('mp_obstacles in plan base motions function', mp_obstacles)

        path, coll_or_not = plan_base_motion(
            # env.robots[robot_number].robot_ids[0],
            env.robots[robot_number],
            env,
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # map_2d=grid,
            # occupancy_range= occupancy_range,
            # grid_resolution=env.grid_resolution,
            # robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
            resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=mp_obstacles,
            algorithm="rrg",
            max_time=maxtime
            # optimize_iter=10,
        )

        return path, coll_or_not

def plan_base_motion_dynamic(env, goal, robot_number = 0):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        # state = env.get_state()
        print("start dynamic obstacle", (env.task.dynamic_objects[robot_number].get_position()))
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
            env,
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

# @ray.remote
def multiproc_dry_run_agent(env, path):
        t0 = time()
        if path is not None :
            # if self.mode in ["gui", "iggui", "pbgui"]:
            print("length of paths, number of waypoints 1 and 2", len(path))
            waypoint_counter_fetch = 0
            # waypoint_counter_turtlebot = 0
            for way_point in (path):
                # print("setting agent position with waypoints", way_point, waypoint_counter_fetch)
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                waypoint_counter_fetch+=1
                env.simulator.step()#or sync
                if time() - t0 > 2:
                    print("timeout in the agent dry run function", waypoint_counter_fetch)
                    break
                # set_base_values_with_z(
                #     env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                # )
                # waypoint_counter_turtlebot+=1

# @ray.remote
def multiproc_dry_run_obstacle(env, path):
        t0 = time()
        if path is not None :
            # if self.mode in ["gui", "iggui", "pbgui"]:
            print("length of path obstacle", len(path))
            # waypoint_counter_fetch = 0
            waypoint_counter_turtlebot = 0
            for way_point in (path):
                # print("setting robot position with waypoints", wa)
                # print("setting obstacle position with waypoints", way_point, waypoint_counter_turtlebot)
                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                waypoint_counter_turtlebot+=1
                env.simulator.step()#or sync
                if time() - t0 > 2:
                    print("timeout in the obs dry run function", waypoint_counter_turtlebot)
                    break

def dry_run_base_plan(env, path, path1, time_to_apply_the_path):
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
                    return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:]
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
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:]
            else:
                for j in range (len(path1), len(path)):
                    set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    )
                    env.simulator.step()
                    waypoint_counter_fetch+=1

                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of agent alone", time()- t0)
                        print("waypoint for the agent, waypoint counter", path[j], waypoint_counter_fetch)
                        return path[waypoint_counter_fetch:], path1[waypoint_counter_turtlebot:]
        else:
            env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            print("path/path1 is not found, and setting the agent to its initial position")

      

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

# def main():
#     # config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav.yaml")
#     config_filename = os.path.join(igibson.example_config_path, "fetch_turtlebot_room_rearrangement.yaml")
#     # config_filename = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
#     env = iGibsonEnv(config_file=config_filename, mode="headless")
#     env.reset()


#     goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
#     path_for_base1, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
#     reverse_path_from_b2a, coll_again_for_dynamic = plan_base_motion_dynamic(env, [-1.5, 0.75, 0], 0)
#     # reverse_path_from_b2a = path_for_base1[::-1]
#     path_for_base1 = path_for_base1 + reverse_path_from_b2a
#     # print("path for base1 last", path_for_base1)
#     # print("path for base1 last reversed", reverse_path_from_b2a)
#     if path_for_base1 is None:
#         print("Oh, No! not able to find the path for the dynamic turtlebot")
#     env.reset()#added this to handle the case where the turtlebot was at 0.55, 0.95, but that was probably an rrt issue


#     distance_fn = get_base_distance_fn([1, 1, 0])
#     counter = 0
#     collision_counter = 0
#     short_interval_success_counter = 0
#     new_path_for_agent = None

#     for j in range(1):
#         # env.reset()
#         # for i in range(100):
        
#         if j != 0:
#             env.reset()
#         # goal = env.task.target_pos #[-1, 1, 0]#
#         # start = env.task.initial_pos
#         # print("start in the main function", start)
#         path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
#         # print("path for base1", path_for_base)

#         if path_for_base is None:
#             print("Oh, No! not able to find the path for the fetch agent")
        
#         if collision1:
#             print("let's reset the environment in hope of a new and better start and goal")
#             env.reset()
#             path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
        
#             if path_for_base is None:
#                 print("Oh, No! not able to find the path for the fetch agent even after a reset, what can i do man!")
#             if collision1:
#                 print("no point of reset, in collision again")
#                 env.reset()
#                 path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
#                 if collision1:
#                     print("this is it, no more reset, in collision again")

#         # dry_run_base_plan(env, path_for_base, path_for_base1)
#         # dry_run_base_plan(env, path_for_base, None)
#         # dry_run_base_plan_timed(env, path_for_base, path_for_base1, 1.5)
        
#         processes = []
#         # for i in range(0,2):
#         p1 = multiprocessing.Process(target= multiproc_dry_run_agent, args=(env, path_for_base,))
#         processes.append(p1)
#         p1.start()

#         p2 = multiprocessing.Process(target= multiproc_dry_run_obstacle, args=(env, path_for_base1,))
#         processes.append(p2)
#         p2.start()
        
#         for process in processes:
#             process.join()
#         print("current position of robot after reference traj applied ", env.robots[0].get_position())
#         print("current position of turtlebot after reference traj applied ", env.task.dynamic_objects[0].get_position())

#         # p1.terminate()
#         # p2.terminate()



#         # updated_agent_path, path_for_base1 =  dry_run_base_plan(env, path_for_base, path_for_base1, 0.5)

#         # print("distance between current and goal after reference run", distance_fn(env.robots[0].get_position(), env.task.target_pos))
#         # print("current position of turtlebot after reference traj applied ", env.task.dynamic_objects[0].get_position())
#         # while_loop_counter = 0
#         # while distance_fn(env.robots[0].get_position(), env.task.target_pos) >= 0.01 and while_loop_counter < 2:
#         #     # print("in the while loop number", while_loop_counter+1)
#         #     print("distance between current and goal in the while loop number", distance_fn(env.robots[0].get_position(), env.task.target_pos), while_loop_counter+1)
#         #     env.task.initial_pos = env.robots[0].get_position()
#         #     new_path_for_agent, collision2 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)

#         #     if new_path_for_agent is None:
#         #         print("Oh, No! not able to find the path for the fetch agent in the second try")
            
#         #     # dry_run_base_plan_timed(env, new_path_for_agent, path_for_base1, 1)#shorten this 
#         #     updated_agent_path, path_for_base1 =  dry_run_base_plan(env, new_path_for_agent, path_for_base1, 1.5)
#         #     while_loop_counter+=1
        
#         # # if new_path_for_agent is not None or path_for_base is not None:
#         # if distance_fn(env.robots[0].get_position(), env.task.target_pos) < 0.01:
#         #     counter += 1
#         #     if while_loop_counter > 0:
#         #         short_interval_success_counter+=1
#         # if collision1 and collision2:
#         #     collision_counter += 1

#         # print('counter, collision_counter, while_loop_counter, short_interval counter, total in the dynamic nav file', counter,
#         #     collision_counter, while_loop_counter, short_interval_success_counter, j+1)

#     env.close()


if __name__ == "__main__":
    # main()
    config_filename = os.path.join(igibson.example_config_path, "fetch_turtlebot_room_rearrangement.yaml")
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
    # env = iGibsonEnv(config_file=config_filename, mode="headless")

    env=iGibsonRayEnv.remote(config_file=config_filename, mode="headless")
    env.reset.remote()


    goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
    # path_for_base1, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
    # path_for_base1, collision_for_dynamic_obs = ray.get(env.plan_base_motion_dynamic.remote(goal1))
    path_for_base1, collision_for_dynamic_obs = ray.get(env.plan_base_motion_dynamic.remote(goal1))
    # reverse_path_from_b2a, coll_again_for_dynamic = plan_base_motion_dynamic(env, [-1.5, 0.75, 0], 0)
    path_for_base1 = path_for_base1 #+ reverse_path_from_b2a
    
    if path_for_base1 is None:
        print("Oh, No! not able to find the path for the dynamic turtlebot")
    env.reset.remote()#added this to handle the case where the turtlebot was at 0.55, 0.95, but that was probably an rrt issue


    distance_fn = get_base_distance_fn([1, 1, 0])
    counter = 0
    collision_counter = 0
    short_interval_success_counter = 0
    new_path_for_agent = None

    for j in range(1):        
        if j != 0:
            env.reset()
        # path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
        path_for_base, collision1 = ray.get(env.plan_base_motions.remote([-0.5, 0.5, 0]))
        env.reset.remote()

        # print("current position of ROBOT before applied ", env.robots[0].get_position())
        # print("current position of turtlebot before applied ", env.task.dynamic_objects[0].get_position())
        # print("env type",type(env))

        # print(path_for_base)
        # print(path_for_base1)

        if path_for_base is None:
            print("Oh, No! not able to find the path for the fetch agent")
        
        if collision1:
            print("let's reset the environment in hope of a new and better start and goal")
            env.reset()
            path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
        
            if path_for_base is None:
                print("Oh, No! not able to find the path for the fetch agent even after a reset, what can i do man!")
            if collision1:
                print("no point of reset, in collision again")
                env.reset()
                path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0, maxtime=10)
                if collision1:
                    print("this is it, no more reset, in collision again")
        ray.get(env.get_current_position.remote())
        
        start_time = time.time()
        processes = []
        # # for i in range(0,2):
        # p1 = multiprocessing.Process(target= multiproc_dry_run_agent, args=(env, path_for_base,))
        # p1 = multiprocessing.Process(target= env.multiproc_dry_run_agent_ray.remote, args=(path_for_base,))
        # processes.append(p1)
        # p1.start()

        # # p2 = multiprocessing.Process(target= multiproc_dry_run_obstacle, args=(env, path_for_base1,))
        # p2 = multiprocessing.Process(target= env.multiproc_dry_run_obstacle_ray.remote, args=(path_for_base1,))
        # processes.append(p2)
        # p2.start()
        
        # for process in processes:
        #     process.join(2.5)

        # with Pool(processes=4) as pool:         # start 4 worker processes
        #     result = pool.apply(multiproc_dry_run_agent, (env, path_for_base))
            # result.get(timeout=3)
        p1 = env.multiproc_dry_run_agent_ray.remote(path_for_base)
        processes.append(p1)
        p2 = env.multiproc_dry_run_obstacle_ray.remote(path_for_base1)
        processes.append(p2)

        # ray.get(p1)
        # ray.get(p2)

        ray.get(processes)

        print("time taken to run the executions in main function", time.time()-start_time)

        ray.get(env.get_current_position.remote())

        # multiproc_dry_run_agent(env, path_for_base)
        # multiproc_dry_run_obstacle(env, path_for_base1)

        # print("current position of robot after reference traj applied ", env.robots[0].get_position())
        # print("current position of turtlebot after reference traj applied ", env.task.dynamic_objects[0].get_position())

    env.close.remote()


