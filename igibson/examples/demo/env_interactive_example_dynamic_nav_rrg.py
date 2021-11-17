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
    # plan_joint_motion,
    set_base_values_with_z,
    # set_joint_positions,
)

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.utils.utils import rotate_vector_2d
# from igibson.render.profiler import Profiler



def plan_base_motions(env, goal, robot_number = 0):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        state = env.get_state()
        print("Start FETCH ", env.robots[robot_number].get_position())
        print("Goal Fetch", goal)
        # print("robot_ids for the 0 robot in plan base motion function", env.robots[robot_number].robot_ids)
        x, y, theta = goal
        grid = state["occupancy_grid"]


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
            algorithm="rrt",
            # optimize_iter=10,
        )

        return path, coll_or_not


def dry_run_base_plan(env, path, path1):
        """
        Dry run base motion plan by setting the base positions without physics simulation
        :param path: base waypoints or None if no plan can be found
        """
        if path is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            for way_point in path:
                # print("setting robot position with waypoints")
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )
                env.simulator.step()
                # time.sleep(0.005) # for animation
        else:
            # the next line is for non-gui modes
            # set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)
            print("path is not found")
        
        if path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            for way_point in path1:
                # print("setting robot position with waypoints 2")
                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )
                env.simulator.sync()
                # time.sleep(0.005) # for animation
        else:
            # the next line is for non-gui modes, only the final waypoint is being set 
            # set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)
            print("path 2 is not found")

        # this is for updating both of the robots at the same time and is much better
        # if path is not None and path1 is not None:
        #     # if self.mode in ["gui", "iggui", "pbgui"]:
        #     print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
        #     for way_point, way_point1 in zip(path, path1):
        #         # print("setting robot position with waypoints")
        #         set_base_values_with_z(
        #             env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
        #         )

        #         set_base_values_with_z(
        #             env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
        #         )
        #         env.simulator.step()#or sync
            
        #     # the next if else part is to accommodate the case where the number of waypoints are different in both paths,
        #     # and run the longer one after both ran together for the length of the shorter path
        #     if len(path) <= len(path1):
        #         for i in range (len(path), len(path1)):
        #             set_base_values_with_z( env.task.dynamic_objects[0].robot_ids[0], [path1[i][0], path1[i][1], path1[i][2]], z=env.initial_pos_z_offset
        #             )
        #             env.simulator.step()
        #     else:
        #         for j in range (len(path1), len(path)):
        #             set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
        #             )
        #             env.simulator.step()
        # else:
        #     print("path/path1 is not found")

def dry_run_base_plan_timed(env, path, path1, time_to_apply_the_path):
        """
        Dry run base motion plan by setting the base positions without physics simulation
        :param path: base waypoints or None if no plan can be found
        """
        t0 = time()
        # if path is not None:
        #     for way_point in path:
        #         set_base_values_with_z(
        #             env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
        #         )
        #         env.simulator.step()
        #         if time()- t0 > time_to_apply_the_path:
        #             print("time out for the dry run", time()- t0)
        #             # yield way_point
        #             break
        # else:
        #     print("path is not found")

        # if path1 is not None:
        #     for way_point in path1:
        #         set_base_values_with_z(
        #             env.task.dynamic_objects[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
        #         )
        #         env.simulator.sync()
        # else:
        #     print("path 2 is not found")
        
        # this is for updating both of the robots at the same time and is much better
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            print("length of paths, number of waypoints 1 and 2", len(path), len(path1))
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot position with waypoints")
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                env.simulator.step()#or sync
                
                if time()- t0 > time_to_apply_the_path:
                    print("time out for the dry run together", time()- t0)
                    print("waypoint for the agent", way_point)
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
                    
                    # maybe this part should be before the setbasevalues
                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of obstacle", time()- t0)
                        print("waypoint for the obstacle", path1[i])
                        return
            else:
                for j in range (len(path1), len(path)):
                    set_base_values_with_z( env.robots[0].robot_ids[0], [path[j][0], path[j][1], path[j][2]], z=env.initial_pos_z_offset
                    )
                    env.simulator.step()

                    if time()- t0 > time_to_apply_the_path:
                        print("time out for the dry run of agent alone", time()- t0)
                        print("waypoint for the agent", path[j])
                        return
        else:
            env.land(env.robots[0], env.task.initial_pos, env.task.initial_orn)
            print("path/path1 is not found, and setting the agent to its initial position")

def main():
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav.yaml")
    config_filename = os.path.join(igibson.example_config_path, "fetch_turtlebot_room_rearrangement.yaml")
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="headless")
    
    # env.reset()
    # goal = env.task.target_pos #[-1, 1, 0]#
    # start = env.task.initial_pos
    # # print("start in the main function", start)
    # path_for_base = plan_base_motions(env, goal, 0)
    # if path_for_base is None:
    #     print("Oh, No! not able to find the path for the fetch agent")

    # goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
    # path_for_base1 = plan_base_motion_dynamic(env, goal1, 0)
    
    # if path_for_base1 is None:
    #     print("Oh, No! not able to find the path for the dynamic turtlebot")
    # # dry_run_base_plan(env, path_for_base, path_for_base1)
    # # dry_run_base_plan(env, path_for_base, None)
    # dry_run_base_plan_timed(env, path_for_base, path_for_base1, 1)
    # new_path_for_agent = plan_base_motions(env, goal, 0)

    # dry_run_base_plan_timed(env, new_path_for_agent, path_for_base1, 5)
    
    # if new_path_for_agent is not None:
    #     counter += 1

    # print('counter in the dynamic nav file', counter)
    
    ######## the loop for the action step process and running it for some iterations #######
    # for _ in range(24000):  # at least 10 seconds
    #     p.stepSimulation()
    env.reset()
    goal1 = [1, 1, 0]#env.task.target_pos1 1, -1 works for dynamic one
    path_for_base1, collision_for_dynamic_obs = plan_base_motion_dynamic(env, goal1, 0)
    # print("path for base1", path_for_base1)

    if path_for_base1 is None:
        print("Oh, No! not able to find the path for the dynamic turtlebot")
    counter = 0
    collision_counter = 0
    new_path_for_agent = None
    for j in range(100):
        # env.reset()
        # for i in range(100):
        # # with Profiler("Environment action step"):
        #     action = env.action_space.sample()
        #     # action1 = env.action_space1.sample()
        #     state, reward, done, info = env.step(action)
        #     if done:
        #         logging.info("Episode finished after {} timesteps".format(i + 1))
        #         break
        
        if j != 0:
            env.reset()
        # goal = env.task.target_pos #[-1, 1, 0]#
        # start = env.task.initial_pos
        # print("start in the main function", start)
        path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0)
        # print("path for base1", path_for_base)

        if path_for_base is None:
            print("Oh, No! not able to find the path for the fetch agent")
        
        if collision1:
            print("let's reset the environment in hope of a new and better start and goal")
            env.reset()
            path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0)
        
            if path_for_base is None:
                print("Oh, No! not able to find the path for the fetch agent even after a reset, what can i do man!")
            if collision1:
                print("no point of reset, in collision again")
                env.reset()
                path_for_base, collision1 = plan_base_motions(env, env.task.target_pos, 0)
                if collision1:
                    print("this is it, no more reset, in collision again")

        # dry_run_base_plan(env, path_for_base, path_for_base1)
        # dry_run_base_plan(env, path_for_base, None)
        dry_run_base_plan_timed(env, path_for_base, path_for_base1, 2)

        new_path_for_agent, collision2 = plan_base_motions(env, env.task.target_pos, 0)

        if new_path_for_agent is None:
            print("Oh, No! not able to find the path for the fetch agent in the second try")
        
        dry_run_base_plan_timed(env, new_path_for_agent, path_for_base1, 5)
        
        if new_path_for_agent is not None or path_for_base is not None:
            counter += 1
        if collision1 and collision2:
            collision_counter += 1

        print('counter, collision_counter, total in the dynamic nav file', counter, collision_counter, j+1)

    env.close()


if __name__ == "__main__":
    main()
