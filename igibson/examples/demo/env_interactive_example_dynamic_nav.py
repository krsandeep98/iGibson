import logging
import os
# import time

import sys
logging.disable(sys.maxsize)

import numpy as np
import pybullet as p

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

# def plan_base_motion(env, goal, robot_number = 0):
#         """
#         Plan base motion given a base subgoal
#         :param goal: base subgoal
#         :return: waypoints or None if no plan can be found
#         """
#         # if env.marker is not None:
#         #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

#         state = env.get_state()
#         print("goal in the plan base motion function", goal)
#         print("start in the plan base motion function", env.robots[robot_number].get_position())
#         # print("robot_ids for the 0 robot in plan base motion function", env.robots[robot_number].robot_ids)
#         x, y, theta = goal
#         grid = state["occupancy_grid"]
#         # print("grid in the plan base motion function", grid.shape)#(120, 120, 1)

#         yaw = env.robots[robot_number].get_rpy()[2]
#         occupancy_range = env.sensors["scan_occ"].occupancy_range
#         half_occupancy_range = occupancy_range / 2.0
#         robot_position_xy = env.robots[robot_number].get_position()[:2]
#         corners = [
#             robot_position_xy + rotate_vector_2d(local_corner, -yaw)
#             for local_corner in [
#                 np.array([half_occupancy_range, half_occupancy_range]),
#                 np.array([half_occupancy_range, -half_occupancy_range]),
#                 np.array([-half_occupancy_range, half_occupancy_range]),
#                 np.array([-half_occupancy_range, -half_occupancy_range]),
#             ]
#         ]
#         # env.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
#         path = plan_base_motion_2d(
#             env.robots[robot_number].robot_ids[0],
#             [x, y, theta],
#             (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
#             map_2d=grid,
#             occupancy_range= occupancy_range,
#             grid_resolution=env.grid_resolution,
#             robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
#             resolutions=np.array([0.05, 0.05, 0.05]),
#             obstacles=[],
#             algorithm="rrt",
#             optimize_iter=10,
#         )

#         return path

def plan_base_motions(env, goal, robot_number = 0):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        state = env.get_state()
        print("goal in the plan base motion function", goal)
        print("start in the plan base motion function", env.robots[robot_number].get_position())
        # print("robot_ids for the 0 robot in plan base motion function", env.robots[robot_number].robot_ids)
        x, y, theta = goal
        grid = state["occupancy_grid"]
        # print("grid in the plan base motion function", grid.shape)#(120, 120, 1)

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
        # env.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        # path = plan_base_motion_2d and plan_base_motion(
        path = plan_base_motion(
            env.robots[robot_number].robot_ids[0],
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # map_2d=grid,
            # occupancy_range= occupancy_range,
            # grid_resolution=env.grid_resolution,
            # robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
            resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=[],
            # algorithm="rrtstar",
            # optimize_iter=10,
        )

        return path

def plan_base_motion_dynamic(env, goal, robot_number = 0):
        """
        Plan base motion given a base subgoal
        :param goal: base subgoal
        :return: waypoints or None if no plan can be found
        """
        # if env.marker is not None:
        #     env.set_marker_position_yaw([goal[0], goal[1], 0.05], goal[2])

        state = env.get_state()
        print("goal in the plan base motion function", goal)
        print("start in the plan base motion function", env.task.dynamic_objects[robot_number].get_position())
        # print("robot_ids for the 0 robot in plan base motion function", env.task.dynamic_objects[robot_number].robot_ids)
        x, y, theta = goal
        grid = state["occupancy_grid"]
        print("grid in the plan base motion function", grid.shape)#(120, 120, 1)

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
        # env.base_mp_resolutions = np.array([0.05, 0.05, 0.05])
        path = plan_base_motion(
            env.task.dynamic_objects[robot_number].robot_ids[0],
            [x, y, theta],
            (tuple(np.min(corners, axis=0)), tuple(np.max(corners, axis=0))),
            # map_2d=grid,
            # occupancy_range= occupancy_range,
            # grid_resolution=env.grid_resolution,
            # robot_footprint_radius_in_map=env.sensors["scan_occ"].robot_footprint_radius_in_map,
            resolutions=np.array([0.05, 0.05, 0.05]),
            obstacles=[],
            # algorithm="rrt",
            # optimize_iter=10,
        )

        return path


def dry_run_base_plan(env, path, path1):
        """
        Dry run base motion plan by setting the base positions without physics simulation
        :param path: base waypoints or None if no plan can be found
        """
        # if path is not None:
        #     # if self.mode in ["gui", "iggui", "pbgui"]:
        #     for way_point in path:
        #         print("setting robot position with waypoints")
        #         set_base_values_with_z(
        #             env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
        #         )
        #         env.simulator.sync()
        #         # time.sleep(0.005) # for animation
        # else:
        #     # the next line is for non-gui modes
        #     # set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)
        #     print("path is not found")
        
        # if path1 is not None:
        #     # if self.mode in ["gui", "iggui", "pbgui"]:
        #     for way_point in path1:
        #         print("setting robot position with waypoints 2")
        #         set_base_values_with_z(
        #             env.task.dynamic_objects[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
        #         )
        #         env.simulator.sync()
        #         # time.sleep(0.005) # for animation
        # else:
        #     # the next line is for non-gui modes
        #     # set_base_values_with_z(self.robot_id, [path[-1][0], path[-1][1], path[-1][2]], z=self.initial_height)
        #     print("path 2 is not found")

        # this is for updating both of the robots at the same time and is much better
        if path is not None and path1 is not None:
            # if self.mode in ["gui", "iggui", "pbgui"]:
            for way_point, way_point1 in zip(path, path1):
                # print("setting robot position with waypoints")
                set_base_values_with_z(
                    env.robots[0].robot_ids[0], [way_point[0], way_point[1], way_point[2]], z=env.initial_pos_z_offset
                )

                set_base_values_with_z(
                    env.task.dynamic_objects[0].robot_ids[0], [way_point1[0], way_point1[1], way_point1[2]], z=env.initial_pos_z_offset
                )
                env.simulator.sync()
                # time.sleep(0.005) # for animation
        else:
            print("path/path1 is not found")


def main():
    config_filename = os.path.join(igibson.example_config_path, "turtlebot_dynamic_nav.yaml")
    # config_filename = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="gui")
    
    env.reset()
    goal = env.task.target_pos
    start = env.task.initial_pos
    print("start in the main function", start)
    path_for_base = plan_base_motions(env, goal, 0)

    goal1 = [1, -5, 0]#env.task.target_pos1 1, -1 works for dynamic one
    path_for_base1 = plan_base_motion_dynamic(env, goal1, 0)
    
    dry_run_base_plan(env, path_for_base, path_for_base1)
    # dry_run_base_plan(env, path_for_base, None)
    
    
    ######## the loop for the action step process and running it for some iterations #######
    # for _ in range(24000):  # at least 10 seconds
    #     p.stepSimulation()
    # for j in range(1):
    #     env.reset()
    #     for i in range(100):
    #     # with Profiler("Environment action step"):
    #         action = env.action_space.sample()
    #         # action1 = env.action_space1.sample()
    #         state, reward, done, info = env.step(action)
    #         if done:
    #             logging.info("Episode finished after {} timesteps".format(i + 1))
    #             break
    env.close()


if __name__ == "__main__":
    main()
