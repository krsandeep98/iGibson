import logging
import os

import sys
logging.disable(sys.maxsize)

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.profiler import Profiler


def main():
    config_filename = os.path.join(igibson.example_config_path, "locobot_point_nav.yaml")
    config_filename2 = os.path.join(igibson.example_config_path, "turtlebot_point_nav.yaml")
    env = iGibsonEnv(config_file=config_filename, mode="gui")
    env2 = iGibsonEnv(config_file=config_filename2, mode="headless")
    for j in range(10):
        env2.reset()
        env.reset()
        for i in range(10):
            with Profiler("Environment action step"):
                action = env.action_space.sample()
                state, reward, done, info = env.step(action)

                action2 = env2.action_space.sample()
                state2, reward2, done2, info2 = env2.step(action2)
                if done:
                    logging.info("Episode1 finished after {} timesteps".format(i + 1))
                    break
                # if done2:
                #     logging.info("Episode2 finished after {} timesteps".format(i + 1))
                #     break
    env.close()
    # env2.close()


if __name__ == "__main__":
    main()
