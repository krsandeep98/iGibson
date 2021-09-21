from random import random
from time import time
import numpy as np
from .utils import INF, argmin


class OptimalNode(object):

    def __init__(self, config, parent=None, d=0, path=[], iteration=None):
        self.config = config
        self.parent = parent
        self.children = set()
        self.d = d
        self.path = path
        if parent is not None:
            self.cost = parent.cost + d
            self.parent.children.add(self)
        else:
            self.cost = d
        self.solution = False
        self.creation = iteration
        self.last_rewire = iteration
        self.valid = True

    def set_solution(self, solution):
        if self.solution is solution:
            return
        self.solution = solution
        if self.parent is not None:
            self.parent.set_solution(solution)

    def retrace(self):
        if self.parent is None:
            return self.path + [self.config]
        return self.parent.retrace() + self.path + [self.config]

    def rewire(self, parent, d, path, iteration=None):
        if self.solution:
            self.parent.set_solution(False)
        self.parent.children.remove(self)
        self.parent = parent
        self.parent.children.add(self)
        if self.solution:
            self.parent.set_solution(True)
        self.d = d
        self.path = path
        self.update()
        self.last_rewire = iteration

    def update(self):
        self.cost = self.parent.cost + self.d
        for n in self.children:
            n.update()

    def clear(self):
        self.node_handle = None
        self.edge_handle = None

    def draw(self, env):
        from manipulation.primitives.display import draw_node, draw_edge
        color = (0, 0, 1, .5) if self.solution else (1, 0, 0, .5)
        self.node_handle = draw_node(env, self.config, color=color)
        if self.parent is not None:
            self.edge_handle = draw_edge(
                env, self.config, self.parent.config, color=color)

    def __str__(self):
        return self.__class__.__name__ + '(' + str(self.config) + ')'
    __repr__ = __str__


def safe_path(sequence, collision):
    path = []
    for q in sequence:
        if collision(q):
            break
        path.append(q)
    return path

def safe_path_update(sequence, collision):
    path = []
    safety_counter = len(sequence)
    # print("type of sequence in safe path update", type(sequence))
    # print("the sequence", sequence)
    for count, q in enumerate(sequence):
        if collision(q):
            # break
            print("config in collision in safe path update fn, which one and length", q, count, len(sequence))
            # q.valid = False
            if safety_counter == len(sequence):
                safety_counter = count
            break
        # else:
            # q.valid = True
            # path.append(q)
    return sequence[:safety_counter], safety_counter


class rrg_dynamic(object):

    def __init__(self, goal, distance, sample, extend, collision, collision_d):
        
        self.nodes = []
        self.goal = goal
        self.distance = distance
        self.sample = sample
        self.extend = extend
        self.collision_static = collision
        self.collision_dynamic = collision_d
        self.reference_traj = None


    # def rrg(start, goal, distance, sample, extend, collision, radius=0.5, max_time=INF, max_iterations=INF, goal_probability=.2, informed=True):
    def rrg(self, start, radius=0.5, max_time=INF, max_iterations=INF, goal_probability=.2, informed=True):
        
        # nodes = [OptimalNode(start)]
        self.nodes.append(OptimalNode(start))
        goal_n = None
        t0 = time()# max time is in secs, for the ref one we should give around 50seconds maybe
        it = 0

        while (time() - t0) < max_time and it < max_iterations:
            do_goal = goal_n is None and (it == 0 or random() < goal_probability)
            s = self.goal if do_goal else self.sample()
            # Informed RRT*
            if informed and goal_n is not None and self.distance(start, s) + self.distance(s, self.goal) >= goal_n.cost:
                continue
            if it % 100 == 0:
                print(it, time() - t0, goal_n is not None, do_goal, (goal_n.cost if goal_n is not None else INF))
            it += 1
            # print(it, len(nodes))

            nearest = argmin(lambda n: self.distance(n.config, s), self.nodes)
            path = safe_path(self.extend(nearest.config, s), self.collision_static)
            if len(path) == 0:
                continue
            new = OptimalNode(path[-1], parent=nearest, d=self.distance(
                nearest.config, path[-1]), path=path[:-1], iteration=it)
            
            # if do_goal:# and distance(new.config, goal) < 1e-6:
            # this distance of 0.01 is different from the one which is being used to detect if goal has been reached in the main file
            # that is slightly smaller and if the following is satisfied then, the main function would obviously be true
            if do_goal and self.distance(new.config, self.goal) < 1e-2:
                print("YAY!! distance between start and goal in rrg call, and size of tree", self.distance(start, self.goal), len(self.nodes))
                print("goal_n is being set in rrg call", it , time()- t0)
                goal_n = new
                goal_n.set_solution(True)
                self.reference_traj = goal_n.retrace()
                break
            # TODO - k-nearest neighbor version
            # neighbors = filter(lambda n: distance(
            #    n.config, new.config) < radius, nodes)
            # print('num neighbors', len(list(neighbors)))
            k = 10
            k = np.min([k, len(self.nodes)])
            dists = [self.distance(n.config, new.config) for n in self.nodes]
            neighbors = [self.nodes[i] for i in np.argsort(dists)[:k]]
            #print(neighbors)

            self.nodes.append(new)

            for n in neighbors:
                d = self.distance(n.config, new.config)
                if n.cost + d < new.cost:#this is not needed in the rrg step as we add all the neighbors to the new node
                    path = safe_path(self.extend(n.config, new.config), self.collision_static)
                    if len(path) != 0 and self.distance(new.config, path[-1]) < 1e-2:
                        new.rewire(n, d, path[:-1], iteration=it)
            # for n in neighbors:  # TODO - avoid repeating work
            #     d = distance(new.config, n.config)
            #     if new.cost + d < n.cost:
            #         path = safe_path(extend(new.config, n.config), collision)
            #         if len(path) != 0 and distance(n.config, path[-1]) < 1e-2:
            #             n.rewire(new, d, path[:-1], iteration=it)
        if goal_n is None:
            print("NO!!! distance between start and goal in rrg call and length", self.distance(start, self.goal), len(self.nodes))
            print("goal_n is none in rrg call", it, time()- t0)
        #     return None, False
        
        # return goal_n.retrace(), False



    def rrg_update(self, start, ref_path, max_time=INF, max_iterations=INF, goal_probability=.2, informed=True):
        
        # have some check for the ref path being none and if that's the case then make it return something
        # or just find more nodes and sample and get a path in the available time
        # collision_var = True
        
        # this also returns false if we detect collision in start or goal in the new call for rrg short 
        # and we just return true for the time being when we had something for the ref path, 
        # TODO need to change that according to how safe the ref path actually is
        if ref_path is None:
            print("calling the short rrg in the update fn")
            collision_var = self.rrg_short(start, self.collision_dynamic, max_time, max_iterations)
            # if not collision_var :
            #     return False
            # else:
            #     return True
            return collision_var

        else:
            # the safe path function has to output something to show that the initial config was in collision and we then end the iteration there according to that variable
            # updated_path = safe_path(ref_path, self.collision_dynamic)
            updated_path, safety_counter = safe_path_update(ref_path, self.collision_dynamic)
            # need to see how much this path has changed and act accordingly if the last waypoint is still the goal, then it's fine
            # otherwise we want to get a new path which goes through the valid nodes only
            
            # updated_path is None was never going in the loop even when there was nothing in the path
            
            # this is for init config in collision, TODO maybe replan, but since its already in coll, can't really do much
            # got to abort the iteration at this step
            if safety_counter == 0:
                print ("the reference path and more importantly initial config is in collision in update function")
                # return False
                return 0
            
            # this loop has to plan for the waypoints (valid) to goal if the goal is still valid
            else:
                self.reference_traj = updated_path
                print ("setting the reference path in update function to later one only")
                # return True
                return 4
    
    def rrg_short(self, start, collision, max_time=INF, max_iterations=INF, goal_probability=.2, informed=True):
        
        # this returns true if there is no collision detected for the start and goal and we are either able to find the path or not
        # it would return false if there is collision in start or goal according to the dynamic obstacles

        # nodes = [OptimalNode(start)]
        # self.nodes.append(OptimalNode(start))
        if collision(self.goal):
            print("goal is in collision in rrg short function")
            # return False
            return 1
        if collision(start):
            print("start is in collision in rrg short function")
            # return False
            return 0
        goal_n = None
        t0 = time()# max time is in secs, for the ref one we should give around 50seconds maybe
        it = 0

        while (time() - t0) < max_time and it < max_iterations:
            do_goal = goal_n is None and (random() < goal_probability)#removed the it=0 case as it's not really the first iteration since we are starting from the ref roadmap
            s = self.goal if do_goal else self.sample()
            # Informed RRT*
            if informed and goal_n is not None and self.distance(start, s) + self.distance(s, self.goal) >= goal_n.cost:
                continue
            # if it % 100 == 0:
            #     print(it, time() - t0, goal_n is not None, do_goal, (goal_n.cost if goal_n is not None else INF))
            it += 1
            # print(it, len(nodes))

            nearest = argmin(lambda n: self.distance(n.config, s), self.nodes)
            path = safe_path(self.extend(nearest.config, s), collision)
            if len(path) == 0:
                continue
            new = OptimalNode(path[-1], parent=nearest, d=self.distance(
                nearest.config, path[-1]), path=path[:-1], iteration=it)
            
            # if do_goal:# and distance(new.config, goal) < 1e-6:
            # this distance of 0.01 is different from the one which is being used to detect if goal has been reached in the main file
            # that is slightly smaller and if the following is satisfied then, the main function would obviously be true
            if do_goal and self.distance(new.config, self.goal) < 1e-2:
                print("YAY!! distance between start and goal in update short rrg call", self.distance(start, self.goal), len(self.nodes))
                print("goal_n is being set in short rrg call, time", it, time()- t0)
                goal_n = new
                goal_n.set_solution(True)
                self.reference_traj = goal_n.retrace()
                # break
                # return True
                return 4
            # TODO - k-nearest neighbor version
            # neighbors = filter(lambda n: distance(
            #    n.config, new.config) < radius, nodes)
            # print('num neighbors', len(list(neighbors)))
            
            #this k can be reduced to take care of the short time 
            k = 5#10
            k = np.min([k, len(self.nodes)])
            dists = [self.distance(n.config, new.config) for n in self.nodes]
            neighbors = [self.nodes[i] for i in np.argsort(dists)[:k]]
            #print(neighbors)

            self.nodes.append(new)

            for n in neighbors:
                d = self.distance(n.config, new.config)
                if n.cost + d < new.cost:#this is not needed in the rrg step as we add all the neighbors to the new node
                    path = safe_path(self.extend(n.config, new.config), collision)
                    if len(path) != 0 and self.distance(new.config, path[-1]) < 1e-2:
                        new.rewire(n, d, path[:-1], iteration=it)
            # for n in neighbors:  # TODO - avoid repeating work
            #     d = distance(new.config, n.config)
            #     if new.cost + d < n.cost:
            #         path = safe_path(extend(new.config, n.config), collision)
            #         if len(path) != 0 and distance(n.config, path[-1]) < 1e-2:
            #             n.rewire(new, d, path[:-1], iteration=it)
        if goal_n is None:
            # print("NO!!! distance between start and goal in the update rrg short call and length", self.distance(start, self.goal), len(self.nodes))
            # print("goal_n is none in rrg short call", it, time()- t0)
            # return True
            return 2
  