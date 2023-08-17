from importlib.resources import path
from gym_driving.assets.car import *
from gym_driving.envs.environment import *
from gym_driving.envs.driving_env import *
from gym_driving.assets.terrain import *

import time
import pygame, sys
from pygame.locals import *
import random
import math
import argparse
import numpy as np

# Do NOT change these values
TIMESTEPS = 1000
FPS = 30
NUM_EPISODES = 10

class Task1():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()


    def next_action(self, state):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED
        """
        x = state[0]
        y = state[1]
        v = state[2]
        angle = state[3]
        if angle > 180:
            angle = -(360-angle)
        # Replace with your implementation to determine actions to be taken
        if  angle - 180*math.atan((y - 0)/(x - 350))/math.pi >= 3:
            action_steer = 0
            action_acc = 2
        if angle - 180*math.atan((y - 0)/(x - 350))/math.pi <= -3:
            action_steer = 2
            action_acc = 2
        if angle - 180*math.atan((y - 0)/(x - 350))/math.pi < 3 and angle - 180*math.atan((y - 0)/(x - 350))/math.pi > -3:
            action_steer = 1
            action_acc = 4
        # if angle - 180*math.atan((y - 0)/(x - 350))/math.pi > 0 and angle - 180*math.atan((y - 0)/(x - 350))/math.pi< 3:
        #     action_steer = 1
        #     action_acc = 4

        # print(state)
        # action_steer = 1
        # action_acc = 4

        action = np.array([action_steer, action_acc])  

        return action

    def controller_task1(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
    
        ######### Do NOT modify these lines ##########
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        simulator = DrivingEnv('T1', render_mode=render_mode, config_filepath=config_filepath)

        time.sleep(3)
        ##############################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):
        
            ######### Do NOT modify these lines ##########
            
            # To keep track of the number of timesteps per epoch
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset()
            
            # Variable representing if you have reached the road
            road_status = False
            ##############################################

            # The following code is a basic example of the usage of the simulator
            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            # Writing the output at each episode to STDOUT
            print(str(road_status) + ' ' + str(cur_time))

class Task2():

    def __init__(self):
        """
        Can modify to include variables as required
        """

        super().__init__()

    def closest_pit(self, states, pits_centres):
        pit = math.sqrt((states[0]-pits_centres[0][0])**2 + (states[1]-pits_centres[0][1])**2)
        index_of_pit = 0
        for i in range(1,4):
            if math.sqrt((states[0]-pits_centres[i][0])**2 + (states[1]-pits_centres[i][1])**2) < pit:
                pit = math.sqrt((states[0]-pits_centres[i][0])**2 + (states[1]-pits_centres[i][1])**2)
                index_of_pit = i
        return index_of_pit

    def next_action(self, state, pits_centres):
        """
        Input: The current state
        Output: Action to be taken
        TO BE FILLED

        You can modify the function to take in extra arguments and return extra quantities apart from the ones specified if required
        """

        # Replace with your implementation to determine actions to be taken
        pit_index = self.closest_pit(state, pits_centres)
        x = state[0]
        y = state[1]
        v = state[2]
        angle = state[3]
        xpit = pits_centres[pit_index][0]
        ypit = pits_centres[pit_index][1]
        th = 1.5
        if angle > 180:
            angle = -(360-angle)

        if x < xpit + 75 and x > xpit - 75:#and y < ypit -50 and y > ypit + 50:
            while(angle > th or angle < -th):
                if angle > th:
                    action_steer = 0
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
                elif angle < -th:
                    action_steer = 2
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
            if angle > -th and angle < th:
                action_steer = 1
                action_acc = 3

        
        elif y < ypit + 75 and y > ypit - 75 and y > 0 and x < xpit - 50:
            while(angle + 90 > th or angle + 90 < -th):
                if angle + 90 > th:
                    action_steer = 0
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
                elif angle + 90 < -th:
                    action_steer = 2
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
            if angle + 90 > -th and angle + 90 < th:
                action_steer = 1
                action_acc = 3
        
        elif y < ypit + 75 and y > ypit - 75 and y < 0 and x < xpit - 50:
            while(angle - 90 > th or angle - 90 < -th):
                if angle - 90 > th:
                    action_steer = 0
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
                elif angle - 90 < -th:
                    action_steer = 2
                    action_acc = 0
                    action = np.array([action_steer, action_acc])  
                    return action
            if angle - 90 > -th and angle - 90 < th:
                action_steer = 1
                action_acc = 4
        
        else:
            if angle > 180:
                angle = -(360-angle)
            # Replace with your implementation to determine actions to be taken
            if  angle - 180*math.atan((y - 0)/(x - 350))/math.pi > 3:
                action_steer = 0
                action_acc = 2
            elif angle - 180*math.atan((y - 0)/(x - 350))/math.pi < -3:
                action_steer = 2
                action_acc = 2
            # elif angle - 180*math.atan((y - 0)/(x - 350))/math.pi <= 0 and angle - 180*math.atan((y - 0)/(x - 350))/math.pi >= -3:
            #     action_steer = 1
            #     action_acc = 4
            elif angle - 180*math.atan((y - 0)/(x - 350))/math.pi >= -3 and angle - 180*math.atan((y - 0)/(x - 350))/math.pi <= 3:
                action_steer = 1
                action_acc = 3


        # action_steer = 1
        # action_acc = 4 
        # print(pits_centres)
        action = np.array([action_steer, action_acc])  

        return action

    def controller_task2(self, config_filepath=None, render_mode=False):
        """
        This is the main controller function. You can modify it as required except for the parts specifically not to be modified.
        Additionally, you can define helper functions within the class if needed for your logic.
        """
        
        ################ Do NOT modify these lines ################
        pygame.init()
        fpsClock = pygame.time.Clock()

        if config_filepath is None:
            config_filepath = '../configs/config.json'

        time.sleep(3)
        ###########################################################

        # e is the number of the current episode, running it for 10 episodes
        for e in range(NUM_EPISODES):

            ################ Setting up the environment, do NOT modify these lines ################
            # To randomly initialize centers of the traps within a determined range
            ran_cen_1x = random.randint(120, 230)
            ran_cen_1y = random.randint(120, 230)
            ran_cen_1 = [ran_cen_1x, ran_cen_1y]

            ran_cen_2x = random.randint(120, 230)
            ran_cen_2y = random.randint(-230, -120)
            ran_cen_2 = [ran_cen_2x, ran_cen_2y]

            ran_cen_3x = random.randint(-230, -120)
            ran_cen_3y = random.randint(120, 230)
            ran_cen_3 = [ran_cen_3x, ran_cen_3y]

            ran_cen_4x = random.randint(-230, -120)
            ran_cen_4y = random.randint(-230, -120)
            ran_cen_4 = [ran_cen_4x, ran_cen_4y]

            ran_cen_list = [ran_cen_1, ran_cen_2, ran_cen_3, ran_cen_4]            
            eligible_list = []

            # To randomly initialize the car within a determined range
            for x in range(-300, 300):
                for y in range(-300, 300):

                    if x >= (ran_cen_1x - 110) and x <= (ran_cen_1x + 110) and y >= (ran_cen_1y - 110) and y <= (ran_cen_1y + 110):
                        continue

                    if x >= (ran_cen_2x - 110) and x <= (ran_cen_2x + 110) and y >= (ran_cen_2y - 110) and y <= (ran_cen_2y + 110):
                        continue

                    if x >= (ran_cen_3x - 110) and x <= (ran_cen_3x + 110) and y >= (ran_cen_3y - 110) and y <= (ran_cen_3y + 110):
                        continue

                    if x >= (ran_cen_4x - 110) and x <= (ran_cen_4x + 110) and y >= (ran_cen_4y - 110) and y <= (ran_cen_4y + 110):
                        continue

                    eligible_list.append((x,y))

            simulator = DrivingEnv('T2', eligible_list, render_mode=render_mode, config_filepath=config_filepath, ran_cen_list=ran_cen_list)
        
            # To keep track of the number of timesteps per episode
            cur_time = 0

            # To reset the simulator at the beginning of each episode
            state = simulator._reset(eligible_list=eligible_list)
            ###########################################################

            # The following code is a basic example of the usage of the simulator
            road_status = False

            for t in range(TIMESTEPS):
        
                # Checks for quit
                if render_mode:
                    for event in pygame.event.get():
                        if event.type == QUIT:
                            pygame.quit()
                            sys.exit()

                action = self.next_action(state, ran_cen_list)
                state, reward, terminate, reached_road, info_dict = simulator._step(action)
                fpsClock.tick(FPS)

                cur_time += 1

                if terminate:
                    road_status = reached_road
                    break

            print(str(road_status) + ' ' + str(cur_time))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="config filepath", default=None)
    parser.add_argument("-t", "--task", help="task number", choices=['T1', 'T2'])
    parser.add_argument("-r", "--random_seed", help="random seed", type=int, default=0)
    parser.add_argument("-m", "--render_mode", action='store_true')
    parser.add_argument("-f", "--frames_per_sec", help="fps", type=int, default=30) # Keep this as the default while running your simulation to visualize results
    args = parser.parse_args()

    config_filepath = args.config
    task = args.task
    random_seed = args.random_seed
    render_mode = args.render_mode
    fps = args.frames_per_sec

    FPS = fps

    random.seed(random_seed)
    np.random.seed(random_seed)

    if task == 'T1':
        
        agent = Task1()
        agent.controller_task1(config_filepath=config_filepath, render_mode=render_mode)

    else:

        agent = Task2()
        agent.controller_task2(config_filepath=config_filepath, render_mode=render_mode)
