"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

You need to complete the following methods:
    - give_pull(self): This method is called when the algorithm needs to
        select the arms to pull for the next round. The method should return
        two arrays: the first array should contain the indices of the arms
        that need to be pulled, and the second array should contain how many
        times each arm needs to be pulled. For example, if the method returns
        ([0, 1], [2, 3]), then the first arm should be pulled 2 times, and the
        second arm should be pulled 3 times. Note that the sum of values in
        the second array should be equal to the batch size of the bandit.
    
    - get_reward(self, arm_rewards): This method is called just after the
        give_pull method. The method should update the algorithm's internal
        state based on the rewards that were received. arm_rewards is a dictionary
        from arm_indices to a list of rewards received. For example, if the
        give_pull method returned ([0, 1], [2, 3]), then arm_rewards will be
        {0: [r1, r2], 1: [r3, r4, r5]}. (r1 to r5 are each either 0 or 1.)
"""

import numpy as np

# START EDITING HERE
# You can use this space to define any helper functions that you need.
# END EDITING HERE

class AlgorithmBatched:
    def __init__(self, num_arms, horizon, batch_size):
        self.num_arms = num_arms
        self.horizon = horizon
        self.batch_size = batch_size
        assert self.horizon % self.batch_size == 0, "Horizon must be a multiple of batch size"
        # START EDITING HERE
        self.eps = 0.05
        self.success = np.zeros(num_arms)
        self.failure = np.zeros(num_arms)
        self.counts = np.zeros(num_arms)
        self.beta = np.full(num_arms, np.max(np.random.beta(1,1)))
        self.n = 1
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        n = self.num_arms
        if n <= 3:
            num_arms = 1
        elif n <= 10 and n > 3:
            num_arms = 3
        else:
            num_arms = 5
        arms_to_pull = []
        frequency = []
        
        # if self.n <= int(self.horizon*self.eps/(self.batch_size)):
            
        #     arms_to_pull = [np.random.randint(self.num_arms) for x in range(self.batch_size)]
        #     frequency = [1 for x in range(self.batch_size)]
        #     return arms_to_pull, frequency
        # # elif self.n <= 2*int(self.horizon*self.eps/(self.batch_size)):
        # #     values = self.success/self.counts
        # #     arms_to_pull = np.argsort(values)[::-1][:num_arms]
        # #     sum_beta = 0
        # #     for i in arms_to_pull:
        # #         sum_beta += self.beta[i]**50
        # #         # half_batch_size = int(self.batch_size/2)
        # #         # sum = 0
        # #     for i in range(len(arms_to_pull) -1):
        # #         frequency.append(int(self.batch_size*(self.beta[arms_to_pull[i]])**50/sum_beta))
        # #             # sum += int(half_batch_size/2)
        # #             # print(sum, half_batch_size)
        # #             # half_batch_size = half_batch_size/2
        # #     Sum = sum(frequency)
        # #     frequency.append(self.batch_size-Sum)
        # #     return arms_to_pull, frequency
        # else:
        if int(num_arms) <= self.batch_size:
            
            arms_to_pull = np.argsort(self.beta)[::-1][:num_arms]
            sum_beta = 0
            for i in arms_to_pull:
                sum_beta += self.beta[i]**30
            # half_batch_size = int(self.batch_size/2)
            # sum = 0
            # frequency.append(int(self.batch_size*(self.beta[arms_to_pull[0]])**50/sum_beta))
            # frequency.append(int(self.batch_size*(self.beta[arms_to_pull[0]])**50/sum_beta))
            for i in range(0,len(arms_to_pull) -1):
                
                frequency.append(int(self.batch_size*(self.beta[arms_to_pull[i]])**30/sum_beta))
                # sum += int(half_batch_size/2)
                # print(sum, half_batch_size)
                # half_batch_size = half_batch_size/2
            Sum = sum(frequency)
            frequency.append(self.batch_size-Sum)
        else:
            arms_to_pull = np.argsort(self.beta)[::-1][:self.batch_size]
            frequency = [1 for x in range(self.batch_size)]
        # print("DEBUG", arms_to_pull, frequency)
        return arms_to_pull, frequency
            # END EDITING HERE
    
    def get_reward(self, arm_rewards):
        # START EDITING HERE
        arm_index = arm_rewards.keys()

        for i in arm_index:
            self.counts[i] += len(arm_rewards.get(i))
        for i in arm_index:
            rewards = arm_rewards.get(i)
            for j in range(len(rewards)):
                self.success[i] += rewards[j]
        for i in arm_index:
            self.failure[i] = self.counts[i] - self.success[i]
        for i in range(self.num_arms):
            new_beta = np.max(np.random.beta(self.success[i]+1, self.failure[i]+1))
            self.beta[i] = new_beta
        
        self.n += 1       
        # END EDITING HERE