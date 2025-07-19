import random
import math
import numpy as np

class EXP3():
    def __init__(self,arms):
        self.num_actions = arms
        self.weights = [1.0] * self.num_actions
        self.gamma = 0.07
    # draw: [float] -> int
    # pick an index from the given list of floats proportionally
    # to the size of the entry (i.e. normalize to a probability
    # distribution and draw according to the probabilities).
    def draw(self,prob):
        choice = random.uniform(0, sum(prob))
        choiceIndex = 0

        for weight in prob:
            choice -= weight
            if choice <= 0:
                return choiceIndex
            choiceIndex += 1

    # distr: [float] -> (float)
    # Normalize a list of floats to a probability distribution.  Gamma is an
    # egalitarianism factor, which tempers the distribtuion toward being uniform as
    # it grows from zero to one.
    def distribution(self):
        sum_weights = float(sum(self.weights))
        return tuple((1.0 - self.gamma) * (w / sum_weights) + (self.gamma / len(self.weights)) for w in self.weights)

    def exp3(self,reward, reward_min = 0, reward_max = 1):
        probability_distribution = self.distribution()
        print("probability_distribution: ",probability_distribution)
        choice = self.draw(probability_distribution)
        the_reward = 0
        if(choice < len(reward)):
            the_reward = reward[choice]
            scaled_reward = (the_reward - reward_min) / (reward_max - reward_min) # rewards scaled to 0,1

            estimated_reward = 1.0 * scaled_reward / probability_distribution[choice]
            self.weights[choice] *= math.exp(estimated_reward * self.gamma / self.num_actions) # important that we use estimated reward here!

        return choice, the_reward, estimated_reward, self.weights

class GreedyEpsilon():
    def __init__(self,arms,eps):
        self.num_actions = arms
        self.epsilon = eps
        self.gamma = 0.7
        self.estimates = np.zeros(arms) 

    def draw(self):
        print('estimates: ',self.estimates)
        choice = 0
        if (random.random() < self.epsilon):
            choice = np.random.choice(self.num_actions)
        else:
            choice = np.argmax(self.estimates)
        return choice

    def update(self, x, arm):
       self.estimates[arm]  = (1 - self.gamma)*x + self.gamma * self.estimates[arm]

