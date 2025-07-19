import random 
import math
import matplotlib.pylab as plt
import numpy as np

class Exp3():
    def __init__(self, counts = [], values = [], n_arms = 0):

        self.n_arms = n_arms
        self.counts = [0 for col in range(n_arms)]
        self.G = [0 for col in range(n_arms)]
        init_proba = float(1/float(n_arms))
        self.weights = [1 for col in range(n_arms)]
        self.values = values
        self.t = 0

    
    def select_arm(self, eta):
        def tirage_aleatoire_avec_proba(proba_vect):
            valeur_test = random.uniform(0, 1)
            arm_chosen = -1
            i = 0
            sum_partiel = 0
            while i <= len(proba_vect) and arm_chosen == -1 :
                sum_partiel += (proba_vect[i])
                if sum_partiel > valeur_test :
                    arm_chosen = i
                i += 1
            return arm_chosen
        self.recalcul_proba = False        
        self.proba_vect = [0 for col in range(self.n_arms)]
            
        
        #####################################
        #ALGO CALCUL DU Pi
        max_G = max(self.G)
        sum_exp_eta_x_Gjt_max_G = 0 
        self.t += 1
        
        for i in range(len(self.G)):
            sum_exp_eta_x_Gjt_max_G += math.exp(eta*(self.G[i] - max_G))
        for i in range(len(self.proba_vect)):
            self.proba_vect[i] = math.exp(eta*(self.G[i] - max_G)) / float(sum_exp_eta_x_Gjt_max_G)
            
        ######################################    
            
        arm_chosen = tirage_aleatoire_avec_proba(self.proba_vect)

        return arm_chosen
    
    
    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        if not (self.recalcul_proba):
            if self.counts[chosen_arm] != 1:
                if self.proba_vect[chosen_arm] != 0:
                    if self.proba_vect[chosen_arm] < 0.01: #Pour eviter les problemes de limite de calcul
                        self.G[chosen_arm] =  float(self.G[chosen_arm]) + (float(reward)/0.01)
                    else :
                        self.G[chosen_arm] =  float(self.G[chosen_arm]) + (float(reward)/float(self.proba_vect[chosen_arm]))
            else:
                self.G[chosen_arm] = reward
        else :
            if self.counts[chosen_arm] != 1:
                if self.proba_vect[chosen_arm] != 0:
                    if self.proba_vect[chosen_arm] < 0.01:
                        self.G[chosen_arm] =  float(self.G[chosen_arm]) + (float(reward)/0.01)
                    else :
                        self.G[chosen_arm] =  float(self.G[chosen_arm]) + (float(reward)/float(self.proba_vect_2[chosen_arm]))
            else:
                self.G[chosen_arm] = reward

def compute_reward(sw,sb):
    return np.matrix.trace(sb)/np.matrix.trace(sw)

def experiment(s_w,s_b,algo):
    #algo = Exp3([],[],n)
    vector_arms_chosen = []
    # reward_vect = []
    # reward_cum = 0
    chosen_arm = algo.select_arm(eta)
    reward =  compute_reward(s_b,s_w)
    algo.update(chosen_arm, reward)
    #reward_cum = reward_cum + reward
    #reward_vect.append(reward_cum / float(i))
    return chosen_arm,reward

#Simulation part
def simulate_arm_bernoulli(proba):
    return 1 if random.random() < proba else 0

def bernoulli_exp3(nb_try,  proba_arm_1, proba_arm_2, eta):
    print("exp3 : cas 2 bras suivant une loi de bernoulli, arm1 : %s, arm2: %s" % (proba_arm_1, proba_arm_2))

    algo = Exp3([],[], 2)
    i = 0
    vector_arms_chosen = []
    reward_vect = []
    reward_cum = 0
    while i < nb_try :
        chosen_arm = algo.select_arm(eta)
        vector_arms_chosen.append(chosen_arm)
        if chosen_arm == 0 :
            reward =  simulate_arm_bernoulli(proba_arm_1)
        else:
            reward = simulate_arm_bernoulli(proba_arm_2)
        algo.update(chosen_arm, reward)
        reward_cum = reward_cum + reward
        i += 1
        reward_vect.append(reward_cum / float(i))
    return reward_vect, algo.values, algo.counts, vector_arms_chosen


if __name__ == '__main__':
    #features
    iteration_nb = 1000
    arm_1_probability = 0.4
    arm_2_probability = 0.3
    eta= 0.05

    #bandit algorithm
    rewards_vect, values, count, arms_chosen = bernoulli_exp3(iteration_nb, arm_1_probability, arm_2_probability, eta)
    print(rewards_vect)
    #Offline solution
    best_arm = [max(arm_1_probability, arm_2_probability) for x in range(iteration_nb)]
    cum_sum_arm_2 = np.cumsum(arms_chosen) #arm_chosen = 1 if arm2 selected, else arm_chosen = 0
    cum_sum_arm_1 = [ x - cum_sum_arm_2[x] for x in range(iteration_nb)]
    plt.plot(range(iteration_nb),cum_sum_arm_1, range(iteration_nb), cum_sum_arm_2);
    plt.title("Nombre de tirages de chaque bras");
    plt.legend(["arm1 Proba %s" % arm_1_probability, "arm2 Proba %s" % arm_2_probability], loc="upper left");
    plt.figure();

    #plot
    plt.plot(range(iteration_nb), best_arm, 'k-', range(iteration_nb), rewards_vect, 'r--');
    plt.title("Gain moyen par tour");
    plt.legend(["Offline","UCB1"], loc="lower right");