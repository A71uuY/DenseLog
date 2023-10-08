from math import gamma
import random
import numpy as np
from itertools import product
import copy


class QTable(object):
    def __init__(self, state_num, action_num, agent_num, gamma=0.99, alpha=0.3) -> None:
        self.qt = np.random.random((state_num, np.power(action_num, agent_num)))
        #self.qt = np.zeros((state_num, np.power(action_num, agent_num)))
        self.gamma = gamma
        self.alpha = alpha
        pass

    def get_q(self, state, action):
        return self.qt[state][action]

    def get_q_row(self, state):
        return self.qt[state]

    def update_q(self, reward,state_id,action_id,s_new_id,pi_a_snew):
        if self.alpha > 0.001:
            self.alpha -= 0.0001
        V_snew = np.sum(pi_a_snew * self.qt[s_new_id])
        td_err = reward - self.qt[state_id][action_id] + self.gamma*V_snew
        self.qt[state_id][action_id] += td_err*self.alpha
        return 1


    def update_q_done(self, reward, s, a):
        
        self.qt[s, a] = reward
        return 1


class JointPolicy(object):
    """
    This is the joint policy version, which means agents do not act individually
    """
    def __init__(self, env) -> None:
        super().__init__()
        self.n_agents = env.num_agents
        self.all_states = env.get_all_states()
        self.numS = env.possible_states
        self.numA = env.joint_act_nums
        self.env = env
        self.policy = np.full(self.numA*self.numS, 1/(self.numA)).reshape(self.numS, self.numA)
        self.gamma = 0.9
    def derive_pi(self, occupancy_measure):
        for i in range(len(occupancy_measure)):
            row = occupancy_measure[i]
            total_om = np.sum(row)
            pi_current_s = np.divide(row, total_om)
            self.policy[i] = pi_current_s
        return self.policy

def pick_action(state, policy):
    state = int(state)
    act_probs = policy.policy[state]
    act = np.random.choice(a=np.arange(len(act_probs)), size=1, p=act_probs)
    return int(act[0])

def TD_update_q(qt_tables, policy, env, max_iter):
    """
    Update q-tables by td-error
    """
    for m in range(10):
        s = env.reset()
        for n in range(int(max_iter/10)):
            old_tables = [copy.deepcopy(qt_tables[i]) for i in range(env.num_agents)]
            state_id = env.vec_to_ind(s)
            action_id = pick_action(state_id, policy)
            action = env.ind_to_act(action_id)
            s_new, r, done, info = env.step(action)
            #if (not done) and (s_new is not None):
            s_new_id = env.vec_to_ind(s_new)
            pi_a_snew = policy.policy[s_new_id]
            for i in range(env.num_agents):
                qt_tables[i].update_q(r[i], state_id,action_id,s_new_id,pi_a_snew)
            s = s_new
    return 1


if __name__ == '__main__':
    min_state = np.array([-2, -2, -2, -2, -2])
    state_len = 5
    all_cases = [0 for _ in range(3125)]
    for q in range(-2, 3):
        for w in range(-2, 3):
            for e in range(-2, 3):
                for r in range(-2, 3):
                    for t in range(-2, 3):
                        state = [q, w, e, r, t]
                        diff = state - min_state
                        to_id = 0
                        for i in range(state_len):
                            to_id += np.power(5, 4 - i) * diff[i]
                            all_cases[to_id] = 1
    if sum(all_cases) == 3125:
        print("correct")
    if sum(np.array(all_cases) == 0) > 0:
        print("false")