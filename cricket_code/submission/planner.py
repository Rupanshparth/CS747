import argparse
import numpy as np
import pulp
import math
parser = argparse.ArgumentParser()

def bellman(states,action, probs, F, s, r, gamma):
    re = [0]*states
    for a in range(action):
        val = 0
        for s_next in range(states):
            val += probs[s][s_next][a]*(r[s][s_next][a] + gamma*F[s_next])
        re[a] = val
    return re


def vi(states, actions, probs, rewards, gamma, delta, Vo = None):
    V = Vo if Vo else [0]*states
    while True:
        V1 = [0]*states
        for s in range(states):
            V1[s] = max(bellman(states, action, probs, V, s, r, gamma))
        if math.fabs(sum(np.abs(V)) - sum(np.abs(V1))) < delta:
            break
        else:
            V = V1
    return V1

def policy_iteration(states, actions, probs, gamma, rewards, delta):

    # Set policy iteration parameters
    max_policy_iter = 10000  # Maximum number of policy iterations
    max_value_iter = 10000  # Maximum number of value iterations
    pi = [0]*states
    V = [0]*states


    for i in range(max_policy_iter):
        # Initial assumption: policy is stable
        optimal_policy_found = True

        # Policy evaluation
        # Compute value for each state under current policy
        for j in range(max_value_iter):
            max_diff = 0  # Initialize max difference
            V_new = [0]*states  # Initialize values
            for s in range(0,states):

                # Compute state value
                val = 0  # Get direct reward
                for s_next in range(0,states):
                    val += probs[s][s_next][pi[s]] * (rewards[s][s_next][a] +
                            gamma * V[s_next]
                    )  # Add discounted downstream values

                # Update maximum difference
                max_diff = max(max_diff, abs(val - V[s]))

                V[s] = val  # Update value with highest value
            # If diff smaller than threshold delta for all states, algorithm terminates
            if max_diff < delta:
                break

    # Policy iteration
    # With updated state values, improve policy if needed
    for s in range(0, states):
        
        val_max = V[s]
        for a in range(0, actions):
            val = 0  # Get direct reward
            for s_next in range(0, states):
                val += probs[s][s_next][a] * ( rewards[s][s_next][a] +
                    gamma * V[s_next]
                )  # Add discounted downstream values

            # Update policy if (i) action improves value and (ii) action different from current policy
            if val > val_max and pi[s] != a:
                pi[s] = a
                val_max = val
                optimal_policy_found = False

    # If policy did not change, algorithm terminates
        if optimal_policy_found:
            break

    for s in range(0, states):
        print(str(round(V[s], 6)) + " " + str(pi[s]))
    return
    

def linear_prog():

    # definition of the MDP
    N = 3 # nb of states
    A = 3 # nb of actions
    # the transition tensor: P [current state, action, next state] = probability of transiting to next state when emitting action in current state
    P = np.array ([[[1/2, 1/4, 1/4],
                    [1/16, 3/4, 3/16],
                    [1/4, 1/8, 5/8]], 
                [[1/2, 0, 1/2],
                    [0,0,0],
                    [1/16, 7/8, 1/16]],
                [[1/4, 1/4, 1/2], 
                    [1/8, 3/4, 1/8], 
                    [3/4, 1/16, 3/16]]])
    # the return function: expected return when transiting to next state when emitting action in current state
    R = np.array ([10, 4, 8, 8, 2, 4, 4, 6, 4, 14, 0, 18, 0, 0, 0, 8, 16, 8, 10, 2, 8, 6, 4, 2, 4, 0, 8])
    R = R.reshape ([N, A, N])
    gamma = .9
    # in states 0 and 2, all 3 actions are possible. In state 1, only actions 0 and 2 are possibles:
    possible_actions = np.array ([[0, 1, 2], [0, 2], [0, 1, 2]])

    # define the LP
    v = pulp.LpVariable.dicts ("s", (range (N))) # the variables
    prob = pulp.LpProblem ("taxicab", pulp.LpMinimize) # minimize the objective function
    prob += sum ([v [i] for i in range (N)]) # defines the objective function
    # now, we define the constrain: there is one for each (state, action) pair.
    for i in range (N):
        for a in possible_actions [i]:
            prob += v [i] - gamma * sum (P [i, a, j] * v [j] for j in range(N)) >= sum (P [i, a, j] * R [i, a, j] for j in range(N))

    # Solve the LP
    prob.solve ()
    # after resolution, the status of the solution is available and can be printed:
    # print("Status:", pulp.LpStatus[prob.status])
    # in this case, it should print "Optimal"

    # extract the value function
    V = np.zeros (N) # value function
    for i in range (N):
        V [i] = v [i]. varValue

    # extract the optimal policy
    pi_star = np.zeros ((N), dtype=np.int64)
    vduales = np.zeros ((N, 3))
    s = 0
    a = 0
    for name, c in list(prob.constraints.items()):
        vduales [s, a] = c.pi
        if a < A - 1:
            a = a + 1
        else:
            a = 0
            if s < N - 1:
                s = s + 1
            else:
                s = 0
    for s in range(N):
        pi_star[s] = np.argmax(vduales[s, :])
        

if __name__ == "__main__":
    parser.add_argument("--mdp", type = str)
    parser.add_argument("--algorithm", type = str, default = "vi")
    parser.add_argument("--policy",type=str)
    args = parser.parse_args()

    mdp = args.mdp
    algorithm = args.algorithm
    policy = args.policy
    delta = 1e-3

    mdp_file = open(args.mdp, "r")
    text = mdp_file.readlines()
    mdp_file.close()

    text_states , states = text[0].split()
    text_action, action = text[1].split()
    states = int(states)
    action = int(action)
    probs = []
    r = []
    for i in range(states):
        probs.append([])

        for j in range(states):
            probs[i].append([])

            for k in range(action):
                probs[i][j].append(0)
    # print(probs)
    
    for i in range(states):
        r.append([])

        for j in range(states):
            r[i].append([])

            for k in range(action):
                r[i][j].append(0)

    probs[1][1][1] = 9
    # print(probs)
    for i in range(len(text)):
        trans = text[i].split()
        if trans[0] == "transition":
            
            # print([int(trans[1])],[int(trans[3])],[int(trans[2])])
            probs[int(trans[1])][int(trans[3])][int(trans[2])] = float(trans[5])
            r[int(trans[1])][int(trans[3])][int(trans[2])] = float(trans[4])
        elif trans[0] == "mdptype":
            mdptype = trans[1]
        elif trans[0] == "discount":
            gamma = float(trans[1])

    if algorithm == "vi":
        vi(states, action, probs, r, gamma, delta, Vo = None)
    elif algorithm == "hpi":
        policy_iteration(states, action, probs, gamma, r, delta)
        


