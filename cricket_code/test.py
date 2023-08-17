import argparse
parser = argparse.ArgumentParser()

if __name__ == "__main__":
    parser.add_argument("--mdp", type = str)
    args = parser.parse_args()

    mdp = "/data/mdp/continuing-mdp-2-2.txt"
    probs = []
    file = open(args.mdp, "r")
    text = file.readlines()
    # print(text[0].split())
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
    print(probs)
    
    for i in range(states):
        r.append([])

        for j in range(states):
            r[i].append(0)
    probs[1][1][1] = 9
    print(probs)
    for i in range(len(text)):
        trans = text[i].split()
        if trans[0] == "transition":
            
            # print([int(trans[1])],[int(trans[3])],[int(trans[2])])
            probs[int(trans[1])][int(trans[3])][int(trans[2])] = float(trans[5])
            r[int(trans[1])][int(trans[3])] = float(trans[4])
            # print(trans[5]) 
    print(probs)

            
    # text_new = text[4].split()

    # x, y = 
    # print("The value of x: " + x)text[1].split()


    # x, y = input(text[1]).split()
    # print(text[3][2:8:1])
    file.close()