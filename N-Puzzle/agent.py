import random



class Agent:
    def __init__(self, env, V = None, default_state_value = 0.0):
        self.env = env
        self.V = V
        self.default_state_value = default_state_value
        self.S0_info = None

    # Initializes self.S0_info with S0_string
    def save_S0_info(self, S0_string):
        n = int((len(S0_string) // 2) ** (0.5))
        tmp_state = [[0 for _ in range(n)] for _ in range(n)]
        tmp_agent_pos = [0, 0]

        for i in range(n):
            for j in range(n):
                tmp_state[i][j] = int(S0_string[2 * n * i + j * 2: 2 * n * i + j * 2 + 2])

                if tmp_state[i][j] == 0:
                    tmp_agent_pos = [i, j]
        
        self.S0_info = {"state": tmp_state, "state_string": S0_string, "agent_pos": tmp_agent_pos}


    # Asyncronous value iteration. Sweeps through states using epsilon-greedy policy
    def async_value_iteration(self, S0, 
                              max_episodes = 100, 
                              max_steps = None , 
                              learning_rate = 1.0, 
                              epsilon_start = 0.3, 
                              epsilon_end = 0.05, 
                              default_state_value = 0.0,
                              epsilon_decay_type = "linear"):
        
        if not self.check_validity_from_string(S0):
            return
        
        if self.V is None:
            self.V = dict()

        if max_steps is None:
            max_steps = 4 * (self.env.n ** 3)

        self.V[self.env.terminal_state_string] = 0.0
        self.default_state_value = default_state_value
        self.save_S0_info(S0)
        epsilon = epsilon_start
        epsilon_coeff = (epsilon_end / epsilon_start) ** (1 / max_episodes)     # Used for exponential decay
        m = (epsilon_end - epsilon_start) / max_episodes                       # Used for linear decay
        episode = 0


        while episode < max_episodes:
            self.env.set_state(self.S0_info["state"], self.S0_info["state_string"], self.S0_info["agent_pos"])
            step = 0
            done = False

            while True:
                max_G, _, next_S, next_S_string, next_agent_pos = self.select_action(epsilon)
                
                if self.env.state_string not in self.V:
                    self.V[self.env.state_string] = default_state_value
                
                self.V[self.env.state_string] = self.V[self.env.state_string] + learning_rate * (max_G - self.V[self.env.state_string])
                
                self.env.set_state(next_S, next_S_string, next_agent_pos)
                step += 1
                
                if self.env.state_string == self.env.terminal_state_string:
                    done = True

                if done or step >= max_steps:
                    epsilon = epsilon * epsilon_coeff if epsilon_decay_type == "exponential" else epsilon + m
                    break
            
            episode += 1
            
            if done:
                print("------ Episode {} successful ------".format(episode))  
    

    # Same as asyncronous value iteration but uses a sttack to update the states visited in a trajectory in reverse order
    def async_value_iteration_with_stack(self, S0, 
                                         max_episodes = 100, 
                                         max_steps = None , 
                                         learning_rate = 1.0, 
                                         epsilon_start = 0.3, 
                                         epsilon_end = 0.05, 
                                         default_state_value = 0.0, 
                                         epsilon_decay_type = "linear", 
                                         update_on_increase = True):
        
        if not self.check_validity_from_string(S0):
            return
        
        if self.V is None:
            self.V = dict()
        
        if max_steps is None:
            max_steps = 4 * (self.env.n ** 3)

        self.V[self.env.terminal_state_string] = 0.0  
        self.default_state_value = default_state_value
        self.save_S0_info(S0)
        epsilon = epsilon_start
        epsilon_coeff = (epsilon_end / epsilon_start) ** (1 / max_episodes)     # Used for exponential decay
        m = (epsilon_end - epsilon_start) / max_episodes                       # Used for linear decay
        episode = 0
        Agent.unsolvable_subpuzzle.flag = True
        Agent.select_action.max_potential = -1e9

        while episode < max_episodes:
            stack = []
            self.env.set_state(self.S0_info["state"], self.S0_info["state_string"], self.S0_info["agent_pos"])
            step = 0
            done = False

            while not done and step < max_steps:
                max_G, _, next_S, next_S_string, next_agent_pos = self.select_action(epsilon)

                if self.env.state_string not in self.V:
                    self.V[self.env.state_string] = default_state_value

                if update_on_increase:
                    if max_G > self.V[self.env.state_string] or self.V[self.env.state_string] <= self.default_state_value:
                        self.V[self.env.state_string] = self.V[self.env.state_string] + learning_rate * (max_G - self.V[self.env.state_string])
                else:
                    self.V[self.env.state_string] = self.V[self.env.state_string] + learning_rate * (max_G - self.V[self.env.state_string])

                stack.append((self.env.state, self.env.state_string, self.env.agent_pos))
                self.env.set_state(next_S, next_S_string, next_agent_pos)
                step += 1

                if self.env.state_string == self.env.terminal_state_string:
                    done = True
                
                if done or step >= max_steps:
                    # Pops the last state from the stack, which is the state before the terminal state
                    # This state's value will not be updated anyway
                    stack.pop()

                    while len(stack) > 0:
                        next_S, next_S_string, next_agent_pos = stack.pop()
                        self.env.set_state(next_S, next_S_string, next_agent_pos)
                        max_G, _, _, _, _ = self.select_action(epsilon)
                        if update_on_increase:
                            if max_G > self.V[self.env.state_string] or self.V[self.env.state_string] <= self.default_state_value:
                                self.V[self.env.state_string] = self.V[self.env.state_string] + learning_rate * (max_G - self.V[self.env.state_string])
                        else:
                            self.V[self.env.state_string] = self.V[self.env.state_string] + learning_rate * (max_G - self.V[self.env.state_string])
                        
                    epsilon = epsilon * epsilon_coeff if epsilon_decay_type == "exponential" else epsilon + m

            episode += 1
            
            if done:
                print("------ Episode {} successful ------".format(episode))


    def n_step_TD(self, S0,
                  n,
                  max_episodes = 100, 
                  max_steps = None , 
                  epsilon_start = 0.3, 
                  epsilon_end = 0.05, 
                  default_state_value = 0.0,
                  update_on_increase = True, 
                  epsilon_decay_type = "linear",
                  plus_value_iteration = True, 
                  plus_value_iteration_with_stack = True):
        
        if not self.check_validity_from_string(S0):
            return
        
        if self.V is None:
            self.V = dict()

        if max_steps is None:
            max_steps = 4 * (self.env.n ** 3)

        self.V[self.env.terminal_state_string] = 0.0
        self.default_state_value = default_state_value
        self.save_S0_info(S0)
        epsilon = epsilon_start
        epsilon_coeff = (epsilon_end / epsilon_start) ** (1 / max_episodes)     # Used for exponential decay
        m = (epsilon_end - epsilon_start) / max_episodes                       # Used for linear decay
        episode = 0
        Agent.unsolvable_subpuzzle.flag = True
        Agent.select_action.max_potential = -1e9

        while episode < max_episodes:
            self.env.set_state(self.S0_info["state"], self.S0_info["state_string"], self.S0_info["agent_pos"])
            G = 0
            Queue = []
            Queue.append((self.env.state_string, 0))
            if plus_value_iteration_with_stack:
                Stack = []
            step = 0
            done = False
            while True:
                init_state_val = self.V[self.S0_info["state_string"]] if self.S0_info["state_string"] in self.V else default_state_value
                if self.env.state_string == self.env.terminal_state_string:
                    next_S = self.env.state
                    next_S_string = self.env.state_string
                    next_agent_pos = self.env.agent_pos
                    R = 0
                    max_G = 0
                    if not done:
                        done = True
                        step = max_steps
                else:
                    max_G, R, next_S, next_S_string, next_agent_pos = self.select_action(epsilon)

                if self.env.state_string not in self.V:
                    self.V[self.env.state_string] = default_state_value

                if plus_value_iteration:
                    if max_G > self.V[self.env.state_string]: # and max_G >= self.default_state_value:
                        self.V[self.env.state_string] = max_G

                if plus_value_iteration_with_stack and not done:
                    Stack.append((self.env.state, self.env.state_string, self.env.agent_pos))

                if plus_value_iteration_with_stack and (done or step == max_steps + n - 1):
                    while len(Stack) > 0:
                        S, S_string, agent_pos = Stack.pop()
                        self.env.set_state(S, S_string, agent_pos)
                        max_G_stack, _, _, _, _ = self.select_action(epsilon)

                        if max_G_stack > self.V[self.env.state_string] and max_G_stack >= self.default_state_value:
                            self.V[self.env.state_string] = max_G_stack


                self.env.set_state(next_S, next_S_string, next_agent_pos)
                Queue.append((self.env.state_string, R))
                G += R
                step += 1

                if len(Queue) >= n:
                    state_string, reward = Queue.pop(0)

                    if self.env.state_string not in self.V:
                        curr_V = self.default_state_value
                    else:
                        curr_V = self.V[self.env.state_string]
                    
                    G -= reward
                    
                    if update_on_increase:
                        if ((G + curr_V > self.V[state_string]) or self.V[state_string] == self.default_state_value):
                            self.V[state_string] = self.V[state_string] + 1.0 * (G + curr_V - self.V[state_string])
                    else:
                        self.V[state_string] = self.V[state_string] + 1.0 * (G + curr_V - self.V[state_string])
                
                if init_state_val > self.V[self.S0_info["state_string"]]:
                    print("initial state value decreased from {} to {}".format(init_state_val, self.V[self.S0_info["state_string"]]))
                    print("Step: {}".format(step))
                    

                if step >= max_steps + n:
                    break

            epsilon = epsilon * epsilon_coeff if epsilon_decay_type == "exponential" else epsilon + m
            episode += 1

            if done:
                print("------ Episode {} successful ------".format(episode))   
                print("initial state value: {}".format(self.V[self.S0_info["state_string"]]))

    def n_step_TD_2(self, S0,
                  n = 20,
                  max_episodes = 100, 
                  max_steps = None , 
                  epsilon_start = 0.3, 
                  epsilon_end = 0.05, 
                  default_state_value = 0.0):
        
        if not self.check_validity_from_string(S0):
            return
        
        if self.V is None:
            self.V = dict()
            self.alpha = dict()

        if max_steps is None:
            max_steps = 4 * (self.env.n ** 3)

        self.V[self.env.terminal_state_string] = 0.0
        self.alpha[self.env.terminal_state_string] = 0.0
        self.default_state_value = default_state_value
        self.save_S0_info(S0)
        epsilon = epsilon_start
        epsilon_coeff = (epsilon_end / epsilon_start) ** (1 / max_episodes)
        episode = 0

        while episode < max_episodes:
            self.env.set_state(self.S0_info["state"], self.S0_info["state_string"], self.S0_info["agent_pos"])
            G = 0
            Queue = []
            Queue.append((self.env.state_string, 0))
            step = 0
            done = False
            while True:
                if self.env.state_string == self.env.terminal_state_string:
                    next_S = self.env.state
                    next_S_string = self.env.state_string
                    next_agent_pos = self.env.agent_pos
                    R = 0
                    if not done:
                        done = True
                        step = max_steps
                else:
                    _, R, next_S, next_S_string, next_agent_pos = self.select_action(epsilon)

                self.env.set_state(next_S, next_S_string, next_agent_pos)
                Queue.append((self.env.state_string, R))
                G += R
                step += 1

                if len(Queue) >= n + 1:
                    state_string, reward = Queue.pop(0)

                    if state_string not in self.V:
                        self.V[state_string] = self.default_state_value
                        self.alpha[state_string] = 0.0
                        

                    if self.env.state_string not in self.V:
                        curr_V = self.default_state_value
                    else:
                        curr_V = self.V[self.env.state_string]
                    
                    G -= reward
                    self.alpha[state_string] += 1
                    self.V[state_string] = self.V[state_string] + 1.0 / self.alpha[state_string] * (G + curr_V - self.V[state_string])


                if step >= max_steps + n:
                    break

            epsilon *= epsilon_coeff
            episode += 1

            if done:
                print("------ Episode {} successful ------".format(episode))


    # Returns (maximum expected return from the current state, next state, next state string, next agent position)
    def select_action(self, epsilon):
        results = []

        max_return = -1e90
        max_action = None

        current_potential = self.potential(self.env.state, self.env.agent_pos)
    
        for action in self.env.get_actions():

            nextS, next_agent_pos = self.env.get_next_state(action)
            nextS_string = self.env.get_state_string(nextS)
            
            if nextS_string not in self.V:
                nextS_value = self.default_state_value
            else:
                nextS_value = self.V[nextS_string]

            nextS_potential = self.potential(nextS, next_agent_pos)
            reward = self.env.get_reward(self.env.state, nextS, self.env.state_string, nextS_string)
            shaped_reward = reward + nextS_potential - current_potential
            expected_return = nextS_value + shaped_reward

            if current_potential > Agent.select_action.max_potential:
                Agent.select_action.max_potential = current_potential
                print("Higher potential reached: {}".format(Agent.select_action.max_potential))
                self.env.print_state()

            results.append((shaped_reward, nextS, nextS_string, next_agent_pos))
        
            if expected_return > max_return:
                max_return = expected_return
                max_action = action

        if random.random() > epsilon:
            action = max_action
        else:
            action = random.choice(self.env.get_actions())

        return max_return, results[action][0], results[action][1], results[action][2], results[action][3]


    # A heuristic potential function that can be used for reward shaping
    def potential(self, state, agent_pos):
        
        MD = self.manhattan_distance(state)
        LC = self.linear_conflict(state)
        # WMD = self.weighted_manhattan_distance(state)
        # USP = self.unsolvable_subpuzzle(state, agent_pos)
        # CCR = self.complete_row_column_heuristic(state)
        # h = MD + 2 * LC + USP + 5 * CCR
        h = MD + 2 * LC
        # h = WMD + 2 * LC
        h = h * -1
        return h


    def manhattan_distance(self, state):
        n = self.env.n

        md = 0
        for i in range(n):
            for j in range(n):
                if state[i][j] == 0:
                    # i2 = n - 1
                    # j2 = n - 1
                    i2 = i
                    j2 = j
                else:
                    i2 = (state[i][j] - 1) // n
                    j2 = (state[i][j] - 1) % n
                
                md += abs(i2 - i) + abs(j2 - j)

        return md

   
    # Returns the number of linear conflicts in columns and rows
    def linear_conflict(self, state):
        n = self.env.n
        lc = 0

        # Row lc
        for i in range(n):
            max_tile = -1
            for j in range(n):
                if state[i][j] != 0 and (state[i][j] - 1) // n == i:
                    if state[i][j] > max_tile:
                        max_tile = state[i][j]
                    else:
                        lc += 1

        # Column lc
        for j in range(n):
            max_tile = -1
            for i in range(n):
                if state[i][j] != 0 and (state[i][j] - 1) % n == j:
                    if state[i][j] > max_tile:
                        max_tile = state[i][j]
                    else:
                        lc += 1

        return lc


    def unsolvable_subpuzzle(self, state, agent_pos):
        score = self.complete_row_column_score(state)
        n = self.env.n        
        sub_state_string = ""
        if score != 0:
            for i in range(score, n):
                for j in range(score, n):
                    tmp = str(state[i][j])
                    if len(tmp) == 1:
                        sub_state_string += "0"
                    sub_state_string += tmp
        
            if not self.env.is_valid(sub_state_string):
                if Agent.unsolvable_subpuzzle.flag:
                    print("Attention")
                    # print("inversion count: {}".format(self.env.inversion_count(self.env.state_string)))
                    # print("agent pos from bottom: {}".format(self.env.agent_pos_from_bottom(self.env.state_string)))
                    Agent.unsolvable_subpuzzle.flag = False
                # return int(min([agent_pos[0] - score + 1, agent_pos[1] - score + 1])) + 3
                return 2

        return 0


    def weighted_manhattan_distance(self, state):
        n = self.env.n

        md = 0
        for i in range(n):
            for j in range(n):
                if state[i][j] == 0:
                    # i2 = n - 1
                    # j2 = n - 1
                    i2 = i
                    j2 = j
                else:
                    i2 = (state[i][j] - 1) // n
                    j2 = (state[i][j] - 1) % n
                
                degree = min(i2, j2)
                if degree >= n - 3:
                    degree = n - 3
                weight = n - 2 - degree
                md += weight * (abs(i2 - i) + abs(j2 - j))

        return md

    def complete_row_column_heuristic(self, state):
        score = self.complete_row_column_score(state)
        return self.env.n - score - 3


    # Returns a score based on the number of completed rows
    def complete_row_score(self, state):
        result = 0
        n = len(state)
        for i in range(n):
            in_order = 0
            for j in range(n):
                if state[i][j] == i * n + j + 1:
                    in_order += 1
            if in_order == n:
                result += n - i
                    
        return result


    # Returns a score based on the number of completed top and left edges
    def complete_row_column_score(self, state):
        result = 0
        n = self.env.n
        
        for i in range(n - 3):
            in_order_row = 0
            in_order_column = 0
            for j in range(n):
                if state[i][j] == i * n + j + 1:
                    in_order_row += 1
                if state[i][j] != i * n + j + 1:
                    return result
                if state[j][i] == j * n + i + 1:
                    in_order_column += 1
                if state[j][i] != j * n + i + 1:
                    return result
                
            if in_order_row == n and in_order_column == n:
                result += 1

        return result
    

    # Check if the puzzle is solvable starting from the current state
    def check_validity_from_string(self, state_string):
        if self.env.is_valid(state_string):
            return True
        else:
            print("State is not valid!")
            return False


    def exploit(self, S0_string):
        
        self.save_S0_info(S0_string)

        if not self.check_validity_from_string(S0_string):
            return

        self.env.set_state(self.S0_info["state"], self.S0_info["state_string"], self.S0_info["agent_pos"])
        step = 0
        self.env.print_state()

        while self.env.state_string != self.env.terminal_state_string:
            step += 1
            _, _, next_S, next_S_string, next_agent_pos = self.select_action(0.0)
            
            self.env.set_state(next_S, next_S_string, next_agent_pos)
            
            print("- step: {} -".format(step))
            self.env.print_state()

            # if step > 4 * (self.env.n ** 3):
            #     break
            if step > 20000:
                break