import copy



class Environment:
    def __init__(self, n):
        self.n = n
        self.state = [[n * i + j + 1 for j in range(n)] for i in range(n)]
        self.state[n - 1][n - 1] = 0
        self.state_string = self.get_state_string(self.state)
        self.agent_pos = [n - 1, n - 1]
        self.terminal_state_string = self.state_string
        self.initial_state_info = None

        Environment.get_reward.flag = True


    # Set state and state string and agent pos of the environment
    def set_state(self, state, state_string, agent_pos):
        self.state = state
        self.state_string = state_string
        self.agent_pos = agent_pos


    # Returns list of actions
    def get_actions(self):
        return [0, 1, 2, 3]
    

    # 0: up, 1: right, 2: down, 3: left
    # Returns next state and agent's next position
    def get_next_state(self, action):
        
        if action == 0:
            if self.agent_pos[0] == 0:
                return self.state, self.agent_pos
            else:
                result = copy.deepcopy(self.state)
                result[self.agent_pos[0]][self.agent_pos[1]] = result[self.agent_pos[0] - 1][self.agent_pos[1]]
                result[self.agent_pos[0] - 1][self.agent_pos[1]] = 0
                return result, [self.agent_pos[0] - 1, self.agent_pos[1]]

        elif action == 1:
            if self.agent_pos[1] == self.n - 1:
                return self.state, self.agent_pos
            else:
                result = copy.deepcopy(self.state)
                result[self.agent_pos[0]][self.agent_pos[1]] = result[self.agent_pos[0]][self.agent_pos[1] + 1]
                result[self.agent_pos[0]][self.agent_pos[1] + 1] = 0
                return result, [self.agent_pos[0], self.agent_pos[1] + 1]

        elif action == 2:
            if self.agent_pos[0] == self.n - 1:
                return self.state, self.agent_pos
            else:
                result = copy.deepcopy(self.state)
                result[self.agent_pos[0]][self.agent_pos[1]] = result[self.agent_pos[0] + 1][self.agent_pos[1]]
                result[self.agent_pos[0] + 1][self.agent_pos[1]] = 0
                return result, [self.agent_pos[0] + 1, self.agent_pos[1]]

        elif action == 3:
            if self.agent_pos[1] == 0:
                return self.state, self.agent_pos
            else:
                result = copy.deepcopy(self.state)
                result[self.agent_pos[0]][self.agent_pos[1]] = result[self.agent_pos[0]][self.agent_pos[1] - 1]
                result[self.agent_pos[0]][self.agent_pos[1] - 1] = 0
                return result, [self.agent_pos[0], self.agent_pos[1] - 1]


    # Converts "state", which is in [[1, 2, 3], [4, 5, 6], [7, 8, 0]] format, to a string, i.e "010203040506070800"
    def get_state_string(self, state):
        result = ""
        for i in range(self.n):
            for j in range(self.n):
                tmp = str(state[i][j])
                if len(tmp) == 1:
                    result += "0" + tmp
                else:
                    result += tmp
        return result
                

    def get_reward(self, state, next_state, state_string, next_state_string):
        if next_state_string == self.terminal_state_string:
            R = 1e8
        else:
            R = -0.5
            # score = self.complete_row_column_score(next_state)
            # R = score - self.n + 2
            # n = self.n
        return R


    # Returns a score based on the number of completed top and left edges
    def complete_row_column_score(self, state):
        result = 0
        n = self.n
        
        for i in range(n - 3):
            in_order_row = 0
            in_order_column = 0
            for j in range(i, n):
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
        

    # Returns a score based on the number of completed columns
    def complete_column_score(self, state):
        result = 0
        
        for j in range(self.n):
            in_order = 0
            for i in range(self.n):
                if state[i][j] == i * self.n + j + 1:
                    in_order += 1
            if in_order == self.n:
                result += self.n - j
        
        return result
    

    # Checks if the state is solvable
    def is_valid(self, state_string):
        n = int((len(state_string) // 2) ** (0.5))

        inversions = self.inversion_count(state_string)
        agent_pos = self.agent_pos_from_bottom(state_string)

        if n % 2 == 1:
            if inversions % 2 == 0:
                return True
            else:
                return False
        else:
            if (inversions % 2 == 1) and (agent_pos % 2 == 0):
                return True
            elif (inversions % 2 == 0) and (agent_pos % 2 == 1):
                return True
            else:
                return False
            

    def inversion_count(self, state_string):
        result = 0
        lst = []
        n = int((len(state_string) // 2) ** (0.5))

        for i in range(0, len(state_string), 2):
            val = int(state_string[i:i + 2])
            if val == 0:
                continue
            lst.append(val)

        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                if lst[i] > lst[j]:
                    result += 1

        return result
    

    def agent_pos_from_bottom(self, state_string):
        n = int((len(state_string) // 2) ** (0.5))
        for i in range(0, len(state_string), 2):
            if int(state_string[i:i + 2]) == 0:
                return n - ((i // 2) // n)


    def print_state(self):
        for i in range(self.n):
            for j in range(self.n):
                print(self.state[i][j], end = " ")
                if 0 <= self.state[i][j] and self.state[i][j] < 10:
                    print(" ", end = "")
            print()