import numpy as np


# Define the environment
class RefuelEnv:
    def __init__(self):
        # Initialize the environment parameters
        self.state = np.array([[1, 0],
             [2, 0], [2, 0], [2, 0], [2, 0],
             [3, 0]])
        self.actions = ['move', 'capture', 'release', 'uncover', 'screw', 'insert', 'replenish']
        self.tool = {'uncoverTool': 1, 'screwTool': 2, 'insertTool': 3, 'replenishTool':4}
        self.position = {'toolPos': 2, 'refuelPos': 3}

    def reset(self):
        # Reset the environment to the initial state
        self.state = np.array([[1, 0],
             [2, 0], [2, 0], [2, 0], [2, 0],
             [3, 0]])

    def step(self, action, *args):
        # Execute the given action, return the next state and reward

        if action == 'move':
            state, reward, done = self.move(*args)
        elif action == 'capture':
            state, reward, done = self.capture(*args)
        elif action == 'release':
            state, reward, done = self.release(*args)
        elif action == 'uncover':
            state, reward, done = self.uncover()
        elif action == 'screw':
            state, reward, done = self.screw()
        elif action == 'insert':
            state, reward, done = self.insert()
        elif action == 'replenish':
            state, reward, done = self.replenish()

        if self.state[5][1] == 4:
            done = True
        return state, reward, done

    def move(self, position):
        next_position = self.position[position]
        if self.state[0][0] == next_position:
            return self.state, -10, False
        else:
            eff = np.zeros((6, 2))
            for i in range(6):
                eff[i][0] = next_position
            M = np.zeros((6, 2))
            M[0][0] = 1
            for i in range(1, 5):
                if self.state[i][1] == 1:
                    M[i][0] = 1
            self.state = np.dot(self.state, np.where(M == 1, 0, 1)) + np.dot(eff, M)
            return self.state, -1, False

    def capture(self, tool_type):
        u = self.tool[tool_type]
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['toolPos']
        pre[u][0] = self.position['toolPos']
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[1][0] = 1
        pre_M[u][0] = 1
        pre_M[u][0] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            eff = np.zeros((6, 2))
            eff[0][0] = self.position['toolPos']
            eff[u][0] = self.position['toolPos']
            eff[0][1] = 1
            eff[u][1] = 1
            M = np.zeros((6, 2))
            M[0][1] = 1
            M[u][1] = 1
            self.state = np.dot(self.state, np.where(M == 1, 0, 1)) + np.dot(eff, M)
            return self.state, -1, False

    def release(self, tool_type):
        u = self.tool[tool_type]
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['toolPos']
        pre[u][0] = self.position['toolPos']
        pre[0][1] = 1
        pre[u][1] = 1
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[1][0] = 1
        pre_M[u][0] = 1
        pre_M[u][0] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            eff = np.zeros((6, 2))
            eff[0][0] = self.position['toolPos']
            eff[u][0] = self.position['toolPos']
            M = np.zeros((6, 2))
            M[0][1] = 1
            M[u][1] = 1
            self.state = np.dot(self.state, np.where(M == 1, 0, 1)) + np.dot(eff, M)
            return self.state, -1, False

    def uncover(self):
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['refuelPos']
        pre[0][1] = 1
        pre[1][0] = self.position['refuelPos']
        pre[1][1] = 1
        pre[5][1] = 0
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[0][1] = 1
        pre_M[1][0] = 1
        pre_M[1][1] = 1
        pre_M[5][1] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            self.state[5][1] = 1
            return self.state, 10, False

    def screw(self):
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['refuelPos']
        pre[0][1] = 1
        pre[2][0] = self.position['refuelPos']
        pre[2][1] = 1
        pre[5][1] = 1
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[0][1] = 1
        pre_M[2][0] = 1
        pre_M[2][1] = 1
        pre_M[5][1] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            self.state[5][1] = 2
            return self.state, 10, False

    def insert(self):
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['refuelPos']
        pre[0][1] = 1
        pre[3][0] = self.position['refuelPos']
        pre[3][1] = 1
        pre[5][1] = 2
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[0][1] = 1
        pre_M[3][0] = 1
        pre_M[3][1] = 1
        pre_M[5][1] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            self.state[5][1] = 3
            return self.state, 10, False

    def replenish(self):
        pre = np.zeros((6, 2))
        pre[0][0] = self.position['refuelPos']
        pre[0][1] = 1
        pre[4][0] = self.position['refuelPos']
        pre[4][1] = 1
        pre[5][1] = 2
        pre_M = np.zeros((6, 2))
        pre_M[0][0] = 1
        pre_M[0][1] = 1
        pre_M[4][0] = 1
        pre_M[4][1] = 1
        pre_M[5][1] = 1
        if np.dot(self.state, pre_M) != np.dot(pre, pre_M):
            return self.state, -10, False
        else:
            self.state[5][1] = 4
            return self.state, 10, False

