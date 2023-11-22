#class State:
#   def __init__(self):
#      self.robotPos = 0
#      # 0表示未抓持工具，1表示抓持uncoverTool，2表示抓持screwTool，3表示抓持insertTool，4表示抓持replenishTool
#      self.Tool = 0
#      # 0表示初始状态，1表示完成uncover操作，2表示完成screw操作，3表示完成insert操作，4表示完成加注操作
#      self.replenish = 0
import numpy as np


class Refuel_Model:
    def __init__(self):
        self.state = [0, 0, 0]
        self.toolPos = 0
        self.replenishPos = 1
        self.actions = ['move_to_toolPos', 'move_to_replenishPos',
                        'cap_uncoverTool', 'cap_screwTool', 'cap_insertTool', 'cap_replenishTool', 'release_Tool',
                        'uncover', 'screw', 'insert', 'replenish']
        self.states = ['robotPos', 'Tool', 'replenish']
        self.action_dim = 11
        self.state_dim = 3

    def find(self, state):
        flag = 0
        if state[2] == 4:
            flag = 1

        return flag

    def execute(self, state, next_state):
        flag = 0
        if state[2] + 1 == next_state[2]:
            flag = 1
        return flag

    # 判断由当前状态进入下一个状态是否符合实际
    def conflict(self, state, next_state):
        flag = 0

        if state[0] == next_state[0] and state[1] == next_state[1] and state[2] == next_state[2]:
            flag = 1
            return flag

        if state[0] == self.toolPos:
            if state[1] != 0 and next_state[1] != 0 and state[1] != next_state[1]:
                flag = 1
                return flag
            if state[2] != next_state[2]:
                flag = 1
                return flag

        if state[0] == self.replenishPos:
            if state[2] > next_state[2] or next_state[2] > state[2] + 1:
                flag = 1
                return flag
            if state[1] != next_state[1]:
                flag = 1
                return flag
            if state[2]!= next_state[2] and state[1] != next_state[2]:
                flag = 1
                return flag

        return flag

    def transform(self, state, action):
        next_state = np.zeros(3)
        next_state[0] = state[0]
        next_state[1] = state[1]
        next_state[2] = state[2]
        if action == 0:
            next_state[0] = self.toolPos
        elif action == 1:
            next_state[0] = self.replenishPos
        elif action == 2:
            next_state[1] = 1
        elif action == 3:
            next_state[1] = 2
        elif action == 4:
            next_state[1] = 3
        elif action == 5:
            next_state[1] = 4
        elif action == 6:
            next_state[1] = 0
        elif action == 7:
            next_state[2] = 1
        elif action == 8:
            next_state[2] = 2
        elif action == 9:
            next_state[2] = 3
        elif action == 10:
            next_state[2] = 4

        flag_conflict = self.conflict(state, next_state)
        if flag_conflict == 1: # 检测到冲突
            return state, -10, False

        flag_find = self.find(next_state)
        if flag_find == 1:
            return next_state, 10, True

        flag_execute = self.execute(state, next_state)
        if flag_execute == 1:
            return next_state, 10, False

        return next_state, -0.1, False

