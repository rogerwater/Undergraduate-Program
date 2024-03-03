import numpy as np
import random


class RefuelingEnv():
    def __init__(self):
        # 初始化状态
        # 第一行表示机器人的位置，grab状态
        # 第二行至第五行表示四个工具的位置和grab状态，初始状态下都位于工具箱处
        # 第六行表示加注位置，姿态和加注状态，初始状态下加注状态为1，表示未加注
        # 未被抓取的工具的grab状态为1，被抓取的工具的grab状态为2
        self.state_space_shape = (6, 4, 1)
        self.action_space_shape = 14
        self.robot_init_state = (-7, 2, 2, 1)
        self.tool_state = (-7, 2, 2, 1)
        self.refueling_state = (2, 4, 3, 1)
        self.tool_actions = ['uncover', 'screw', 'insert', 'replenish']
        self.action_space = ['move_to_toolbox', 'move_to_refueling_position'] + \
                            ['grab_tool_' + tool for tool in self.tool_actions] + \
                            ['release_tool_' + tool for tool in self.tool_actions] + \
                            self.tool_actions  # 14 actions in total

        self.action_mapping = {idx: action for idx, action in enumerate(self.action_space)}
        self.reverse_action_mapping = {action: idx for idx, action in enumerate(self.action_space)}

        self.current_action_index = 1  # 判断加注任务执行到哪一步：1表示未开始，2表示uncover，3表示screw，4表示insert，5表示replenish
        self.state = None
        self.reset()

    def reset(self):
        # 重置环境
        robot_state = np.array([self.robot_init_state], dtype=int)
        tools_state = np.array([self.tool_state] * 4, dtype=int)
        refueling_state = np.array([self.refueling_state], dtype=int)
        self.state = np.concatenate((robot_state, tools_state, refueling_state), axis=0)
        return self.state.flatten()

    def get_action_by_index(self, index):
        # 通过索引获取字符串
        index = int(index)
        return self.action_mapping.get(index, None)

    def get_index_by_action(self, action):
        # 通过字符串获取索引
        return self.reverse_action_mapping.get(action, None)

    def step(self, action):
        reward = -1
        done = False

        if action == 'move_to_toolbox':
            self.state[0][0] = self.tool_state[0]
            self.state[0][1] = self.tool_state[1]
            self.state[0][2] = self.tool_state[2]

            # 如果工具被抓取，则机器人移动时工具也跟着移动
            if self.state[0][3] == 2:
                for i in range(1, 5):
                    if self.state[i][3] == 2:
                        self.state[i][0] = self.tool_state[0]
                        self.state[i][1] = self.tool_state[1]
                        self.state[i][2] = self.tool_state[2]
                        break

            reward = -3

        elif action == 'move_to_refueling_position':
            self.state[0][0] = self.refueling_state[0]
            self.state[0][1] = self.refueling_state[1]
            self.state[0][2] = self.refueling_state[2]

            # 如果工具被抓取，则机器人移动时工具也跟着移动
            if self.state[0][3] == 2:
                for i in range(1, 5):
                    if self.state[i][3] == 2:
                        self.state[i][0] = self.refueling_state[0]
                        self.state[i][1] = self.refueling_state[1]
                        self.state[i][2] = self.refueling_state[2]
                        break

            reward = -3

        elif action.startswith('grab_tool_'):
            tool_index = self.tool_actions.index(action[10:])
            # 前提条件：机器人在工具箱处，工具在工具箱处，工具未被抓取
            if self.state[0][0] == self.tool_state[0] and self.state[tool_index + 1][0] == self.tool_state[0] and \
                    self.state[0][1] == self.tool_state[1] and self.state[tool_index + 1][1] == self.tool_state[1] and \
                    self.state[0][2] == self.tool_state[2] and self.state[tool_index + 1][2] == self.tool_state[2] and \
                    self.state[tool_index + 1][3] == 1 and self.state[0][3] == 1:
                self.state[tool_index + 1][3] = 2
                self.state[0][3] = 2
            else:
                reward = -3

        elif action.startswith('release_tool_'):
            tool_index = self.tool_actions.index(action[13:])
            # 前提条件：机器人在工具箱处，工具在机器人处，工具被抓取
            if self.state[0][0] == self.tool_state[0] and self.state[tool_index + 1][0] == self.state[0][0] and \
                    self.state[0][1] == self.tool_state[1] and self.state[tool_index + 1][1] == self.state[0][1] and \
                    self.state[0][1] == self.tool_state[1] and self.state[tool_index + 1][1] == self.state[0][1] and \
                    self.state[tool_index + 1][3] == 2 and self.state[0][3] == 2:
                self.state[tool_index + 1][3] = 1
                self.state[0][3] = 1
            else:
                reward = -3

        elif action in self.tool_actions:
            # 前提条件：机器人在加注位置，工具在机器人处，工具被抓取，加注状态所需工具与当前所持工具一致
            if self.current_action_index <= len(self.tool_actions) and action == self.tool_actions[
                self.current_action_index - 1]:
                if self.state[0][0] == self.refueling_state[0] and self.state[self.current_action_index][0] == self.state[0][0] and \
                        self.state[0][1] == self.refueling_state[1] and self.state[self.current_action_index][1] == self.state[0][1] and \
                        self.state[0][2] == self.refueling_state[2] and self.state[self.current_action_index][2] == self.state[0][2] and \
                        self.state[self.current_action_index][3] == 2 and self.state[0][3] == 2:
                    self.current_action_index += 1
                    self.state[5][3] += 1
                    reward = 30
                else:
                    reward = -3
            else:
                reward = -3

        if self.current_action_index == 5:
            done = True

        return self.state.flatten(), reward, done

    def choose_action(self):
        return random.choice(self.action_space)


if __name__ == "__main__":
    refuel_env = RefuelingEnv()
    refuel_env.reset()
    print("Initial State:")
    print(refuel_env.state)
    # print(refuel_env.action_space)
    # action = 1
    # print(refuel_env.get_action_by_index(action))
    for _ in range(10):
        action = refuel_env.choose_action()
        state, reward, done = refuel_env.step(action)
        print(f"Action: {action}, Reward: {reward}, Done: {done}, State:\n{state}")

    print("Final State:")
    print(refuel_env.state)