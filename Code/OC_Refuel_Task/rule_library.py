import random
import numpy as np
from model import RefuelingEnv


def constraints_library(env, pre_action, cur_action):
    # Get some information from the environment
    holding_tool = False
    action = cur_action
    tool_index = None
    if env.state[0][3] == 2:
        holding_tool = True
        for i in range(4):
            if env.state[2][i] == 2:
                tool_index = i
                break

    # Constraints Library

    # Constraint 1: pre_action == "move_to_toolbox"
    if pre_action == "move_to_toolbox":
        if not holding_tool and not action.startswith("grab_tool"):
            action = "grab_tool_" + random.choice(env.tool_actions)

        if holding_tool and not action.startswith("release_tool"):
            action = "release_tool_" + env.tool_actions[tool_index]

    # Constraint 2: pre_action == "move_to_refueling_position"
    if pre_action == "move_to_refueling_position":
        if holding_tool and not action in env.tool_actions:
            action = env.tool_actions[tool_index]
        if not holding_tool and action != "move_to_toolbox":
            action = "move_to_toolbox"

    # Constraint 3: pre_action.startswith("grab_tool")
    if pre_action.startswith("grab_tool") and action != "move_to_refueling_position":
        action = "move_to_refueling_position"

    # Constraint 4: pre_action.startswith("release_tool")
    if pre_action.startswith("release_tool") and not action.startswith("grab_tool"):
        action = "grab_tool_" + random.choice(env.tool_actions)

    # Constraint 5: pre_action in env.tool_actions
    if pre_action in env.tool_actions and action != "move_to_toolbox":
        action = "move_to_toolbox"

    return action


class RuleLibrary:
    def __init__(self):
        self.rule_dict = {}

    def add_rule(self, info, suggested_action):
        self.rule_dict[info] = suggested_action

    def get_suggested_action(self, info):
        return self.rule_dict.get(info, None)


class PotentialBasedRewardShaping:
    def __init__(self, gamma, w):
        self.rule_library = RuleLibrary()
        self.gamma = gamma
        self.w = w
        self.env = RefuelingEnv()

    def potential_function(self, info, action):
        # Calculate the potential function
        suggested_action = self.rule_library.get_suggested_action(info)
        if suggested_action is None:
            return 0.0, False, None
        if action == suggested_action:
            return self.w, False, action
        else:
            return -0.1, True, suggested_action

    def shaped_reward(self, reward, potential):
        return reward + self.gamma * potential

    def add_rules_automatically_human(self):
        print("Adding 4 rules automatically before training...")
        self.rule_library.add_rule((0, 1, "toolbox_position"), "grab_tool_uncover")
        self.rule_library.add_rule((0, 2, "toolbox_position"), "grab_tool_unscrew")
        self.rule_library.add_rule((0, 3, "toolbox_position"), "grab_tool_insert")
        self.rule_library.add_rule((0, 4, "toolbox_position"), "grab_tool_replenish")

    def add_rules_automatically(self):
        print("Adding 8 rules automatically before training...")
        self.rule_library.add_rule((0, 1, "toolbox_position"), "grab_tool_uncover")
        self.rule_library.add_rule((0, 2, "toolbox_position"), "grab_tool_unscrew")
        self.rule_library.add_rule((0, 3, "toolbox_position"), "grab_tool_insert")
        self.rule_library.add_rule((0, 4, "toolbox_position"), "grab_tool_replenish")

        self.rule_library.add_rule((1, 1, "refueling_position"), "uncover")
        self.rule_library.add_rule((2, 2, "refueling_position"), "unscrew")
        self.rule_library.add_rule((3, 3, "refueling_position"), "insert")
        self.rule_library.add_rule((4, 4, "refueling_position"), "replenish")

    def get_info_from_state(self, state):
        # get hold_state
        if state[0][3] == 1:
            hold_state = 0
        elif state[0][3] == 2 and state[2][0] == 2:
            hold_state = 1
        elif state[0][3] == 2 and state[2][1] == 2:
            hold_state = 2
        elif state[0][3] == 2 and state[2][2] == 2:
            hold_state = 3
        elif state[0][3] == 2 and state[2][3] == 2:
            hold_state = 4

        # get refuel_state
        refuel_state = state[3][3]

        # get position
        if state[0][0] == state[1][0] and state[0][1] == state[1][1] and state[0][2] == state[1][2]:
            position = "toolbox_position"
        else:
            position = "refueling_position"

        info = (hold_state, refuel_state, position)
        return info
