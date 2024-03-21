import random
import numpy as np

'''
def constraints_library(env, pre_action, cur_action):
    # Get some information from the environment
    holding_tool = False
    tool_index = None
    refuel_state = int(env.state[3][3])
    action = cur_action

    if env.state[0][3] == 2:
        holding_tool = True
        for i in range(4):
            if env.state[2][i] == 2:
                tool_index = i
                break

    # Constraints Library

    # Constraint 1: pre_action == "move_to_toolbox"
    if pre_action == "move_to_toolbox":
        if not holding_tool:
            action = "grab_tool_" + env.tool_actions[refuel_state - 1]

        if holding_tool:
            action = "release_tool_" + env.tool_actions[tool_index]

    # Constraint 2: pre_action.statwith("grab_tool")
    if pre_action.startswith("grab_tool"):
        action = "move_to_refueling_position"

    # Constraint 3: pre_action == "move_to_refueling_position"
    if pre_action == "move_to_refueling_position":
        if holding_tool and tool_index == refuel_state - 1:
            action = env.tool_actions[refuel_state - 1]

    # Constraint 4: pre_action.startswith("release_tool")
    if pre_action.startswith("release_tool"):
        action = "grab_tool_" + env.tool_actions[refuel_state - 1]

    # Constraint 5: pre_action in env.tool_actions
    if pre_action in env.tool_actions:
        action = "move_to_toolbox"

    return action
'''


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
