from openai import OpenAI
import os
from model import RefuelingEnv


def replanning(action_sequence):
    # Initial replanning
    env = RefuelingEnv()
    obs = env.reset()
    better_action_sequence = []
    for action in action_sequence:
        # print(action)
        next_obs, reward, done = env.step(action)
        if all(a == b for a, b in zip(obs, next_obs)):
            pass
        else:
            better_action_sequence.append(action)
            obs = next_obs

    '''
    # Use GPT3.5 turbo to get optimal action sequence based on action sequence and better action sequence
    # Set the proxy
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    # Apply the API key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Define the text prompt
    prompt = "You are an engineer proficient in task planning. " \
             "In the on-orbit filling task learning environment, you need to optimize the action sequence generated using reinforcement learning algorithm. " \
             "I will give the initial action sequence obtained using the reinforcement learning algorithm. " \
             "You need to check the generated action sequence, delete redundant actions and ensure the rationality of action execution. " \
             "For example, if the robot keeps moving continuously, it only needs to move to its final position in one step. " \
             "For example, if a tool is grabbed multiple times, it only needs to be reduced to one. " \
             "Please optimize the action sequence based on the above examples and do not delete other correct actions. " \
             "Note that you can only output the simplest optimized action sequence and output it in the form of a python list."
    prompt_action_sequence  = ", ".join(action_sequence)
    print("action_sequence: ", better_action_sequence)
    message = f"Initial action sequence: {prompt_action_sequence}"

    # Generate completions using the API
    completions = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": message}
        ],
        model="gpt-3.5-turbo",
    )

    # Extract the message from the API response
    optimal_action_sequence = completions.choices[0].message.content
    print(optimal_action_sequence)
    '''

    return better_action_sequence


if __name__ == "__main__":
    replanning(["grab_tool_uncover", "grab_tool_replenish", "release_tool_uncover", "grab_tool_uncover",
                "move_to_refueling_position", "uncover", "move_to_toolbox", "release_tool_uncover",
                "grab_tool_unscrew", "move_to_refueling_position", "unscrew", "move_to_toolbox",
                "release_tool_unscrew", "grab_tool_insert", "move_to_refueling_position", "insert",
                "move_to_toolbox", "move_to_refueling_position", "move_to_toolbox", "release_tool_insert",
                "grab_tool_replenish", "move_to_refueling_position", "replenish"])

