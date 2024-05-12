from openai import OpenAI
import os


# Function to add advice for the task planning environment
def add_advice(advice):
    # Set the proxy
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

    # Apply the API key
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    # Define the text prompt
    prompt = "You are an engineer proficient in reinforcement learning and task planning. " \
             "In the on-orbit filling task learning environment, you need to convert user input into executable python code. " \
             "The specific parameters involved are as follows: " \
             "There are 5 grabbing states, 0 means not grabbing the tool, 1 means grabbing the uncover tool, 2 means grabbing the unscrew tool, 3 means grabbing the insert tool, and 4 means grabbing the replenish tool." \
             "There are 5 refueling states, 1 means initial state, 2 means uncovered, 3 means unscrewed, 4 means inserted, and 5 means replenished." \
             "There are 2 positions, toolbox_position means the toolbox position, and refueling_position means the refueling position." \
             "Please convert suggestions involving natural language input for the above parameters into python code." \
             "For example, if the input suggestion is: If the robot does not grab the tool at this time, the refueling state is the initial state, and the robot is at the tool box position, the robot should perform the action grab_tool_uncover at this time." \
             "The corresponding python code should be: pbrs.rule_library.add_rule((0, 1, 'toolbox_position'), 'grab_tool_uncover')\\" \
             "Note that you only need to output the corresponding python code"
    # print(prompt)
    advice = advice

    # Generate completions using the API
    completions = client.chat.completions.create(
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": advice}
        ],
        model="gpt-3.5-turbo",
    )

    # Extract the message from the API response
    generated_code = completions.choices[0].message.content
    return generated_code


if __name__ == "__main__":
    advice = input("Please enter the advice: ")
    generated_code = add_advice(advice)
    print(generated_code)
    # exec(generated_code)
