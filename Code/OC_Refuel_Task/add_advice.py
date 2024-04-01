import openai
import json
import os

# Set the proxy
os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

# Apply the API key
openai.api_key = "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"

# Define the text prompt
prompt = "Please convert the following sentences into a more regular form:\n \
         <example> \n \
         original sentence: grab_tool_uncover action should follow the move_to_toolbox action \n \
         regular sentence: grab_tool_uncover should follow move_to_toolbox \n \
         </example> \n \
         original sentence: uncover action should follow the move_to_replenish_position action \n \
         regular sentence:"

# Generate completions using the API
completions = openai.Completion.create(
    model="gpt-3.5-turbo",
    messages=prompt,
    max_tokens=200,
    n=1,
    stop=None,
    temperature=0.5,
)

# Extract the message from the API response
message = completions.choices[0].message.content
print(message)