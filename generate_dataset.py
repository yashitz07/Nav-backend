import pandas as pd
import ast

# Load the original dataset
df = pd.read_csv("path_data.csv")

# Container for new rows
step_rows = []

# Iterate through each row to generate step-wise data
for _, row in df.iterrows():
    start = row["start"]
    goal = row["goal"]
    path = ast.literal_eval(row["preferred_path"])  # Convert stringified list to actual list
    distance = row["distance"]
    turns = row["turns"]

    # For each hop in the path, add a row (except last node)
    for i in range(len(path) - 1):
        step_rows.append({
            "current_node": path[i],
            "target_node": goal,
            "next_node": path[i + 1],
            "distance": distance,
            "turns": turns
        })

# Create the new DataFrame
step_df = pd.DataFrame(step_rows)

# Save to CSV
step_df.to_csv("step_data.csv", index=False)
print("✅ Step-wise dataset saved as step_data.csv")
