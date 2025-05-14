import pandas as pd
import networkx as nx
import random

# Define the graph
edges = [
    ("lobby", "lift"), ("lift", "stairs"), ("stairs", "hallway"),
    ("stairs", "washroom"), ("hallway", "cafeteria"), ("cafeteria", "exit"),
    ("exit", "office2"), ("office2", "office1"), ("office1", "gym"), ("gym", "washroom")
]
G = nx.Graph()
G.add_edges_from(edges)
nodes = list(G.nodes)

# Parameters
num_rows = 600
generated_rows = []

# Generate paths
while len(generated_rows) < num_rows:
    start = random.choice(nodes)
    goal = random.choice(nodes)
    if start == goal:
        continue
    try:
        path = nx.shortest_path(G, start, goal)
        total_distance = round(random.uniform(5.0, 20.0), 2)
        total_turns = random.randint(1, 4)
        for i in range(len(path) - 1):
            generated_rows.append({
                "start": path[i],
                "goal": goal,
                "next_preferred_node": path[i + 1],
                "distance": total_distance,
                "turns": total_turns
            })
            if len(generated_rows) >= num_rows:
                break
    except nx.NetworkXNoPath:
        continue

# Save
df = pd.DataFrame(generated_rows)
df.to_csv("step_data2.csv", index=False)
print(f"✅ Generated {len(df)} rows and saved to step_data.csv")
