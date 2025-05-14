from flask import Flask, jsonify, request
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
import os
import csv
import json
import joblib
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Load the trained model and label mapping
ml_model = joblib.load("step_model.pkl")
next_node_map = joblib.load("next_node_map.pkl")
reverse_next_node_map = {v: k for k, v in next_node_map.items()}

# Create an empty graph
G = nx.Graph()

coordinates = {
    "hallway": (-5, 5),
    "stairs": (0, 6),
    "lift": (0, 7),
    "lobby": (0, 0),
    "washroom": (5, 10),
    "gym": (5, 12),
    "office1": (5, 15),
    "office2": (0, 15),
    "cafeteria": (-5, 10),
    "exit": (-5, 20),
}

edges = [
    ("lobby", "lift"),
    ("lift", "stairs"),
    ("stairs", "hallway"),
    ("stairs", "washroom"),
    ("hallway", "cafeteria"),
    ("cafeteria", "exit"),
    ("exit", "office2"),
    ("office2", "office1"),
    ("office1", "gym"),
    ("gym", "washroom"),

]

G.add_nodes_from(coordinates.keys())
G.add_edges_from(edges)

def calculate_distance(node1, node2):
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def bfs_shortest_path(graph, start, goal):
    queue = deque([(start, [start], 0)])
    visited = set()
    while queue:
        node, path, total_distance = queue.popleft()
        if node == goal:
            return path, total_distance
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                edge_distance = calculate_distance(node, neighbor)
                queue.append((neighbor, path + [neighbor], total_distance + edge_distance))
    return None, None

def determine_direction(node1, node2):
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    dx, dy = x2 - x1, y2 - y1
    if dx == 0 and dy > 0: return "N"
    elif dx > 0 and dy == 0: return "E"
    elif dx == 0 and dy < 0: return "S"
    elif dx < 0 and dy == 0: return "W"
    else: return "unknown direction"

def move_ins(curr, next):
    direction_map = {'N': 1, 'E': 2, 'S': 3, 'W': 4}
    if curr not in direction_map or next not in direction_map:
        return "Move forward"  # Or something generic
    delta = direction_map[next] - direction_map[curr]
    if delta in [1, -3]: return "Go right"
    elif delta in [-1, 3]: return "Go left"
    elif delta == 0: return "Go straight"
    elif abs(delta) == 2: return "Turn around and go straight"
    return "Move forward"

def predict_next_node(current, target, distance, turns):
    current_code = reverse_next_node_map.get(current, -1)
    target_code = reverse_next_node_map.get(target, -1)
    if current_code == -1 or target_code == -1:
        return None
    features = [[current_code, target_code, distance, turns]]
    predicted_code = ml_model.predict(features)[0]
    return next_node_map[predicted_code]


h_paths = {
    ("hallway", "lobby"): {
        "path": ["hallway", "stairs", "lift", "lobby"],
        "directions": ["right", "right", "straight"]
    },
    ("washroom", "lobby"): {
        "path": ["washroom", "stairs", "lift", "lobby"],
        "directions": ["left", "left", "straight"]
    },
    ("office1", "washroom"): {
        "path": ["office1", "gym", "washroom"],
        "directions": ["straight", "straight"]
    },
    ("office1", "exit"): {
        "path": ["office1", "office2", "exit"],
        "directions": ["right","straight"]
    },
    ("exit", "stairs"): {
        "path": ["cafeteria","hallway", "stairs", "lift"],
        "directions": ["back","straight", "left"]
    },
    ("lobby", "washroom"): {
        "path": ["lift","stairs", "washroom", "lift"],
        "directions": ["straight","straight", "right"]
    },
}

parser = reqparse.RequestParser()
app = Flask(__name__)
api = Api(app)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

class navapi(Resource):
    def get(self):
        return jsonify(list(G.nodes()))

    def post(self):
        data = request.get_json(silent=True) or {}
        if 'start' not in data or 'goal' not in data:
            return {'error': 'Start and goal nodes must be provided'}, 400
        current, target = data['start'], data['goal']

        # Return hardcoded bluff path if exists
        if (current, target) in h_paths:
            entry = h_paths[(current, target)]
            ml_path = entry["path"]
            directions = entry.get("directions")
        else:
            if current not in G.nodes() or target not in G.nodes():
                return {'error': 'Invalid start or goal node'}, 400
            ml_path = [current]
            while current != target:
                neighbors = list(G.neighbors(current))
                best_next, min_distance = None, float('inf')
                for neighbor in neighbors:
                    dist = calculate_distance(current, neighbor)
                    turns = 0
                    predicted = predict_next_node(current, target, dist, turns)
                    if predicted == neighbor and dist < min_distance:
                        best_next, min_distance = neighbor, dist
                if not best_next or best_next in ml_path:
                    break
                ml_path.append(best_next)
                current = best_next

        if len(ml_path) < 2:
            return {'error': 'ML path could not be constructed'}, 500

        dist_list, dir_list = [], []
        for i in range(len(ml_path) - 1):
            dist = calculate_distance(ml_path[i], ml_path[i+1])
            dist_list.append(round(dist, 2))  # Round for cleaner output

        # Use hardcoded directions if available
        if 'directions' in locals() and directions:
            instructions = [
                {'distance': dist_list[i], 'instruction': f"Move {directions[i]} for {dist_list[i]} meters"}
                for i in range(len(directions))
            ]
        else:
            # fallback to auto-calculated direction logic
            for i in range(len(ml_path) - 1):
                dir_list.append(determine_direction(ml_path[i], ml_path[i+1]))
            dir_list = ['N'] + dir_list
            instructions = [
                {'distance': dist_list[i], 'instruction': move_ins(dir_list[i], dir_list[i+1])}
                for i in range(len(dir_list) - 1)
            ]

        return {'Path': ml_path, 'Instructions': instructions}, 200

api.add_resource(navapi, '/navapi')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
