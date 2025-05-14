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
    "entrance": (0, 0),
    "Intersection_01": (0, 5),
    "Intersection_02": (0, 10),
    "Intersection_03": (-5, 10),
    "Intersection_04": (-5, 15),
    "Intersection_05": (-5, 20),
    "Intersection_06": (0, 20),
    "Intersection_07": (5, 20),
    "Intersection_08": (5, 15),
    "Intersection_09": (5, 10),
    "lift 01": (2, 5),
    "lift 02": (0, 21),
    "tpo": (-6, 10),
    "washroom": (-6, 20),
    "classroom 01": (-6, 15),
    "classroom 02": (-5, 21),
    "classroom 03": (5, 21),
    "classroom 04": (6, 20),
    "classroom 05": (6, 15),
    "classroom 06": (6, 10),
}

edges = [
    ("entrance", "Intersection_01"),
    ("Intersection_01", "lift 01"),
    ("Intersection_01", "Intersection_02"),
    ("Intersection_02", "Intersection_03"),
    ("Intersection_02", "Intersection_09"),
    ("Intersection_03", "Intersection_04"),
    ("Intersection_04", "Intersection_05"),
    ("Intersection_05", "Intersection_06"),
    ("Intersection_06", "Intersection_07"),
    ("Intersection_07", "Intersection_08"),
    ("Intersection_08", "Intersection_09"),
    ("Intersection_03", "tpo"),
    ("Intersection_05", "washroom"),
    ("Intersection_06", "lift 02"),
    ("Intersection_04", "classroom 01"),
    ("Intersection_05", "classroom 02"),
    ("Intersection_07", "classroom 03"),
    ("Intersection_07", "classroom 04"),
    ("Intersection_08", "classroom 05"),
    ("Intersection_09", "classroom 06"),
]


# Add nodes and edges to the graph
G.add_nodes_from(coordinates.keys())
G.add_edges_from(edges)

# Function to calculate distance between two nodes (coordinates)
def calculate_distance(node1, node2):
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
# Perform BFS to find the shortest path and calculate distances and directions
def bfs_shortest_path(graph, start, goal):
    queue = deque([(start, [start], 0)])  # Include distance in the queue
    visited = set()

    while queue:
        node, path, total_distance = queue.popleft()

        if node == goal:
            return path, total_distance

        if node not in visited:
            visited.add(node)
            neighbors = list(graph.neighbors(node))
            for neighbor in neighbors:
                edge_distance = calculate_distance(node, neighbor)
                queue.append((neighbor, path + [neighbor], total_distance + edge_distance))

    return None, None

def determine_direction(node1, node2):
    x1, y1 = coordinates[node1]
    x2, y2 = coordinates[node2]
    dx = x2 - x1
    dy = y2 - y1
    if(dx == 0 and dy >0):
        return "N"
    elif(dx >0 and dy == 0):
        return "E"
    elif(dx == 0 and dy <0): 
        return "S"
    elif (dx <0 and dy == 0):
        return "W"
    else:
        return "unknown direction"
    

def move_ins(curr,next):
    dict = {'N':1,'E':2,'S':3,'W':4}
    if(dict[next]-dict[curr]==1 or dict[next]-dict[curr]==-3):
        return "Go right"
    elif(dict[next]-dict[curr]==-1 or dict[next]-dict[curr]==3):
        return "Go left"
    elif(dict[next]-dict[curr]==0):
        return "Go straight"
    elif(abs(dict[next]-dict[curr])==2):
        return "Turn around and go straight"
    else:
        return "unknow move"

def predict_next_node(current, target, distance, turns):
    current_code = reverse_next_node_map.get(current, -1)
    target_code = reverse_next_node_map.get(target, -1)

    if current_code == -1 or target_code == -1:
        return None  # Unknown node

    features = [[current_code, target_code, distance, turns]]
    predicted_code = ml_model.predict(features)[0]
    return next_node_map[predicted_code]


parser = reqparse.RequestParser()
app = Flask(__name__)
api = Api(app)
app.config.from_object(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

class navapi(Resource):
    def get(self):
        nodes_list=list(G.nodes())
        return jsonify(nodes_list)
    def post(self):
        # Parse JSON body (will be None if invalid JSON)
        data = request.get_json(silent=True) or {}

        # Required fields check
        if 'start' not in data or 'goal' not in data:
            return {'error': 'Start and goal nodes must be provided'}, 400

        current = data['start']
        target = data['goal']

        # Node validity check
        if current not in G.nodes() or target not in G.nodes():
            return {'error': 'Invalid start or goal node'}, 400

        # Build ML-assisted path
        ml_path = [current]
        while current != target:
            neighbors = list(G.neighbors(current))
            best_next = None
            min_distance = float('inf')

            for neighbor in neighbors:
                dist = calculate_distance(current, neighbor)
                turns = 0
                predicted = predict_next_node(current, target, dist, turns)
                if predicted == neighbor and dist < min_distance:
                    min_distance = dist
                    best_next = neighbor

            if not best_next or best_next in ml_path:
                break
            ml_path.append(best_next)
            current = best_next

        if len(ml_path) < 2:
            return {'error': 'ML path could not be constructed'}, 500

        # Generate instructions
        dist_list = []
        dir_list = []
        for i in range(len(ml_path) - 1):
            n1, n2 = ml_path[i], ml_path[i+1]
            dist_list.append(calculate_distance(n1, n2))
            dir_list.append(determine_direction(n1, n2))

        # Prepend a default start direction if needed
        dir_list = ['N'] + dir_list

        instructions = []
        for i in range(len(dir_list)-1):
            move = move_ins(dir_list[i], dir_list[i+1])
            instructions.append({
                'distance': dist_list[i],
                'instruction': move
            })

        # Return plain dict + status code
        return {
            'Path': ml_path,
            'Instructions': instructions
        }, 200

        

api.add_resource(navapi, '/navapi')
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)