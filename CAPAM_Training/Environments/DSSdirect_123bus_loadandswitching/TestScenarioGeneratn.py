"""
Using this file I am generating a predefined set of test scenarios.
Currently I have only considered the number and location of outage as uncertain or varying.
"""

import random
import pickle
from math import ceil
import networkx as nx
from DSS_Initialize import initialize

# Parameters
Total_scenarios = 100
Num_centers = 25 # well-spaced diverse clusters considering dia = 21
Max_attempts = 5000  # safeguard

# Set seed for reproducibility
SEED =  999
random.seed(SEED)

# Initialize base graph
_, G_init, _ = initialize()
G_base = G_init.copy()
node_list = list(G_init.nodes())

# Parameters
max_rad = nx.diameter(G_init)  # maximum radius to create a subgraph
max_percfail = 0.3
# Pick 25 diverse center nodes around which the subgraphs are formed
diverse_centers = random.sample(node_list, Num_centers) 

# Container for scenarios
test_scenarios = []
test_seen = set()
scenario_id = 1
attempts = 0

def invar_sort_edge_list(edges): # this give order invariant and sorted list
    return tuple(sorted((min(u, v), max(u, v)) for (u, v) in edges))

def generate_outage(center_node, radius, severity):
    Gsub = nx.ego_graph(G_base, center_node, radius=radius, undirected=False) # Form subgraph
    sub_edges = list(Gsub.edges())
    if not sub_edges:
        return None  # No edges to fail
    num_out = max(1, ceil(len(sub_edges) * severity))
    out_edges = random.sample(sub_edges, k=num_out)
    return out_edges

# Main loop
# Load training keys
with open("Outage_scenarios_train.pkl", "rb") as f:
    training_scenarios = pickle.load(f)

training_keys = set(invar_sort_edge_list(s["OutageEdges"]) for s in training_scenarios)
    
while len(test_scenarios) < Total_scenarios and attempts < Max_attempts:
        center = random.choice(diverse_centers)
        radius = ceil(random.uniform(0, max_rad / 3))
        severity = random.uniform(0, max_percfail)

        outage_edges = generate_outage(center, radius, severity)
        if not outage_edges:
            attempts += 1
            continue        


        key = invar_sort_edge_list(outage_edges)
        if key in training_keys or key in test_seen: # Ensure that the test scenarios are unique and also not present in test set
            attempts += 1
            continue
        
        test_seen.add(key)
        test_scenarios.append({
            "ScenarioID": scenario_id,
            "OutageEdges": outage_edges
        })
        scenario_id += 1
        attempts += 1  
        print(f"Scenario no:{scenario_id}")

# Check and save
if len(test_scenarios) < Total_scenarios:
    raise RuntimeError(f"Only generated {len(test_scenarios)} unique test scenarios after {attempts} attempts.")
    
with open("Outage_scenarios_test.pkl", "wb") as f:
    pickle.dump(test_scenarios, f)
    

print(f"Successfully saved {len(test_scenarios)} unique test scenarios after {attempts} attempts.")