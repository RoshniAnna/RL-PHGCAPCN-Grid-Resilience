"""
Using this file I am generating a predefined set of training scenarios.
Currently I have only considered the number and location of outage as uncertain or varying.
"""

import random
import pickle
from math import ceil
import networkx as nx
from DSS_Initialize import initialize

# Parameters
Total_scenarios = 10000
Num_centers = 25 # well-spaced diverse clusters considering dia = 21
Max_attempts = 100000  # safeguard

# Set seed for reproducibility
SEED =  42
random.seed(SEED)

# Initialize base graph
_, G_init, _ = initialize()
G_base = G_init.copy()
node_list = list(G_init.nodes())

max_rad = nx.diameter(G_init)  # maximum radius to create a subgraph
max_percfail = 0.3

# Pick 25 diverse center nodes around which the subgraphs are formed
diverse_centers = random.sample(node_list, Num_centers) 

# Container for scenarios
scenarios = []
seen_sets = set()
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
while len(scenarios) < Total_scenarios and attempts < Max_attempts:
      center = random.choice(diverse_centers)
      radius = ceil(random.uniform(0, max_rad / 3)) 
      severity = random.uniform(0, max_percfail)

      outage_edges = generate_outage(center, radius, severity)
      if not outage_edges:
          attempts += 1
          continue
      key = invar_sort_edge_list(outage_edges)
      if key in seen_sets:
          attempts += 1
          continue

      seen_sets.add(key)
      scenarios.append({
        "ScenarioID": scenario_id,
        "OutageEdges": outage_edges
        })
      scenario_id += 1
      attempts += 1
      print(f"Scenario no:{scenario_id}")
# Check success
if len(scenarios) < Total_scenarios:
    raise RuntimeError(f"Only generated {len(scenarios)} unique scenarios after {attempts} attempts.")

# Save
with open("Outage_scenarios_train.pkl", "wb") as f:
    pickle.dump(scenarios, f)

print(f"Successfully saved {len(scenarios)} unique outage scenarios after {attempts} attempts.")


