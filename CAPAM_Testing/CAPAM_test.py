"""
In this file I am testing the model for the preset test scenarios with varying number and location of outages
"""
import sys
import os
import sys, os
# Get this script's directory
base_dir = os.path.dirname(__file__)
# Add to sys.path for imports
sys.path.insert(0, base_dir)

import numpy as np
import pandas as pd
import torch
import ast
from stable_baselines3 import PPO
from Environments.DSSdirect_123bus_loadandswitching.DSS_OutCtrl_Env import *
from Policies.bus_123.Feature_Extractor import CustomGNN
from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

env = DSS_OutCtrl_Env()
model_path = os.path.join(base_dir, "Trained Models", "best_model.zip") # CHANGE MODEL HERE
model = PPO.load(model_path, env=env)

# Load pre-generated 100 scenarios
scenario_file = os.path.join(base_dir, "Test Dataset", "Outage_scenarios_test2_cleaned.pkl") # CHANGE TEST SAMPLE HERE
# Loading the test outage scenarios
with open(scenario_file, "rb") as f:
     test_scenarios = pickle.load(f)

results = []

for outage_scenario in test_scenarios:
    scenario_id = outage_scenario["ScenarioID"]
    outage_edges = outage_scenario["OutageEdges"]
    
    try:
        obs = env.test_funcn(outage_edges)

        device = model.policy.device
        obs_tensor = {key: torch.tensor([val]).to(device) for key, val in obs.items()}

        # Model forward pass
        with torch.no_grad():
            action, values, log_probs = model.policy.forward(obs_tensor)
            action_np = action[0].detach().cpu().numpy()

        new_obs, reward, done, truncated, info = env.step(action_np)

        energy_supplied = new_obs["EnergySupp"][0]
        voltage_violation = new_obs["VoltageViolation"][0]

        results.append({
            "Scenario": scenario_id,
            "OutageEdges": outage_edges,
            "Reward": reward,
            "EnergySupplied": energy_supplied,
            "VoltageViolation": voltage_violation
        })

    except Exception as e:
        print(f"Scenario {scenario_id} failed: {e}")
        results.append({
            "Scenario": scenario_id,
            "OutageEdges": outage_edges,
            "Reward": None,
            "EnergySupplied": None,
            "VoltageViolation": None,
            "Error": str(e)
        })
    
# Save full results
df_results = pd.DataFrame(results)
df_results.to_excel(os.path.join(os.path.dirname(__file__), "CAPAM_TestSample2_BestModel_results.xlsx"), index=False) # CHANGE according to model

# Summary statistics
columns_to_analyze = ["Reward", "EnergySupplied", "VoltageViolation"]

summary = {
    "Metric": [],
    "Mean": [],
    "StdDev": []
}

for col in columns_to_analyze:
    summary["Metric"].append(col)
    summary["Mean"].append(df_results[col].mean())
    summary["StdDev"].append(df_results[col].std())

summary_df = pd.DataFrame(summary)
summary_df.to_excel(os.path.join(os.path.dirname(__file__), "CAPAM_TestSample2_BestModel_summary.xlsx"), index=False) # CHANGE according to model
