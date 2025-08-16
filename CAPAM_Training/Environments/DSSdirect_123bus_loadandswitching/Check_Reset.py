"""
In this file I am checking the reset function -- see if the predefined outage scenarios generated for training works well
"""

import pickle
import logging
import numpy as np
from DSS_Initialize import initialize
from Check_state_action_reward import get_state
import networkx as nx

logging.basicConfig(level=logging.INFO)

# def invar_sort_edge_list(edges):
#     return tuple(sorted((min(u, v), max(u, v)) for (u, v) in edges))

def validate_scenarios(scenario_file, label):

    with open(scenario_file, "rb") as f:
        scenarios = pickle.load(f)

    success_indices = []
    nan_indices = []
    fail_indices = []
    total = len(scenarios)

    logging.info(f"Validating {label} scenarios...")

    for idx, scenario in enumerate(scenarios):
        print(idx)
        DSSCktObj, G_init, _ = initialize()
        G = G_init.copy()
        out_edges = scenario["OutageEdges"]

        try:
            for (u, v) in out_edges:
                branch_name = G_init.edges[(u, v)]["label"][0]
                DSSCktObj.dss.Text.Command(f"Open {branch_name} term=1")

            DSSCktObj.dss.Solution.Solve()

            if not DSSCktObj.dss.Solution.Converged():
                fail_indices.append(idx)
                logging.warning(f"[{label} #{idx}]  Did not converge.")
                continue

            G.remove_edges_from(out_edges)
            obs = get_state(DSSCktObj, G, out_edges)

            # Check for NaN/Inf
            if any(np.isnan(x).any() or np.isinf(x).any() for x in obs.values() if isinstance(x, np.ndarray)):
                nan_indices.append(idx)
                logging.warning(f"[{label} #{idx}]  Observation contains NaN or Inf.")
                continue

            success_indices.append(idx)

        except Exception as e:
            fail_indices.append(idx)
            logging.error(f"[{label} #{idx}]  Exception occurred: {e}")

    # Summary
    logging.info(f"\n Summary for {label} scenarios:")
    logging.info(f"Total: {total}")
    logging.info(f"Successful: {len(success_indices)}")
    logging.info(f"NaN/Inf in observations: {len(nan_indices)}")
    logging.info(f"Did not converge or crashed: {len(fail_indices)}")

    # Print indices
    logging.info(f"\nSuccessful scenario indices: {success_indices}")
    logging.info(f"NaN/Inf scenario indices: {nan_indices}")
    logging.info(f"Non-converging/crashed scenario indices: {fail_indices}")

    # Optional: save indices
    with open(f"{label.lower()}_scenario_validation_summary.pkl", "wb") as f:
        pickle.dump({
            "success": success_indices,
            "nan": nan_indices,
            "fail": fail_indices
        }, f)

#########  TRAINING SCENARIOS #############
# validate_scenarios("Outage_scenarios_train.pkl", "Training")

# Clean Scenarios
# # Load original training scenarios
# with open("Outage_scenarios_train.pkl", "rb") as f:
#     training_scenarios = pickle.load(f)

# #  Load validation summary (from the previous run)
# with open("training_scenario_validation_summary.pkl", "rb") as f:
#     validation_summary = pickle.load(f)

# # Identify indices to remove (fail + nan)
# bad_indices = set(validation_summary["fail"] + validation_summary["nan"])

# # Filter out bad scenarios
# cleaned_training = [
#     scenario for i, scenario in enumerate(training_scenarios) if i not in bad_indices
# ]

# # Renumber ScenarioID
# for i, scenario in enumerate(cleaned_training, 1):
#     scenario["ScenarioID"] = i

# # Save the cleaned scenario set
# with open("Outage_scenarios_train_cleaned.pkl", "wb") as f:
#     pickle.dump(cleaned_training, f)

# validate_scenarios("Outage_scenarios_train_cleaned.pkl", "Training_cleaned")

#########  TESTING SCENARIOS #############
# validate_scenarios("Outage_scenarios_test.pkl", "Testing")

# Clean Scenarios
# Load original training scenarios
# with open("Outage_scenarios_test.pkl", "rb") as f:
#     testing_scenarios = pickle.load(f)

# #  Load validation summary (from the previous run)
# with open("testing_scenario_validation_summary.pkl", "rb") as f:
#     validation_summary = pickle.load(f)

# # Identify indices to remove (fail + nan)
# bad_indices = set(validation_summary["fail"] + validation_summary["nan"])

# # Filter out bad scenarios
# cleaned_testing = [
#     scenario for i, scenario in enumerate(testing_scenarios) if i not in bad_indices
# ]

# # Renumber ScenarioID
# for i, scenario in enumerate(cleaned_testing, 1):
#     scenario["ScenarioID"] = i

# # Save the cleaned scenario set
# with open("Outage_scenarios_test_cleaned.pkl", "wb") as f:
#     pickle.dump(cleaned_testing, f)
validate_scenarios("Outage_scenarios_test_cleaned.pkl", "Testing_cleaned")