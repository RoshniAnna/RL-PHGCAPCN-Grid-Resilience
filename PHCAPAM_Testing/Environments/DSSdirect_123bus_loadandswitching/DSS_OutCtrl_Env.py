import gymnasium as gym
import random
import pickle
from gymnasium import spaces
import numpy as np
import logging
from random import sample, uniform
from math import ceil
from Environments.DSSdirect_123bus_loadandswitching.DSS_Initialize import *
from Environments.DSSdirect_123bus_loadandswitching.state_action_reward import *
from gymnasium.utils import seeding
from typing import List, Tuple
logging.basicConfig(
    format="%(asctime)s %(levelname)s: %(message)s", level=logging.WARNING
)


class DSS_OutCtrl_Env(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        print(
            "Initializing 123-bus env for outage management with generators, sectionalizing and tie switches (load control included)"
        )
        self.DSSCktObj, G_init, conv_flag = (
            initialize()
        )  # the DSSCircuit is set up and initialized
        # Set up action and observation space variables
        self.outedges = []  # to track the outage conditions(list of multi-line outage)
        self.G = G_init.copy()
        self.action_space = spaces.MultiBinary(n_actions)
        self.observation_space = spaces.Dict(
            {
                "EnergySupp": spaces.Box(low=0, high=2, shape=(1,)),
                "NodeFeat(BusVoltage)": spaces.Box(
                    low=0, high=2, shape=(len(G_init.nodes()), 3)
                ),
                "EdgeFeat(Branchflow)": spaces.Box(
                    low=0, high=2, shape=(len(G_init.edges()),)
                ),
                "Adjacency": spaces.Box(
                    low=0, high=1, shape=(len(G_init.nodes()), len(G_init.nodes()))
                ),
                "Topo_Laplacian": spaces.Box(
                    low=0, high=1, shape=(len(G_init.nodes()), len(G_init.nodes()))
                ),
                "VoltageViolation": spaces.Box(low=0, high=1000, shape=(1,)),
                "ConvergenceViolation": spaces.Box(low=0, high=1, shape=(1,)),
                "ActionMasking": spaces.Box(low=0, high=1, shape=(n_actions,)),
            }
        )
        print("Env initialized")
        # Loading the outage scenarios
        with open("Outage_scenarios_train_cleaned.pkl", "rb") as f:
             self.outage_scenarios = pickle.load(f)

    def step(self, action):
        # Getting observation before action is executed
        observation = get_state(
            self.DSSCktObj, self.G, self.outedges
        )  # function to get state of the network
        # Executing the switching action
        try:
            self.DSSCktObj, self.G = take_action(
                action, self.outedges
            )  # function to implement the action
            # Getting observation after action is taken
            obs_post_action = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = get_reward(obs_post_action)  # function to calculate reward
        except:
            obs_post_action = get_state(self.DSSCktObj, self.G, self.outedges)
            reward = np.array([-1.0])
            # get_reward(obs_post_action)
        done = True
        info = {"is_success": done, "episode": {"r": reward, "l": 1}}
        logging.info("Step success")
        return obs_post_action, reward[0], done, False, info

    def reset(self, seed=0):
        # In reset function we simulate different line outage scenarios
        logging.info("Resetting environment...")
        self.DSSCktObj, G_init, conv_flag = initialize()  # initial set up
        self.G = G_init.copy()

        # Select outage scenarios randomly from predefined training set
        scenario = random.choice(self.outage_scenarios)
        out_edges = scenario["OutageEdges"]   

        try:
            for (u, v) in out_edges:
                branch_name = G_init.edges[(u, v)]["label"][0] # Read the line outages
                self.DSSCktObj.dss.Text.Command(f"Open {branch_name} term=1")  # disable the outage line in DSS circuit
            
            self.DSSCktObj.dss.Solution.Solve() # Solve the circuit

            if not self.DSSCktObj.dss.Solution.Converged():
                raise RuntimeError("DSS solver failed to converge. This should not happen in cleaned data.")

            self.G.remove_edges_from(out_edges)  # each instance of the graph includes the outage scenario
            self.outedges = out_edges  # outage scenario

            obs = get_state(self.DSSCktObj, self.G, self.outedges) # Extract the network state
            
            # Check for invalid values
            if any(np.isnan(v).any() or np.isinf(v).any() for v in obs.values() if isinstance(v, np.ndarray)):           
                raise ValueError("Observation contains NaN or Inf. This should not happen in cleaned data.")


            logging.info("Reset complete.\n")
            info = {"is_success": False, "episode": {"r": 0, "l": 0}}
            return obs, info
        
        except Exception as e:
            logging.error(f"Reset failed unexpectedly: {e}")
            raise

    def render(self, mode="human", close=False):
        pass

    def test_funcn(self, out_edges: List[Tuple[str, str]]):
        self.DSSCktObj, G_init, conv_flag = initialize()  # initial set up
        self.G = G_init.copy()
        for (u, v) in out_edges:
            branch_name = G_init.edges[(u, v)]["label"][0] # Read the line outages
            self.DSSCktObj.dss.Text.Command(f"Open {branch_name} term=1")  # disable the outage line in DSS circuit
        
        self.DSSCktObj.dss.Solution.Solve() # Solve the circuit

        if not self.DSSCktObj.dss.Solution.Converged():
            raise RuntimeError("DSS solver failed to converge. This should not happen in cleaned data.")

        self.G.remove_edges_from(out_edges)  # each instance of the graph includes the outage scenario
        self.outedges = out_edges  # outage scenario

        obs = get_state(self.DSSCktObj, self.G, self.outedges) # Extract the network state
            
        # Check for invalid values
        if any(np.isnan(v).any() or np.isinf(v).any() for v in obs.values() if isinstance(v, np.ndarray)):           
            raise ValueError("Observation contains NaN or Inf. This should not happen in cleaned data.")
        
        return obs