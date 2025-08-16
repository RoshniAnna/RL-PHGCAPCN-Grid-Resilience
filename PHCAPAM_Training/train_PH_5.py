# -*- coding: utf-8 -*-
import re
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
import warnings
import pickle
import math
import os
import wandb
import glob
import signal
from wandb.integration.sb3 import WandbCallback
from Configs.training_config import get_training_config


warnings.filterwarnings("ignore", category=FutureWarning, message="adjacency_matrix will return a scipy.sparse array instead of a matrix in Networkx 3.0")

os.environ["WANDB_DIR"] = os.environ.get("SCRATCH_DIR", os.getcwd())

def learning_rate_schedule(initial_value: float) -> Callable[[float], float]:

    def func(progress_remaining: float) -> float:
        decay_rate = 0.001
        return initial_value * math.exp((-progress_remaining**2*decay_rate))

    return func


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = DSS_OutCtrl_Env()
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

def extract_step_number(filename):
    match = re.search(r"(\d+)_steps\.zip", filename)
    return int(match.group(1)) if match else -1
    
    
if __name__ == '__main__':
    # Get the scratch directory from environment variable or default to current directory
    SCRATCH_DIR = os.environ.get("SCRATCH_DIR", os.getcwd())
    ckpt_dir = os.path.join(SCRATCH_DIR, "ckpts_PerHom")
    tensorboard_log_dir = os.path.join(SCRATCH_DIR, "TensorboardLogs")
    save_prefix = os.path.join(SCRATCH_DIR, "models", "PH_lap")
    ckpt_name = "PH_lap"

    # Make sure the directories exist
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_prefix), exist_ok=True)    

    env_size = 123

    if env_size == 13:
        from Environments.DSSdirect_13bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_34.Feature_Extractor import CustomGNN
        from Policies.bus_13.CustomPolicies import ActorCriticGCAPSPolicy
    elif env_size == 34: # will add more conditions once the 13 bus is fixed
        from Environments.DSSdirect_34bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_34.Feature_Extractor import CustomGNN
        from Policies.bus_34.CustomPolicies import ActorCriticGCAPSPolicy
    elif env_size == 123: # will add more conditions once the 13 bus is fixed
        from Environments.DSSdirect_123bus_loadandswitching.DSS_OutCtrl_Env import DSS_OutCtrl_Env
        from Policies.bus_123.Feature_Extractor import CustomGNN
        from Policies.bus_123.CustomPolicies import ActorCriticGCAPSPolicy

    env = DSS_OutCtrl_Env()
    
    training_config = get_training_config()
    training_config.model_save = os.path.join(SCRATCH_DIR, "models/")
    training_config.logger = os.path.join(SCRATCH_DIR, "logs/")
    training_config.device = torch.device("cuda:0" if training_config.use_cuda else "cpu")   
    tb_logger_location = training_config.logger + training_config.node_encoder + "_debug"
    save_prefix = training_config.model_save + training_config.node_encoder
    
    
    # --- Check for existing checkpoint ---
    checkpoint_files = glob.glob(os.path.join(ckpt_dir, f"{ckpt_name}_*steps.zip"))
    checkpoint_files = sorted(checkpoint_files, key=extract_step_number)
    
    latest_checkpoint = checkpoint_files[-1] if checkpoint_files else None
    print(f"Latest checkpoint file is: {latest_checkpoint}", flush=True)

    
    config = {
        "policy_type": "Capam_PH",
        "total_timesteps": 5000000,
        "env_name": "123_Bus",
    }
    run = wandb.init(
        project="sb3",
        name=f'TDA_PH_1',
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=False,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )
    
 
    
    # features_dim=training_config.features_dim
    features_dim = (128)
    

    if training_config.node_encoder == "CAPAM":

        policy_kwargs = dict(
            features_extractor_class=CustomGNN,
            features_extractor_kwargs=dict(features_dim=training_config.features_dim,node_dim=3),
            #activation_fn=torch.nn.Sigmoid,
            net_arch=dict(
                    pi=
                    [training_config.features_dim,
                     2*training_config.features_dim,
                     2*training_config.features_dim,
                     training_config.features_dim],
                    vi=[training_config.features_dim,
                     2*training_config.features_dim,
                     2*training_config.features_dim,
                     training_config.features_dim]
                ),
            device=training_config.device
            # optimizer_class = th.optim.RMSprop,
            # optimizer_kwargs = dict(alpha=0.89, eps=rms_prop_eps, weight_decay=0)
        )

    else:

        policy_kwargs = dict(
            # features_extractor_class=CustomGNN,
            # features_extractor_kwargs=dict(features_dim=training_config.features_dim, node_dim=3),
            # activation_fn=torch.nn.Sigmoid,
            net_arch=[
                dict(
                    pi=
                    [training_config.features_dim,
                     2 * training_config.features_dim,
                     2 * training_config.features_dim,
                     training_config.features_dim],
                    vi=[training_config.features_dim,
                        2 * training_config.features_dim,
                        2 * training_config.features_dim,
                        training_config.features_dim]
                )]
            # device=training_config.device
            # optimizer_class = th.optim.RMSprop,
            # optimizer_kwargs = dict(alpha=0.89, eps=rms_prop_eps, weight_decay=0)
        )

    # --- Define callbacks ---
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path=ckpt_dir,
        name_prefix=ckpt_name,
        save_replay_buffer=True,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=ckpt_dir,
        log_path=ckpt_dir,
        eval_freq=5000,
        deterministic=True,
        render=False,
    )
    callbacks = CallbackList([
        checkpoint_callback,
        eval_callback,
        WandbCallback(
            gradient_save_freq=5000,
            model_save_path="wandb/" + ckpt_dir,
            verbose=2,
        )
    ])
    

    # --- Instantiate model (load or new) ---
    if latest_checkpoint is not None:
        print(f"Resuming training from checkpoint: {latest_checkpoint}")
        model = PPO.load(latest_checkpoint, env=env, tensorboard_log=tensorboard_log_dir)
    else:
        model = PPO(
            policy=ActorCriticGCAPSPolicy if training_config.node_encoder == "CAPAM" else "MultiInputPolicy",
            env=env,
            tensorboard_log=tensorboard_log_dir,
            policy_kwargs=policy_kwargs,
            verbose=1,
            n_steps=training_config.n_steps,
            batch_size=training_config.batch_size,
            gamma=training_config.gamma,
            learning_rate=learning_rate_schedule(training_config.learning_rate),
            ent_coef=training_config.ent_coef,
        )

    # --- Save on termination ---
    def save_on_exit(signum, frame):
        print("Received termination signal, saving model.")
        model.save(save_prefix + "_interrupted")
        artifact = wandb.Artifact("PH_interrupted_model", type="model")
        artifact.add_file(save_prefix + "_interrupted.zip")
        wandb.log_artifact(artifact)
        run.finish()
        exit(0)

    signal.signal(signal.SIGTERM, save_on_exit)

    # --- Begin learning ---
    model.learn(
        total_timesteps=1230000,
        callback=callbacks,
        reset_num_timesteps=False  # <-- Critical to avoid resetting to 0
    )

    model.save(save_prefix+"_final")

    # Optionally log final model to wandb
    artifact = wandb.Artifact("PH_final_model", type="model")
    artifact.add_file(save_prefix + "_final.zip")
    wandb.log_artifact(artifact)
    run.finish()