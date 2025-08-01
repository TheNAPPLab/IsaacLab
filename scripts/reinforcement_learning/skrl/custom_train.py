import os
import torch
import torch.nn as nn
import sys
from datetime import datetime

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)

import isaaclab_tasks  # noqa: F401

from source.isaaclab_rl.isaaclab_rl.skrl import SkrlVecEnvWrapper
from source.isaaclab_tasks.isaaclab_tasks.direct.robot_inspection.inspection_cfg import Isaac3dinspectionEnvCfg
set_seed(42)
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=True,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, initial_log_std=0.0, 
                 reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        self.features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=observation_space.shape[2], out_channels=32,
                    kernel_size=8, stride=4, padding=0),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, 
                     kernel_size=4, stride=2, padding=0),
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, 
                     kernel_size=3, stride=1, padding=0),
            nn.ELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            #permute(STATES, (0, 3, 1, 2)) 
            sample_input = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
            features_size = self.features_extractor(sample_input).shape[1]
        
        self.policy_net = nn.Sequential(
            nn.Linear(features_size, 512),
            nn.ELU()
        )

        self.value_net = nn.Sequential(
            nn.Linear(features_size, 64),
            nn.ELU()
        )
        # Action Head, MU and STD
        self.mean_layer = nn.Sequential(
            nn.Linear(64, action_space.shape[0]),
            nn.Tanh()
        )
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        # Value Head
        self.value_layer = nn.Linear(512, 1)
        self._shared_features = None

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        states = inputs["states"].permute(0, 3, 1, 2)

        if role == "policy":
            features = self.features_extractor(states)
            self.shared_output = features
            policy_output = self.policy_net(features)
            mean = self.mean_layer(policy_output)
            return mean, self.log_std_parameter, {}
        
        elif role == "value":
            if self._shared_features is not None:
                features = self._shared_features
                self._shared_features = None
            else:
                features = self.features_extractor(states)
            value_output = self.value_net(features)
            value = self.value_layer(value_output)
            return value, {}

def main():
    env_cfg = Isaac3dinspectionEnvCfg()
    #load and wrap the Isaac Lab environment
    
    env = load_isaaclab_env(task_name="Isaac-Inspection-Camera-Direct-v0", cfg=env_cfg, render_mode=None)
    # env = wrap_env(env)
    env = SkrlVecEnvWrapper(env, ml_framework="torch")
    device = env.device

    memory = RandomMemory(memory_size=-1, num_envs=env.num_envs, device=device)

    models = {}
    models['policy'] = Shared(env.observation_space, env.action_space, env.device)
    models['value'] =  models["policy"] # Shared(env.observation_space, env.action_space, env.device)

    cfg = PPO_DEFAULT_CONFIG.copy()
    cfg["rollouts"] =  64 # memory_size
    cfg["learning_epochs"] = 8
    cfg["mini_batches"] = 4 # 16 * 512 / 8192
    cfg["discount_factor"] = 0.99
    cfg["lambda"] = 0.95
    cfg["learning_rate"] = 3e-4
    cfg["learning_rate_scheduler"] = KLAdaptiveRL
    cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
    cfg["random_timesteps"] = 0
    cfg["learning_starts"] = 0
    cfg["grad_norm_clip"] = 1.0
    cfg["ratio_clip"] = 0.2
    cfg["value_clip"] = 0.2
    cfg["clip_predicted_values"] = True
    cfg["entropy_loss_scale"] = 0.0
    cfg["value_loss_scale"] = 2.0
    cfg["kl_threshold"] = 0.0
    cfg["rewards_shaper"] = None
    cfg["time_limit_bootstrap"] = True
    cfg["state_preprocessor"] = RunningStandardScaler
    cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    cfg["value_preprocessor"] = RunningStandardScaler
    cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    # logging to TensorBoard and write checkpoints (in timesteps)
    

    log_root_path = os.path.join("logs", "skrl", "3DInspection_direct")
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_torch_CustomTrain"
    log_dir = os.path.join(log_root_path, log_dir)

    os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    
    
    
    cfg["experiment"]["write_interval"] = 16
    cfg["experiment"]["name"] = "CustomTrain"
    cfg["experiment"]["checkpoint_interval"] = 80
    cfg["experiment"]["directory"] = log_root_path
    cfg["experiment"]["experiment_name"] = os.path.basename(log_dir)
    cfg["experiment"]["wandb"] = True
    
    try:
        # Save agent configuration
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), cfg)
        dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), cfg)
        print(f"[INFO] Configuration saved to: {log_dir}/params/")
    except Exception as e:
        print(f"[WARNING] Could not save configuration: {e}")

    agent = PPO(models=models, 
                memory=memory,
                cfg=cfg,
                observation_space=env.observation_space,
                action_space=env.action_space,
                device=env.device)
    cfg_trainer ={"timesteps": 5_000_000,
                   "headless": True,
                   "close_environment_at_exit": True}
   
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    print("[INFO] Starting training...")
    print_dict(cfg_trainer, nesting=4)
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("[INFO] Training interrupted by user")
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        raise
    finally:
        print("[INFO] Closing environment...")
        env.close()

if __name__ == "__main__":
    main()