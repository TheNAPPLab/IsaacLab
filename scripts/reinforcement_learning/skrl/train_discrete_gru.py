import argparse
import os
import torch
import torch.nn as nn
import sys
from datetime import datetime
import numpy as np

from isaaclab.app import AppLauncher
is_eval = False
# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

# parser.add_argument("--task", type=str, default="Isaac-Cartpole-RGB-Camera-Direct-v0", help="Name of the task.")
parser.add_argument("--task", type=str, default="Isaac-Inspection-Camera-Direct-v0", help="Name of the task.")
# parser.add_argument("--task", type=str, default="Isaac-Velocity-Rough-Anymal-C-Direct-v0", help="Name of the task.")
# append AppLauncher cli args

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
_headless = True
args_cli = parser.parse_args()
args_cli.enable_cameras =  True
args_cli.headless = _headless
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from skrl.agents.torch.ppo import PPO_RNN as PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, CategoricalMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab.utils.dict import print_dict
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
# sys.argv.append("--headless")
sys.argv.append("--enable_cameras")


set_seed(42)

class Shared(CategoricalMixin, DeterministicMixin, Model):
    def __init__(self,
                observation_space,
                action_space,
                device,
                clip_actions=False,
                unnormalized_log_prob=True,
                num_envs=1,
                sequence_length=128, _hidden_size=128):
        Model.__init__(self, observation_space, action_space, device)
        CategoricalMixin.__init__(self, unnormalized_log_prob)
        DeterministicMixin.__init__(self, clip_actions)
        self.num_envs = num_envs
        self.sequence_length = sequence_length
        self._hidden_size = _hidden_size

        camera_space = observation_space.spaces["cameras"]
        robot_pose_space = observation_space.spaces["robot-pose"]
        map_space = observation_space.spaces["occupancy_map"]


        self.camera_shape = camera_space.shape
        self.robot_pose_dim = robot_pose_space.shape[0]
        self.map_shape = map_space.shape
        # print(f"DEBUG: camera_shape: {self.camera_shape}")
        # print(f"DEBUG: robot_pose_dim: {self.robot_pose_dim}")

        self.camera_flat_size = np.prod(self.camera_shape).item()
        self.map_flat_size = np.prod(self.map_shape).item()
        self.total_obs_size = self.camera_flat_size + self.robot_pose_dim + self.map_flat_size
        # print(f"DEBUG: camera_flat_size: {self.camera_flat_size}")
        # print(f"DEBUG: total_obs_size: {self.total_obs_size}")

        self.camera_features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.camera_shape[-1], out_channels=32,
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
        self.map_features_extractor = nn.Sequential(
            nn.Conv2d(in_channels=self.map_shape[-1], out_channels=16, kernel_size=8, stride=4),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1),
            nn.ELU(),
            nn.Flatten()
        )

        with torch.no_grad():
            #permute(STATES, (0, 3, 1, 2)) 
            dummy_camera_input = torch.zeros(1, self.camera_shape[-1], *self.camera_shape[:2])
            camera_cnn_output_dim = self.camera_features_extractor(dummy_camera_input).shape[1]

            dummy_map_input = torch.zeros(1, self.map_shape[-1], *self.map_shape[:2])
            map_cnn_output_dim = self.map_features_extractor(dummy_map_input).shape[1]
            # print(f"DEBUG: cnn_output_dim: {cnn_output_dim}")


        self.gru_input_size = camera_cnn_output_dim + self.robot_pose_dim + map_cnn_output_dim
        self.gru_hidden_size = self._hidden_size #H output size of GRU
        self.gru_num_layers = 1
        # print(f"DEBUG: gru_input_size: {self.gru_input_size}")

        self.gru = nn.GRU(input_size=self.gru_input_size,
                          hidden_size=self.gru_hidden_size,
                          num_layers=self.gru_num_layers,
                          batch_first=True)  # batch_first -> (batch, sequence, features)
        #output heads
        self.policy_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, self._hidden_size),
            nn.ELU(),
            nn.Linear(self._hidden_size, self.num_actions)
        )

        self.value_head = nn.Sequential(
            nn.Linear(self.gru_hidden_size, self._hidden_size),
            nn.ELU(),
            nn.Linear(self._hidden_size, 1)
        )
        # Action Head, MU and STD
    def get_specification(self) -> dict:
        return {
                "rnn": {
                        "sequence_length": self.sequence_length,
                        "sizes": [(self.gru_num_layers, 1, self.gru_hidden_size)],
                    }
                }
            
    def unflatten_observations(self, flat_obs):
        """
        Manually unflatten the observation tensor back to camera and robot pose components.
        """
        batch_size = flat_obs.shape[0]
        
        # print(f"DEBUG: Unflattening obs with shape: {flat_obs.shape}")
        # print(f"DEBUG: Expected total size: {self.total_obs_size}")
        
        # Verify the flattened observation has the expected size
        if flat_obs.shape[1] != self.total_obs_size:
            print(f"WARNING: Observation size mismatch! Expected {self.total_obs_size}, got {flat_obs.shape[1]}")
        
        # Split camera and robot pose data
        cam_start, cam_end = 0, self.camera_flat_size
        map_start, map_end = cam_end, cam_end + self.map_flat_size
        pose_start = map_end

        # pose_start, pose_end = cam_end, cam_end + self.robot_pose_dim
        # map_start = pose_end

        camera_flat = flat_obs[:, cam_start:cam_end]
        map_flat = flat_obs[:, map_start:map_end]
        robot_pose = flat_obs[:, pose_start:]
        # print(f"DEBUG: camera_flat shape: {camera_flat.shape}")
        # print(f"DEBUG: robot_pose shape: {robot_pose.shape}")
        
        # # Debug: Print some values to verify splitting (especially useful with all-ones robot pose)
        # print(f"DEBUG: First few camera values: {camera_flat[0, :5]}")
        # print(f"DEBUG: Robot pose values: {robot_pose[0]}")
        
        # Reshape camera data from flat to (batch, height, width, channels)
        camera_reshaped = camera_flat.view(batch_size, *self.camera_shape)
        camera_obs = camera_reshaped.permute(0, 3, 1, 2)

        map_reshaped = map_flat.view(batch_size, *self.map_shape)
        map_obs = map_reshaped.permute(0, 3, 1, 2)

        # print(f"DEBUG: camera_obs final shape: {camera_obs.shape}")

        return camera_obs, robot_pose, map_obs

    def act(self, inputs, role):
        if role == "policy":
            act = CategoricalMixin.act(self, inputs, role)
            # print(f"DEBUG: Action shape: {act.shape}")
            return act
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)
        
    def compute(self, inputs, role):
        states = inputs["states"]
        terminated = inputs.get("terminated", None)
        hidden_states = inputs["rnn"][0]
        camera_obs, robot_pose, map_obs = self.unflatten_observations(states)

        # camera_obs = states["cameras"].permute(0, 3, 1, 2)  # (batch, channels, height, width)
        image_features = self.camera_features_extractor(camera_obs)
        map_features = self.map_features_extractor(map_obs)

        combined_features = torch.cat((image_features, robot_pose, map_features), dim=1)  # (batch, cnn_output_dim + robot_pose_dim)

        if self.training:
            # just return dummy action to debug sim
            # return torch.zeros((self.num_envs, self.num_actions), device=self.device), {"rnn": [hidden_states]}
            rnn_input = combined_features.view(-1, self.sequence_length, combined_features.shape[-1])
            hidden_states = hidden_states.view(self.gru_num_layers, -1, self.sequence_length, self.gru_hidden_size)
            # get the hidden states corresponding to the initial sequence
            hidden_states = hidden_states[:, :, 0, :].contiguous()

            if terminated is not None and torch.any(terminated):
                rnn_outputs = []
                terminated = terminated.view(-1, self.sequence_length)

                indexes = [0] + (terminated[:, :-1].any(dim=0).nonzero(as_tuple=True)[0] + 1).tolist() + [self.sequence_length]

                for i in range(len(indexes) - 1):
                    i0, i1 = indexes[i], indexes[i+1]
                    rnn_output, hidden_states = self.gru(
                        rnn_input[:, i0:i1, :], hidden_states
                    )
                    hidden_states[:, terminated[:, i1 - 1], :] = 0
                    rnn_outputs.append(rnn_output)
                rnn_output = torch.cat(rnn_outputs, dim=1)
            else:
                rnn_output, hidden_states = self.gru(rnn_input, hidden_states)
        else:
            rnn_input = combined_features.unsqueeze(1)
            rnn_output, hidden_states = self.gru(rnn_input, hidden_states)


        #flatten  rnn output
        # flat_gru_output = gru_output.reshape(-1, self.gru_hidden_size)
        rnn_output = torch.flatten(rnn_output, start_dim=0, end_dim=1)

        if role == "policy":
            action_logits = self.policy_head(rnn_output)
            if torch.isnan(action_logits).any() or torch.isinf(action_logits).any():
                print("!!! ERROR: NaN or Inf detected in action_logits from compute() !!!")
                print(action_logits)
            return  action_logits, {"rnn": [hidden_states]}
        elif role == "value":
            value_estimate = self.value_head(rnn_output)
            return value_estimate, {"rnn": [hidden_states]}



# env = load_isaaclab_env(task_name="Isaac-Inspection-Camera-Direct-v0", cfg=env_cfg, render_mode=None)
# env = load_isaaclab_env(task_name="Isaac-Inspection-Camera-Direct-v0", headless=False, num_envs=1)
# env = wrap_env(env)

env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
)
env = gym.make(args_cli.task, cfg=env_cfg)
env = wrap_env(env)

device = env.device
sequence_length = 128
rollout_length = sequence_length * 8

memory = RandomMemory(memory_size=rollout_length, num_envs=1, device=device)

models = {}
models['policy'] = Shared(env.observation_space, env.action_space, env.device, sequence_length=sequence_length)
models['value'] = models["policy"]  # Shared(env.observation_space, env.action_space, env.device)

cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = rollout_length  # memory_size
cfg["learning_epochs"] = 8
cfg["mini_batches"] = 4  #
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-4 #
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.008}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0.0
# cfg["rewards_shaper"] = lambda rewards, *args, **kwargs: rewards * 1.0
cfg["time_limit_bootstrap"] = True

cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)


log_root_path = os.path.join("logs", "skrl", "3DInspection_direct")
log_root_path = os.path.abspath(log_root_path)

experiment_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_ppo_gru_128"

print(f"[INFO] Logging experiment in directory: {log_root_path}")

log_dir = os.path.join(log_root_path, experiment_name)
print(f"[INFO] skrl will log this experiment in: {log_dir}")

os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)



cfg["experiment"]["write_interval"] = 1000
# cfg["experiment"]["name"] = "IsaacLab-scripts_reinforcement_learning_skrl"
cfg["experiment"]["checkpoint_interval"] = 10_000
cfg["experiment"]["directory"] = log_root_path
cfg["experiment"]["experiment_name"] = experiment_name
cfg["experiment"]["wandb"] = True  # Disable wandb in evaluation mode

# try:
#     # Save agent configuration
#     dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), cfg)
#     dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), cfg)
#     print(f"[INFO] Configuration saved to: {log_dir}/params/")
# except Exception as e:
#     print(f"[WARNING] Could not save configuration: {e}")

agent = PPO(models=models, 
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,)
# path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1862000.pt"
# agent.load(path)
cfg_trainer ={"timesteps": 5_000_000,  # total timesteps to train the agent
                "headless": _headless,
               }#  "stochastic_evaluation": False

trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
print("[INFO] Starting training...")
print_dict(cfg_trainer, nesting=4)
if is_eval: 
    print("[INFO] Running in evaluation mode. No training will be performed.")
    # path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1860000.pt"
    # agent.load(path)
    trainer.eval()
else:
    # path = "logs/skrl/3DInspection_direct/2025-08-03_20-01-28_ppo_gru_128/checkpoints/agent_1862000.pt"
    # agent.load(path)
    trainer.train()

