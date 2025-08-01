# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Robot Inspection Environment
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="Isaac-Inspection-Camera-Direct-v0",
    entry_point=f"{__name__}.inspection_env:Isaac3dinspectionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.inspection_env:Isaac3dinspectionEnvCfg",
        #"rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_camera_ppo_cfg.yaml", #rl_games_states_ppo_cfg
         "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_states_ppo_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_cfg_continous.yaml",
         "skrl_cfg_entry_point": f"{agents.__name__}:skrl_camera_ppo_rnn.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_states_ppo_cfg.yaml"
    },
)
