import isaacgym

import os
import pickle
import hydra
import wandb
from attrdict import AttrDict
from omegaconf import OmegaConf
import gym 
from sparse_rl.agent import ddpg_agent
from sparse_rl.utils import SubprocVecEnv, get_env_params

@hydra.main(config_name='main', config_path='config')
def launch(cfg = None):
    # 1. make env
    import os, sys
    sys.path.append('/home/reed/rl/srl/envs')
    import franka_cube
    env = gym.make('FrankaPNP-v0', num_envs=cfg.num_workers, num_cameras=0, headless=True, max_vel=3, base_steps=50, auto_reset=False)
    p = env.env_params()
    env_params = {
        'gripper': p.shared_dim,
        'goal': p.goal_dim,
        'action': p.action_dim,
        'action_max': 1.0,
        'object': p.seperate_dim,
        'n_objects': p.num_goals,
        'max_timesteps': env.cfg.max_steps, 
        'compute_reward': p.compute_reward
    }
    # env_params = get_env_params(cfg.env_name, cfg.env_kwargs)
    # env = SubprocVecEnv([make_env for i in range(cfg.num_workers)])

    # 2. start wandb
    ckpt_data = None
    if cfg.wandb:
        # mode1: init 
        if cfg.wid is None:
            wandb.init(project=cfg.project, name=cfg.name, dir=hydra.utils.get_original_cwd())
        else:
            print('[DEBUG] load data from remote')
            file = wandb.restore('models/checkpoint.pkl', f'jc-bao/{cfg.project}/{cfg.wid}')
            with open(file.name, "rb") as f:
                ckpt_data = pickle.load(f)
                print('[DEBUG] load done')
            if cfg.name == ckpt_data['wandb_run_name']:
                # mode2: resume and continue
                print('[DEBUG] resume old run')
                wandb.init(project=cfg.project, id=cfg.wid, resume="allow", dir=hydra.utils.get_original_cwd())
            else:
                # mode3: resume and new
                print('[DEBUG] start new run')
                wandb.init(project=cfg.project, name=cfg.name, resume="allow", dir=hydra.utils.get_original_cwd())
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True), allow_val_change=True)
        wandb.save(".hydra/*")
        cfg = AttrDict(wandb.config)
    elif not cfg.wid is None:
        print('[WARN] wid was set but wandb is not enabled')
    else:
        cfg = AttrDict(cfg)
    
    # 3. run
    ddpg_trainer = ddpg_agent(cfg, env, env_params, ckpt_data=ckpt_data)
    ddpg_trainer.learn()

if __name__ == '__main__':
    launch()