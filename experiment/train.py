import os
import pickle
import hydra
import wandb
from omegaconf import OmegaConf
import gym, panda_gym
from sparse_rl.agent import ddpg_agent
from sparse_rl.utils import SubprocVecEnv

def get_env_params(env_name, env_kwargs):
    env = gym.make(env_name, **env_kwargs)
    obs = env.reset()
    params = {
        'gripper': obs['gripper_arr'].shape[-1],
        'goal': obs['desired_goal_arr'].shape[-1],
        'action': env.action_space.shape[0],
        'action_max': env.action_space.high[0],
        'object': obs['object_arr'].shape[-1],
        'n_objects': obs['object_arr'].shape[0],
        'max_timesteps': env._max_episode_steps, 
        'compute_reward': env.compute_reward
    }
    env.close()
    return params

@hydra.main(config_name='main', config_path='config')
def launch(cfg = None):
    # 1. make env
    env_params = get_env_params(cfg.env_name, cfg.env_kwargs)
    def make_env():
        import panda_gym
        return gym.make(cfg.env_name, **cfg.env_kwargs)
    env = SubprocVecEnv([make_env for i in range(cfg.num_workers)])

    # 2. make agent
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
        wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.save(".hydra/*")
    elif not cfg.wid is None:
        print('[WARN] wid was set but wandb is not enabled')
    
    # 3. run
    ddpg_trainer = ddpg_agent(cfg, env, env_params, ckpt_data=ckpt_data)
    ddpg_trainer.learn()

if __name__ == '__main__':
    launch()