import os
import pickle
import hydra
import wandb
from omegaconf import OmegaConf
import gym, panda_gym
from sparse_rl.agent import ddpg_agent
from sparse_rl.utils import SubprocVecEnv

def get_env_params(env_name):
    env = gym.make(env_name)
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
    env_params = get_env_params(cfg.env_name)
    def make_env():
        import panda_gym
        return gym.make(cfg.env_name)
    env = SubprocVecEnv([make_env for i in range(cfg.num_workers)])

    # 2. make agent
    ckpt_data, wid = None, None
    if os.path.exists(cfg.ckpt_path):
        with open(cfg.ckpt_path, "rb") as f:
            print(f"Loading data from {cfg.ckpt_path}.")
            ckpt_data = pickle.load(f)
            wid = ckpt_data["wandb_run_id"]
    else:
        print('Fail to load')
    if cfg.wandb:
        wandb.init(project='debug', id=wid, resume="allow", dir=hydra.utils.get_original_cwd())
        if wid is None:
            wandb.config.update(OmegaConf.to_container(cfg, resolve=True))
        wandb.save(".hydra/*")
    
    # 3. run
    ddpg_trainer = ddpg_agent(cfg, env, env_params, ckpt_data=ckpt_data)
    ddpg_trainer.learn()

if __name__ == '__main__':
    launch()