from sparse_rl.utils.vec_env import SubprocVecEnv
from sparse_rl.utils.normalizer import Normalizer, ArrayNormalizer
# from sparse_rl.utils.mpi_utils import sync_networks, sync_grads
import gym

__all__ = ['SubprocVecEnv', 'Normalizer', 'ArrayNormalizer']

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

