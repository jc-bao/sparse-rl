from multiprocessing import Process, Pipe
from collections import OrderedDict
from abc import ABC, abstractmethod
import cloudpickle, pickle
import gym
import numpy as np


class VecEnv(ABC):
    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    @abstractmethod
    def close(self):
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def get_images(self):
        raise NotImplementedError

    def render(self, *args, **kwargs):
        pass

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def _get_indices(self, indices):
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices


class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self, num_envs=venv.num_envs, observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, *args, **kwargs):
        return self.venv.render(*args, **kwargs)

    def get_images(self):
        return self.venv.get_images()


class CloudpickleWrapper(object):
    def __init__(self, var):
        self.var = var

    def __getstate__(self):
        return cloudpickle.dumps(self.var)

    def __setstate__(self, obs):
        self.var = pickle.loads(obs)


def _worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.var()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                remote.send((observation, reward, done, info))
            elif cmd == 'reset':
                observation = env.reset()
                remote.send(observation)
            elif cmd == 'render':
                remote.send(env.render(*data[0], **data[1]))
            elif cmd == 'close':
                remote.close()
                break
            elif cmd == 'get_spaces':
                remote.send((env.observation_space, env.action_space))
            elif cmd == 'env_method':
                method = getattr(env, data[0])
                remote.send(method(*data[1], **data[2]))
            elif cmd == 'get_attr':
                remote.send(getattr(env, data))
            elif cmd == 'set_attr':
                remote.send(setattr(env, data[0], data[1]))
            else:
                raise NotImplementedError
        except EOFError:
            break


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.waiting = False
        self.closed = False
        n_envs = len(env_fns)

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.processes = [
            Process(target=_worker, args=(work_remote,
                    remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for process in self.processes:
            process.daemon = True  # if the main process crashes, we should not cause things to hang
            process.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()

        self.goal_env = False
        self.goal_keys = None
        if isinstance(observation_space, gym.spaces.Dict):
            dummy_env = env_fns[0]()
            self.dummy_env = dummy_env
            if dummy_env.compute_reward is not None:
                self.compute_reward = dummy_env.compute_reward

            if hasattr(dummy_env, 'goal_extraction_function') and dummy_env.goal_extraction_function is not None:
                self.goal_extraction_function = dummy_env.goal_extraction_function
            self.goal_env = True
            self.goal_keys = tuple(observation_space.spaces.keys())
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        if self.goal_env:
            obs = {k: np.stack([o[k] for o in obs]) for k in self.goal_keys}
        else:
            obs = np.stack(obs)
        return obs, np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        if self.goal_env:
            obs = {k: np.stack([o[k] for o in obs]) for k in self.goal_keys}
        else:
            obs = np.stack(obs)
        return obs

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for process in self.processes:
            process.join()
        self.closed = True

    def render(self, mode='human', *args, **kwargs):
        for pipe in self.remotes:
            # gather images from subprocesses
            # `mode` will be taken into account later
            pipe.send(('render', (args, {'mode': 'rgb_array', **kwargs})))
        imgs = [pipe.recv() for pipe in self.remotes]
        # Create a big image by tiling images from subprocesses
        # bigimg = tile_images(imgs)
        if mode == 'human':
            import cv2
            # cv2.imshow('vecenv', bigimg[:, :, ::-1])
            cv2.imshow('vecenv', imgs)
            cv2.waitKey(1)
        elif mode == 'rgb_array':
            return imgs
        else:
            raise NotImplementedError

    def get_images(self):
        for pipe in self.remotes:
            pipe.send(('render', {"mode": 'rgb_array'}))
        imgs = [pipe.recv() for pipe in self.remotes]
        return imgs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in target_remotes]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(('set_attr', (attr_name, value)))
        for remote in target_remotes:
            remote.recv()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_remotes = self._get_target_remotes(indices)
        for remote in target_remotes:
            remote.send(
                ('env_method', (method_name, method_args, method_kwargs)))
        return [remote.recv() for remote in target_remotes]

    def _get_target_remotes(self, indices):
        """
        Get the connection object needed to communicate with the wanted
        envs that are in subprocesses.
        :param indices: (None,int,Iterable) refers to indices of envs.
        :return: ([multiprocessing.Connection]) Connection object to communicate between processes.
        """
        indices = self._get_indices(indices)
        return [self.remotes[i] for i in indices]


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.

    :param img_nhwc: (list) list or array of images, ndim=4 once turned into array. img nhwc
        n = batch index, h = height, w = width, c = channel
    :return: (numpy float) img_HWc, ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    n_images, height, width, n_channels = img_nhwc.shape
    # new_height was named H before
    new_height = int(np.ceil(np.sqrt(n_images)))
    # new_width was named W before
    new_width = int(np.ceil(float(n_images) / new_height))
    img_nhwc = np.array(list(
        img_nhwc) + [img_nhwc[0] * 0 for _ in range(n_images, new_height * new_width)])
    # img_HWhwc
    out_image = img_nhwc.reshape(
        new_height, new_width, height, width, n_channels)
    # img_HhWwc
    out_image = out_image.transpose(0, 2, 1, 3, 4)
    # img_Hh_Ww_c
    out_image = out_image.reshape(
        new_height * height, new_width * width, n_channels)
    return out_image


class DummyVecEnv(VecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns),
                        env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space

        self.dummy_env = env
        if isinstance(obs_space, gym.spaces.Dict):
            assert isinstance(obs_space.spaces, OrderedDict)
            subspaces = obs_space.spaces
            if env.compute_reward is not None:
                self.compute_reward = env.compute_reward
            if hasattr(env, 'goal_extraction_function') and env.goal_extraction_function is not None:
                self.goal_extraction_function = env.goal_extraction_function
        else:
            subspaces = {None: obs_space}

        for key, box in subspaces.items():
            shapes[key] = box.shape
            dtypes[key] = box.dtype
            self.keys.append(key)

        self.buf_obs = {k: np.zeros(
            (self.num_envs, ) + tuple(shapes[k]), dtype=dtypes[k]) for k in self.keys}
        self.buf_dones = np.zeros((self.num_envs, ), dtype=np.bool)
        self.buf_rews = np.zeros((self.num_envs, ), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] =\
                self.envs[env_idx].step(self.actions[env_idx])
            if self.buf_dones[env_idx]:
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        if self.keys == [None]:
            return (np.copy(self._obs_from_buf()), np.copy(self.buf_rews), np.copy(self.buf_dones), self.buf_infos.copy())
        else:
            return ({k: np.copy(v)
                     for k, v in self._obs_from_buf().items()}, np.copy(self.buf_rews), np.copy(self.buf_dones),
                    self.buf_infos.copy())

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        if self.keys == [None]:
            return np.copy(self._obs_from_buf())
        else:
            return {k: np.copy(v) for k, v in self._obs_from_buf().items()}

    def close(self):
        return

    def get_images(self):
        return [env.render(mode='rgb_array') for env in self.envs]

    def render(self, *args, **kwargs):
        if self.num_envs == 1:
            return self.envs[0].render(*args, **kwargs)
        else:
            return super().render(*args, **kwargs)

    def _save_obs(self, env_idx, obs):
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self):
        if self.keys == [None]:
            return self.buf_obs[None]
        else:
            return self.buf_obs

    def get_attr(self, attr_name, indices=None):
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name, value, indices=None):
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def _get_target_envs(self, indices):
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
