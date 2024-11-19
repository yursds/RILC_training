__credits__ = ["Yuri De Santis"]

import datetime
import os
import torch

from torch.utils.tensorboard import SummaryWriter
from stable_baselines3.common.callbacks import BaseCallback


class CB4TB(BaseCallback):
    """
    Custom Callback to visualize specific parameters of the environment in TensorBoard.

    :param verbose: (int, optional) Show details in terminal if set to 1. Defaults to 0.
    :param checkFreq: (int, optional) Frequency to save the model w.r.t self.n_calls. Defaults to 1.
    :param reset_epN: (int, optional) Consecutive episodes of ILC. Defaults to 1.
    :param modelFolder: (str, optional) Path to save trained model. Defaults to "model".
    :param logFolder: (str, optional) Path to save log of training used for TensorBoard. Defaults to "log".
    """

    def __init__(
        self,
        verbose: int = 0,
        checkFreq: int = 1,
        reset_epN: int = 1,
        modelFolder: str = "model",
        logFolder: str = "log",
    ):
        """
        Custom Callback to visualize specific parameters of the environment in TensorBoard.

        :param verbose: (int, optional) Show details in terminal if set to 1. Defaults to 0.
        :param checkFreq: (int, optional) Frequency to save the model w.r.t self.n_calls. Defaults to 1.
        :param reset_epN: (int, optional) Consecutive episodes of ILC. Defaults to 1.
        :param modelFolder: (str, optional) Path to save trained model. Defaults to "model".
        :param logFolder: (str, optional) Path to save log of training used for TensorBoard. Defaults to "log".
        """
        super().__init__(verbose)
        self.check = checkFreq
        self.reset_epN = reset_epN

        l_Path, m_Path, b_Path = self.__log_model_paths(model_path=modelFolder, log_path=logFolder)
        self.modelsPath = m_Path
        self.bestPath = b_Path
        self.logPath = l_Path
        self.writer = SummaryWriter(log_dir=l_Path)

        self.sum_dict = {}  # dict of sum of reward components in self.locals["infos"] of single episode
        self.mean_dict = {}  # dict of mean of reward components in self.locals["infos"] of self.model._stats_window_size episodes
        self.eps_dict = {}  # dict of sum reward components in self.locals["infos"] of episodes
        self.steps_dict = {}  # dict of reward components in self.locals["infos"] of single steps in an episode
        self.keys_list = ['rw_dict']  # specify key of infos that you want to see
        self.step_in_ep = 0  # current step in current episode
        self.n_ep = 0  # current number of episodes
        self.n_update = 0  # current number of policy updates
        self.best_reward = -torch.inf

    def _on_training_start(self):
        """Called at the beginning of training."""
        self.windowSize = self.model._stats_window_size

    def _on_rollout_start(self):
        """Called before each rollout starts."""
        pass

    def _on_step(self) -> bool:
        """Called at each step."""
        infos_: tuple[dict] = self.locals["infos"]
        keys_list = self.keys_list  # exclude info truncated
        rw_dict = "rw_dict"
        self.step_in_ep += 1

        ep = self.n_ep
        update = self.n_update

        # update steps dictionary
        for i in range(self.model.n_envs):
            # single environment info
            info_: dict = infos_[i]
            # transform dictionary in torch.stack(torch.Tensor())
            for key_dict in keys_list:
                tmp_dict: dict = info_[key_dict]

                if key_dict == rw_dict:
                    # create additional dimension
                    values = torch.stack(list(tmp_dict.values())).unsqueeze(0)
                    try:
                        # concatenate with respect to new dimension
                        tmp_: torch.Tensor = self.steps_dict[key_dict]
                        self.steps_dict[key_dict] = torch.cat([tmp_, values], dim=0)
                    except:
                        # only for the first step
                        self.steps_dict[key_dict] = values
                else:
                    # plot only first environment
                    if i == 0 and ep % self.reset_epN == 0:
                        for key_in, value in tmp_dict.items():
                            idx = 0
                            self.writer.add_scalar(f'{key_in}_{idx + 1}/ep_{ep}', value[idx], self.step_in_ep)

        # save model
        if self.n_calls % self.check == 0:
            self.model.save(f"{self.modelsPath}/{self.n_calls}")
            episodes_reward = torch.asarray([ep_info['r'] for ep_info in self.model.ep_info_buffer])
            if episodes_reward.size()[0] >= self.windowSize:
                mean_rewards = torch.mean(episodes_reward[-self.windowSize:]).item()
                if mean_rewards > self.best_reward:
                    self.model.save(f"{self.bestPath}/best_model")
                    self.best_reward = mean_rewards

        # update episode dictionary
        if self.locals["dones"][0]:
            # sum respect number of environment, create additional dimension
            tmp_sum = torch.sum(self.steps_dict[rw_dict], dim=0).unsqueeze(0)
            try:
                # concatenate with respect to new dimension
                tmp_: torch.Tensor = self.eps_dict[rw_dict]
                self.eps_dict[rw_dict] = torch.cat([tmp_, tmp_sum], dim=0)
                # limit memory usage
                if tmp_.size(0) > self.windowSize - 1:
                    self.eps_dict[rw_dict] = self.eps_dict[rw_dict][-self.windowSize:, ]
            except:
                self.eps_dict[rw_dict] = tmp_sum

            i = 0
            self.writer.add_scalar(f'rmse/rmse_{i}', self.locals["infos"][i]["additional"]["rmse"], self.n_ep)

            self.writer.flush()

            # reset steps dictionary
            self.steps_dict = {}
            self.step_in_ep = 0
            self.n_ep += 1

        return True

    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout."""
        steps = self.num_timesteps

        keys_list = ["rw_dict"]
        ep_mean = {}

        # write on tensorboard
        try:
            # compute useful data
            for key_dict in keys_list:
                # compute ep mean reward
                ts_eps: torch.Tensor = self.eps_dict[key_dict]
                ep_mean[key_dict] = ts_eps.mean(0) / self.model.n_envs

            # define readable name for SummaryWriter
            info_: dict = self.locals["infos"][0]
            for key in ep_mean.keys():
                tmp_dict: dict = info_[key_dict]
                for idx, key_in in enumerate(tmp_dict.keys()):
                    self.writer.add_scalar(f'{key}/{key_in}', ep_mean[key_dict][idx], self.n_update)
        except:
            pass

        # check for correct computation of ep mean reward
        if len(self.model.ep_info_buffer) > 0:
            mean_episode_reward = torch.mean(torch.asarray([ep_info['r'] for ep_info in self.model.ep_info_buffer]))
        else:
            mean_episode_reward = 0.0
        self.writer.add_scalar('mean_reward', mean_episode_reward, self.num_timesteps)

        self.n_update += 1
        self.writer.flush()

    def __log_model_paths(self, model_path: str = "models_", log_path: str = "logs_") -> list[str]:
        """
        Create directories for saving models and logs.

        :param model_path: (str) Base path for saving models.
        :param log_path: (str) Base path for saving logs.
        :return: (list[str]) Paths for logs, models, and best model.
        """
        date_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        models_dir = model_path
        logs_dir = log_path
        modelsPath = os.path.join(models_dir, date_now)
        bestPath = os.path.join(modelsPath, 'best_model')
        logsPath = os.path.join(logs_dir, date_now)

        # create folder
        os.makedirs(modelsPath, exist_ok=True)
        os.makedirs(bestPath, exist_ok=True)
        os.makedirs(logsPath, exist_ok=True)

        return logsPath, modelsPath, bestPath

    def _on_training_end(self):
        """Called at the end of training."""
        self.writer.close()