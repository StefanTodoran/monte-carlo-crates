import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CratesCratersPolicy(nn.Module):
    """
    Simple neural network policy for solving the hill climbing task.
    Consists of one common dense layer for both policy and value estimate and
    another dense layer for each.
    """

    def __init__(self, n_obs, n_hidden_1, n_hidden_2, n_actions):
        super(CratesCratersPolicy, self).__init__()

        self.n_obs = n_obs
        self.n_hidden_1 = n_hidden_1
        self.n_hidden_2 = n_hidden_2
        self.n_actions = n_actions

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)  # Move model to GPU if available

        self.dense_in = nn.Linear(n_obs, n_hidden_1)
        self.dense_1 = nn.Linear(n_hidden_1, n_hidden_2)
        self.dense_2 = nn.Linear(n_hidden_2, n_actions)
        self.dense_out = nn.Linear(n_hidden_2, 1)

    def forward(self, obs):
        obs = obs.to(self.device)  # Move input to GPU
        h1_relu = F.relu(self.dense_in(obs.float()))
        h2_relu = F.relu(self.dense_1(h1_relu))

        logits = self.dense_2(h2_relu)
        policy = F.softmax(logits, dim=1)

        value = self.dense_out(h2_relu).view(-1)

        return logits, policy, value

    def step(self, obs):
        """
        Returns policy and value estimates for given observations.
        :param obs: Array of shape [N] containing N observations.
        :return: Policy estimate [N, n_actions] and value estimate [N] for
        the given observations.
        """
        obs = np.array(obs, dtype=np.int64)
        obs = torch.from_numpy(obs)
        _, pi, v = self.forward(obs)
        return pi.cpu().detach().numpy(), v.cpu().detach().numpy()
