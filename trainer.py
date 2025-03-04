import torch
import torch.nn as nn


class Trainer:
    """
    Trainer for an MCTS policy network. Trains the network to minimize
    the difference between the value estimate and the actual returns and
    the difference between the policy estimate and the refined policy estimates
    derived via the tree search.
    """

    def __init__(self, Policy, learning_rate=0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.step_model = Policy()
        self.step_model.to(self.device)

        value_criterion = nn.MSELoss()
        optimizer = torch.optim.SGD(self.step_model.parameters(), lr=learning_rate)

        def train(obs, search_pis, returns):
            obs = torch.from_numpy(obs).to(self.device)
            search_pis = torch.from_numpy(search_pis).to(self.device)
            returns = torch.from_numpy(returns).to(self.device)

            optimizer.zero_grad()
            logits, policy, value = self.step_model(obs)

            logsoftmax = nn.LogSoftmax(dim=1)
            policy_loss = 5 * torch.mean(torch.sum(-search_pis * logsoftmax(logits), dim=1))
            value_loss = value_criterion(value, returns)
            loss = policy_loss + value_loss

            loss.backward()
            optimizer.step()

            return value_loss.cpu().data.numpy(), policy_loss.cpu().data.numpy()

        self.train = train
