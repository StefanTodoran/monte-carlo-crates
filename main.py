import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from cratescraters_env import CratesCratersEnv
from game_board import Direction
from mcts import execute_episode
from policy import CratesCratersPolicy
from replay_memory import ReplayMemory
from trainer import Trainer


def log(test_env, iteration, step_idx, total_rew):
    """
    Logs one step in a testing episode.
    :param test_env: Test environment that should be rendered.
    :param iteration: Number of training iterations so far.
    :param step_idx: Index of the step in the episode.
    :param total_rew: Total reward collected so far.
    """
    time.sleep(0.3)
    print()
    print(f"Training Episodes: {iteration}")
    test_env.render()
    print(f"Step: {step_idx}")
    print(f"Return: {total_rew}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--levels", nargs="+", type=int, default=[0], help="List of level indices to train on")
    parser.add_argument("--memory-size", type=int, default=200, help="Size of replay memory buffer")
    parser.add_argument("--num-simulations", type=int, default=32, help="Number of simulations per episode")
    parser.add_argument("--num-epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--hidden-1-size", type=int, default=64, help="Size of first hidden layer")
    parser.add_argument("--hidden-2-size", type=int, default=24, help="Size of second hidden layer")
    args = parser.parse_args()

    test_env = CratesCratersEnv(args.levels[0])
    n_actions = len(Direction)
    n_obs = len(test_env.game.to_observations())
    print("Number of observations:", n_obs)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainer = Trainer(lambda: CratesCratersPolicy(n_obs, args.hidden_1_size, args.hidden_2_size, n_actions))
    network = trainer.step_model

    mem = ReplayMemory(
        args.memory_size,
        {"ob": np.int64, "pi": np.float32, "return": np.float32},
        {"ob": [n_obs], "pi": [n_actions], "return": []},
        batch_size=32,
    )

    def test_agent(iteration, level_idx):
        test_env = CratesCratersEnv(level_idx)
        total_rew = 0
        state, reward, done, _ = test_env.reset()
        step_idx = 0
        while not done:
            log(test_env, iteration, step_idx, total_rew)
            p, _ = network.step(np.array([state]))
            action = np.argmax(p)
            state, reward, done, _ = test_env.step(action)
            step_idx += 1
            total_rew += reward
        log(test_env, iteration, step_idx, total_rew)

    value_losses = []
    policy_losses = []

    for level_idx in args.levels:
        for epoch in tqdm(range(args.num_epochs)):
            obs, pis, returns, total_reward, done_state = execute_episode(network, args.num_simulations, CratesCratersEnv)
            mem.add_all({"ob": obs, "pi": pis, "return": returns})
            batch = mem.get_minibatch()

            vl, pl = trainer.train(batch["ob"], batch["pi"], batch["return"])
            value_losses.append(vl)
            policy_losses.append(pl)

            if epoch > 0 and epoch % 500 == 0:
                test_agent(epoch, level_idx)
                plt.plot(value_losses, label="value loss")
                plt.plot(policy_losses, label="policy loss")
                plt.legend()
                plt.show()
