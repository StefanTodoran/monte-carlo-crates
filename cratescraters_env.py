import sys

import gym
import numpy as np

from game_board import (
    Direction,
    TileType,
    load_level,
    unsqueeze_board,
)
from game_logic import (
    Game,
    Position,
    do_game_move,
    initialize_game_obj,
    is_win_condition_met,
)
from static_env import StaticEnv
from utils import cprint

tile_visual_width = 3

player_visual = (" Ω ", "purple")

one_way_visuals = {
    Direction.UP: " ▲ ",
    Direction.DOWN: " ▼ ",
    Direction.LEFT: " ◀ ",
    Direction.RIGHT: " ▶ ",
}

tile_visuals = {
    TileType.EMPTY: ("   ", "white"),
    TileType.WALL: ("███", "blue"),
    TileType.DOOR: ("<D>", "green"),
    TileType.KEY: (" K ", "green"),
    TileType.CRATE: (" ■ ", "purple"),
    TileType.CRATER: (" □ ", "purple"),
    TileType.COIN: (" C ", "orange"),
    TileType.SPAWN: ("   ", "white"),
    TileType.FLAG: (" F ", "orange"),
    TileType.BOMB: ("●", "red"),
    TileType.EXPLOSION: (" ✸ ", "red"),
    TileType.LITTLE_EXPLOSION: (" ✶ ", "red"),
    TileType.ONEWAY: (one_way_visuals, "blue"),
    TileType.METAL_CRATE: (" ■ ", "red"),
    TileType.ICE_BLOCK: (" ■ ", "blue"),
}


class CratesCratersEnv(gym.Env, StaticEnv):
    level_id: int = 0

    def __init__(self, level_id: int):
        raw_board = load_level(level_id)
        self.board = unsqueeze_board(raw_board)
        self.game = initialize_game_obj(self.board)

    def reset(self):
        """Resets the environment to an initial state, returning the initial observation and observation information."""
        self.game = initialize_game_obj(self.board)
        return self.game, 0, False, None

    def step(self, action: Direction):
        """Takes a step in the environment using an action returning the next observation, reward, if the environment terminated and observation information."""
        next_game, _ = do_game_move(self.game, action)
        self.game = next_game
        reward = CratesCratersEnv.get_return(self.game, len(self.game.move_history))
        return self.game, reward, self.game.won, None

    def render(self, mode="human"):
        if mode != "human":
            simple_state = {
                "position": self.game.player,
                "coins": self.game.coins,
                "keys": self.game.keys,
                "moves": len(self.game.move_history),
            }
            print(simple_state)
        else:
            # If we want to clear each time we render:
            # os.system("cls" if os.name == "nt" else "clear")
            print("┏" + ("-" * self.game.board.width * tile_visual_width) + "┓")

            for y in range(self.game.board.height):
                print("|", end="")
                for x in range(self.game.board.width):
                    layer = self.game.board.get_layer(y, x)
                    tile = layer.foreground if layer.foreground.id != TileType.EMPTY else layer.background
                    visual, color = tile_visuals[tile.id]
                    if tile.id == TileType.ONEWAY:
                        visual = visual[tile.orientation]

                    if tile.id == TileType.BOMB:
                        visual = visual + str(tile.fuse).rjust(2)

                    if Position(x, y) == self.game.player:
                        visual, color = player_visual

                    cprint(f"{visual}", color, end="")
                print("|")

            print("┗" + ("-" * self.game.board.width * tile_visual_width) + "┛")
            print(f"Coins: {self.game.coins}/{self.game.max_coins}, Keys: {self.game.keys} ({len(self.game.move_history)} moves)")

    @staticmethod
    def next_state(state: Game, action: Direction) -> Game:
        next_game, _ = do_game_move(state, action)
        return next_game

    @staticmethod
    def is_done_state(state):
        return is_win_condition_met(state)

    @staticmethod
    def initial_state(cls):
        raw_board = load_level(cls.level_id)
        board = unsqueeze_board(raw_board)
        return initialize_game_obj(board)

    @staticmethod
    def get_obs_for_states(states: list[Game]):
        return np.array(states)

    @staticmethod
    def get_return(state: Game, step_idx):
        return state.coins * 100 - step_idx


if __name__ == "__main__":
    level_id = int(input("Enter desired level index: "))
    env = CratesCratersEnv(level_id)

    key_to_direction = {
        "W": Direction.UP,
        "A": Direction.LEFT,
        "S": Direction.DOWN,
        "D": Direction.RIGHT,
    }

    game_history = []
    while True:
        env.render()

        if env.is_done_state(env.game):
            print("You won!")
            sys.exit(0)

        undo_hint = "U to undo, " if len(game_history) else ""
        print(f"Use the WASD keys to move. Press {undo_hint}R to reset and Q to quit.")
        keypress = input("Input: ").upper()

        if keypress == "Q":
            print("Quitting game...")
            sys.exit(0)

        elif keypress == "R":
            env.reset()
            continue

        elif keypress == "U" and game_history:
            env.game = game_history.pop()
            continue

        elif keypress in key_to_direction:
            direction = key_to_direction[keypress]
            game_history.append(env.game)
            game, _, _, _ = env.step(direction)

        else:
            print("Invalid keypress. Please try again.")
            continue
