from dataclasses import dataclass
from enum import Enum
from typing import List


class Direction(Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class TileType(Enum):
    EMPTY = 0
    WALL = 1
    DOOR = 2
    KEY = 3
    CRATE = 4
    CRATER = 5
    COIN = 6
    SPAWN = 7
    FLAG = 8
    BOMB = 9
    EXPLOSION = 10
    LITTLE_EXPLOSION = 11
    ONEWAY = 12
    METAL_CRATE = 13
    ICE_BLOCK = 14
    OUTSIDE = 15


@dataclass
class Position:
    x: int
    y: int


@dataclass
class SimpleTile:
    id: TileType
    fuse: int = None
    orientation: Direction = None


@dataclass
class LayerTiles:
    foreground: SimpleTile
    background: SimpleTile


class LayeredBoard:
    def __init__(self, board: List[List[LayerTiles]]):
        self.board = board
        self.height = len(board)
        self.width = len(board[0]) if self.height > 0 else 0

    @classmethod
    def create_empty_board(cls, width: int = 8, height: int = 14) -> "LayeredBoard":
        empty_tile = SimpleTile(id=TileType.EMPTY)
        return cls([[LayerTiles(foreground=empty_tile, background=empty_tile) for _ in range(width)] for _ in range(height)])

    def clone(self) -> "LayeredBoard":
        new_board = [
            [
                LayerTiles(
                    foreground=SimpleTile(**vars(tile.foreground)),
                    background=SimpleTile(**vars(tile.background)),
                )
                for tile in row
            ]
            for row in self.board
        ]
        return LayeredBoard(new_board)

    def slice(self, start: int, end: int) -> "LayeredBoard":
        return LayeredBoard(self.board[start:end])

    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.height and 0 <= x < self.width

    def get_layer(self, y: int, x: int, check_bounds: bool = False) -> LayerTiles:
        if not check_bounds or self.in_bounds(y, x):
            return self.board[y][x]
        outside_tile = SimpleTile(id=TileType.OUTSIDE)
        return LayerTiles(foreground=outside_tile, background=outside_tile)

    def get_tile(self, y: int, x: int, check_bounds: bool = False) -> SimpleTile:
        return self.get_layer(y, x, check_bounds).foreground

    def set_tile(self, y: int, x: int, tile: SimpleTile) -> None:
        self.board[y][x].foreground = tile

    def get_background(self, y: int, x: int, check_bounds: bool = False) -> SimpleTile:
        return self.get_layer(y, x, check_bounds).background

    def set_background(self, y: int, x: int, tile: SimpleTile) -> None:
        self.board[y][x].background = tile

    def find_adjacent_walls(self, y: int, x: int) -> dict:
        return {
            "top": y > 0 and self.board[y - 1][x].background.id == TileType.WALL,
            "left": x > 0 and self.board[y][x - 1].background.id == TileType.WALL,
            "bottom": y + 1 < self.height and self.board[y + 1][x].background.id == TileType.WALL,
            "right": x + 1 < self.width and self.board[y][x + 1].background.id == TileType.WALL,
        }

    def to_observations(self):
        observations: list[int] = []
        for row in self.board:
            for layer in row:
                data = [layer.foreground.id.value, layer.background.id.value]

                if layer.foreground.id == TileType.BOMB:
                    data.append(layer.foreground.fuse)
                elif layer.foreground.id == TileType.ONEWAY:
                    data.append(layer.foreground.orientation.value)
                else:
                    data.append(-1)

                observations.extend(data)

        return observations

    # def to_tokens(self):
    #     observations: list[int] = []
    #     for row in self.board:
    #         for layer in row:
    #             data = [layer.foreground.id.value, layer.background.id.value]

    #             if layer.foreground.id == TileType.BOMB:
    #                 data.append(layer.foreground.fuse)
    #             elif layer.foreground.id == TileType.ONEWAY:
    #                 data.append(layer.foreground.orientation.value)
    #             else:
    #                 data.append(-1)

    #             observations.extend(data)

    #     return observations


def load_level(level_index: int) -> List[List[SimpleTile]]:
    with open("levels.txt", "r") as file:
        raw_lines = file.readlines()

    raw_rows = raw_lines[level_index].strip().split("/")
    board = []

    for row in raw_rows:
        board_row = []
        raw_tiles = row.split(",")

        for tile_data in raw_tiles:
            tile_parts = tile_data.split(".")
            tile_id = int(tile_parts[0])
            tile = SimpleTile(id=TileType(tile_id))

            if tile.id == TileType.ONEWAY and len(tile_parts) > 1:
                tile.orientation = Direction(int(tile_parts[1]))

            if tile.id == TileType.BOMB and len(tile_parts) > 1:
                tile.fuse = int(tile_parts[1])

            board_row.append(tile)

        board.append(board_row)

    return board


def unsqueeze_board(flat_board: List[List[SimpleTile]]) -> LayeredBoard:
    background_tile_types = {
        TileType.EMPTY,
        TileType.WALL,
        TileType.ONEWAY,
        TileType.OUTSIDE,
    }

    layered_board = []
    height = len(flat_board)
    width = len(flat_board[0]) if height > 0 else 0

    for y_pos in range(height):
        row = []

        for x_pos in range(width):
            tile = flat_board[y_pos][x_pos]
            layer = LayerTiles(
                foreground=SimpleTile(id=TileType.EMPTY),
                background=SimpleTile(id=TileType.EMPTY),
            )

            if tile.id in background_tile_types:
                layer.background = tile
            else:
                layer.foreground = tile

            row.append(layer)

        layered_board.append(row)

    return LayeredBoard(layered_board)
