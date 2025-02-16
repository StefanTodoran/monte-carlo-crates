from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from game_board import (
    Direction,
    LayeredBoard,
    Position,
    SimpleTile,
    TileType,
)


@dataclass
class Game:
    board: LayeredBoard = None
    player: Position = None
    max_coins: int = 0
    coins: int = 0
    keys: int = 0
    won: bool = False
    move_history: List[Direction] = field(default_factory=list)


def clone_game_obj(game: Game) -> Game:
    new_game = Game()
    new_game.board = game.board.clone()
    new_game.player = Position(x=game.player.x, y=game.player.y)
    new_game.max_coins = game.max_coins
    new_game.coins = game.coins
    new_game.keys = game.keys
    new_game.won = game.won
    new_game.move_history = game.move_history.copy()
    return new_game


def can_walk_tile(
    y_pos: int,
    x_pos: int,
    game: Game,
    extra: List[TileType] = None,
    direction: Direction = None,
) -> bool:
    if not game.board.in_bounds(y_pos, x_pos):
        return False

    walkable = [
        TileType.EMPTY,
        TileType.SPAWN,
        TileType.LITTLE_EXPLOSION,
        TileType.EXPLOSION,
    ]
    if extra:
        walkable.extend(extra)

    target_space = game.board.get_layer(y_pos, x_pos)
    if target_space.foreground.id == TileType.FLAG and game.coins == game.max_coins:
        return True

    can_walk = (
        target_space.foreground.id in walkable
        and target_space.background.id in walkable
    )

    if (
        can_walk
        and direction is not None
        and target_space.background.id == TileType.ONEWAY
        and not can_enter_oneway(direction, target_space.background)
    ):
        can_walk = False

    return can_walk


def can_enter_oneway(direction: Direction, tile: SimpleTile) -> bool:
    if direction == Direction.UP and tile.orientation == Direction.DOWN:
        return False
    if direction == Direction.DOWN and tile.orientation == Direction.UP:
        return False
    if direction == Direction.LEFT and tile.orientation == Direction.RIGHT:
        return False
    if direction == Direction.RIGHT and tile.orientation == Direction.LEFT:
        return False
    return True


def get_spawn_position(board: LayeredBoard) -> Position:
    """Returns the player spawn position in the given level."""
    for i in range(board.height):
        for j in range(board.width):
            if board.get_tile(i, j).id == TileType.SPAWN:
                return Position(x=j, y=i)
    return Position(x=float("nan"), y=float("nan"))


def count_instances_in_board(board: LayeredBoard, target: TileType) -> int:
    """Returns the number of times a tile type appears in the board."""
    count = 0
    for i in range(board.height):
        for j in range(board.width):
            if board.get_tile(i, j).id == target:
                count += 1
    return count


def can_move_to(game: Game, tile_x: int, tile_y: int) -> Optional[List[Direction]]:
    """
    Checks if the destination position can be reached from the current position.
    Returns a path of directions if possible, None if not.
    """
    if not can_walk_tile(
        tile_y, tile_x, game, [TileType.COIN, TileType.KEY, TileType.ONEWAY]
    ):
        return None

    @dataclass
    class SearchNode:
        x: int
        y: int
        path: List[Direction]

    class PositionSet:
        def __init__(self, width: int):
            self.positions = set()
            self.width = width

        def add(self, node: SearchNode):
            self.positions.add(node.y * self.width + node.x)

        def has(self, node: SearchNode) -> bool:
            return (node.y * self.width + node.x) in self.positions

    class Queue:
        def __init__(self):
            self.items = []

        def enqueue(self, item):
            self.items.append(item)

        def dequeue(self):
            return self.items.pop(0)

        @property
        def is_empty(self):
            return len(self.items) == 0

    visited = PositionSet(game.board.width)
    queue = Queue()
    queue.enqueue(SearchNode(x=game.player.x, y=game.player.y, path=[]))

    while not queue.is_empty:
        current = queue.dequeue()

        if visited.has(current):
            continue

        walkable = [TileType.COIN, TileType.KEY, TileType.ONEWAY]
        direction = current.path[-1] if current.path else None

        if not can_walk_tile(current.y, current.x, game, walkable, direction):
            continue

        visited.add(current)

        if tile_x == current.x and tile_y == current.y:
            return current.path

        queue.enqueue(
            SearchNode(
                x=current.x + 1, y=current.y, path=current.path + [Direction.RIGHT]
            )
        )
        queue.enqueue(
            SearchNode(
                x=current.x - 1, y=current.y, path=current.path + [Direction.LEFT]
            )
        )
        queue.enqueue(
            SearchNode(
                x=current.x, y=current.y + 1, path=current.path + [Direction.DOWN]
            )
        )
        queue.enqueue(
            SearchNode(x=current.x, y=current.y - 1, path=current.path + [Direction.UP])
        )

    return None


# Define constant tiles
explosion_tile = SimpleTile(id=TileType.EXPLOSION)
little_explosion = SimpleTile(id=TileType.LITTLE_EXPLOSION)
empty_tile = SimpleTile(id=TileType.EMPTY)


def do_game_move(game: Game, move: Direction) -> Tuple[Game, bool]:
    """
    Attempts to do a move, returns the successor state and whether the state changed.
    """
    next_game = clone_game_obj(game)
    move_to = Position(x=game.player.x, y=game.player.y)
    one_further = Position(x=game.player.x, y=game.player.y)

    dx = dy = 0
    if move == Direction.UP:
        dy = -1
    elif move == Direction.DOWN:
        dy = 1
    elif move == Direction.LEFT:
        dx = -1
    elif move == Direction.RIGHT:
        dx = 1

    move_to.x += dx
    move_to.y += dy
    one_further.x += dx * 2
    one_further.y += dy * 2

    if not next_game.board.in_bounds(move_to.y, move_to.x):
        return game, False

    # Clear explosion tiles
    dimensions = [next_game.board.height, next_game.board.width]
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            tile = next_game.board.get_tile(i, j)
            if tile.id in [TileType.EXPLOSION, TileType.LITTLE_EXPLOSION]:
                next_game.board.set_tile(i, j, empty_tile)

    move_to_layer = next_game.board.get_layer(move_to.y, move_to.x)
    one_further_layer = next_game.board.get_layer(one_further.y, one_further.x, True)

    # Handle collectibles
    if move_to_layer.foreground.id == TileType.COIN:
        next_game.coins += 1
        next_game.board.set_tile(move_to.y, move_to.x, empty_tile)
    if move_to_layer.foreground.id == TileType.KEY:
        next_game.keys += 1
        next_game.board.set_tile(move_to.y, move_to.x, empty_tile)

    # Handle door
    if game.keys > 0 and move_to_layer.foreground.id == TileType.DOOR:
        next_game.keys -= 1
        next_game.board.set_tile(move_to.y, move_to.x, empty_tile)

    # Handle pushable objects and ice blocks
    if one_further_layer.background.id == TileType.EMPTY or (
        one_further_layer.background.id == TileType.ONEWAY
        and can_enter_oneway(move, one_further_layer.background)
    ):
        # Handle crater filling
        if (
            move_to_layer.foreground.id in fill_capable_tiles
            and one_further_layer.foreground.id == TileType.CRATER
        ):
            next_game.board.set_tile(move_to.y, move_to.x, empty_tile)
            next_game.board.set_tile(one_further.y, one_further.x, empty_tile)

        # Handle pushing
        if (
            move_to_layer.foreground.id in pushable_tiles
            and one_further_layer.foreground.id == TileType.EMPTY
        ):
            next_game.board.set_tile(
                one_further.y, one_further.x, move_to_layer.foreground
            )
            next_game.board.set_tile(move_to.y, move_to.x, empty_tile)

        # Handle ice block sliding
        if (
            move_to_layer.foreground.id == TileType.ICE_BLOCK
            and one_further_layer.foreground.id == TileType.EMPTY
        ):
            curr_x = one_further.x
            curr_y = one_further.y
            prev_x = curr_x
            prev_y = curr_y

            while -1 <= curr_x <= dimensions[1] and -1 <= curr_y <= dimensions[0]:
                curr_layer = next_game.board.get_layer(curr_y, curr_x, True)

                if curr_layer.foreground.id == TileType.CRATER:
                    next_game.board.set_tile(curr_y, curr_x, empty_tile)
                    break

                if (
                    curr_layer.foreground.id != TileType.EMPTY
                    or (
                        curr_layer.background.id == TileType.ONEWAY
                        and not can_enter_oneway(move, curr_layer.background)
                    )
                    or curr_layer.background.id in [TileType.OUTSIDE, TileType.WALL]
                ):
                    next_game.board.set_tile(prev_y, prev_x, move_to_layer.foreground)
                    break

                prev_x = curr_x
                prev_y = curr_y
                curr_x += dx
                curr_y += dy

            next_game.board.set_tile(move_to.y, move_to.x, empty_tile)

    moved = attempt_move(move_to.y, move_to.x, next_game, move)
    if not moved:
        return game, False

    # Handle bomb logic
    for i in range(dimensions[0]):
        for j in range(dimensions[1]):
            tile = next_game.board.get_tile(i, j)
            if tile.id == TileType.BOMB:
                tile.fuse -= 1
                if tile.fuse == 0:
                    for y, x in [(i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)]:
                        if next_game.board.get_tile(y, x, True).id in explodable_tiles:
                            next_game.board.set_tile(y, x, little_explosion)
                    next_game.board.set_tile(i, j, explosion_tile)

    next_game.won = win_condition(next_game)
    next_game.move_history.append(move)

    return next_game, True


def attempt_move(
    y_pos: int, x_pos: int, next_game: Game, direction: Optional[Direction] = None
) -> bool:
    """Attempts to move the player to the given position."""
    if can_walk_tile(y_pos, x_pos, next_game, [TileType.ONEWAY], direction):
        next_game.player.x = x_pos
        next_game.player.y = y_pos
        return True
    return False


def win_condition(game: Game) -> bool:
    """Checks if the win condition has been met."""
    return (
        game.board.get_tile(game.player.y, game.player.x).id == TileType.FLAG
        and game.coins == game.max_coins
    )


def initialize_game_obj(board: LayeredBoard) -> Game:
    """Creates a new game object for the given board."""
    board = board.clone()
    start_pos = get_spawn_position(board)
    board.set_tile(start_pos.y, start_pos.x, empty_tile)

    game = Game()
    game.board = board
    game.player = start_pos
    game.max_coins = count_instances_in_board(board, TileType.COIN)
    game.coins = 0
    game.keys = 0
    game.won = False
    game.move_history = []
    return game


@dataclass
class Offset:
    dx: int
    dy: int


def is_valid_move(game: Game, offset: Offset) -> bool:
    """Checks if a move is valid given the current game state."""
    x_pos = game.player.x + offset.dx
    y_pos = game.player.y + offset.dy

    background = game.board.get_layer(y_pos, x_pos, True).background
    if background.id in [TileType.OUTSIDE, TileType.WALL]:
        return False

    tile = game.board.get_tile(y_pos, x_pos, True)
    if tile.id == TileType.CRATER:
        return False
    if tile.id == TileType.DOOR and game.keys == 0:
        return False
    if tile.id == TileType.FLAG and game.coins != game.max_coins:
        return False

    if background.id == TileType.ONEWAY:
        if background.orientation == Direction.LEFT and x_pos > game.player.x:
            return False
        if background.orientation == Direction.RIGHT and x_pos < game.player.x:
            return False
        if background.orientation == Direction.UP and y_pos > game.player.y:
            return False
        if background.orientation == Direction.DOWN and y_pos < game.player.y:
            return False

    if tile.id in [*pushable_tiles, TileType.ICE_BLOCK]:
        return is_pushable(game.board, Position(x=x_pos, y=y_pos), offset)

    return True


def is_pushable(board: LayeredBoard, position: Position, offset: Offset) -> bool:
    """Checks if a tile can be pushed in the given direction."""
    move_to_layer = board.get_layer(position.y, position.x)
    one_further_layer = board.get_layer(
        position.y + offset.dy, position.x + offset.dx, True
    )

    if offset.dy == -1:
        move = Direction.UP
    elif offset.dy == 1:
        move = Direction.DOWN
    elif offset.dx == -1:
        move = Direction.LEFT
    else:
        move = Direction.RIGHT

    if one_further_layer.background.id == TileType.EMPTY or (
        one_further_layer.background.id == TileType.ONEWAY
        and can_enter_oneway(move, one_further_layer.background)
    ):
        # Check crater filling
        if (
            move_to_layer.foreground.id in fill_capable_tiles
            and one_further_layer.foreground.id == TileType.CRATER
        ):
            return True

        # Check regular pushing
        if (
            move_to_layer.foreground.id in pushable_tiles
            and one_further_layer.foreground.id == TileType.EMPTY
        ):
            return True

        # Check ice block pushing
        if (
            move_to_layer.foreground.id == TileType.ICE_BLOCK
            and one_further_layer.foreground.id == TileType.EMPTY
        ):
            return True

    return False


def is_win_condition_met(next: Game) -> bool:
    return next.board.get_tile(next.player.y, next.player.x).id == TileType.FLAG and (
        next.coins == next.max_coins
    )


fill_capable_tiles = [TileType.ICE_BLOCK]  # Add other fill capable tiles as needed
pushable_tiles = [TileType.BOMB]  # Add other pushable tiles as needed
explodable_tiles = [
    TileType.EMPTY,
    TileType.SPAWN,
]  # Add other explodable tiles as needed
