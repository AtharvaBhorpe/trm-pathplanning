"""
Ground-truth oracle: the optimal action for every cell, found with BFS.

The trick: run a breadth-first search *outward from the goal*. Because every
move costs 1, BFS labels each free cell with its shortest distance to the goal.
The best action at a cell is then simply "step to whichever neighbour is one
step closer to the goal".

This is the label generator — if it's wrong, every result downstream is wrong,
so it's kept small and tested (see the __main__ block and the rerun viz check).

Action encoding:
    0 = North (row - 1)
    1 = South (row + 1)
    2 = East  (col + 1)
    3 = West  (col - 1)
    4 = stay  (the goal cell)
   -100 = masked (obstacle / unreachable) — skipped by the loss
"""
from collections import deque

import numpy as np

_MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W (matches action ids 0..3)
STAY = 4
IGNORE_INDEX = -100  # PyTorch cross-entropy skips cells with this label


def bfs_distance_field(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """
    Shortest distance from every free cell to the goal (4-connected, unit cost).

    grid: (H, W) array, 0 = free, 1 = obstacle.
    goal: (row, col) of the goal (must be free).
    Returns (H, W) int array; obstacles and unreachable cells are -1.
    """
    height, width = grid.shape
    distance = np.full((height, width), -1, dtype=np.int32)
    assert grid[goal] == 0, "Goal cell must be free"

    distance[goal] = 0
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for d_row, d_col in _MOVES:
            n_row, n_col = row + d_row, col + d_col
            # visit free, in-bounds neighbours we haven't reached yet
            if (0 <= n_row < height and 0 <= n_col < width
                    and grid[n_row, n_col] == 0 and distance[n_row, n_col] == -1):
                distance[n_row, n_col] = distance[row, col] + 1
                queue.append((n_row, n_col))
    return distance


def optimal_action_field(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """
    Best action for every cell.

    grid: (H, W) array, 0 = free, 1 = obstacle.
    goal: (row, col) of the goal (must be free).
    Returns (H, W) int array of action ids (see module docstring for encoding).
    """
    height, width = grid.shape
    distance = bfs_distance_field(grid, goal)
    actions = np.full((height, width), IGNORE_INDEX, dtype=np.int64)

    for row in range(height):
        for col in range(width):
            if grid[row, col] == 1:
                continue                      # obstacle -> masked
            if (row, col) == goal:
                actions[row, col] = STAY
                continue
            if distance[row, col] == -1:
                continue                      # unreachable -> masked

            # pick the neighbour that is closest to the goal (smallest distance)
            best_action, best_distance = None, distance[row, col]
            for action_id, (d_row, d_col) in enumerate(_MOVES):
                n_row, n_col = row + d_row, col + d_col
                if (0 <= n_row < height and 0 <= n_col < width
                        and distance[n_row, n_col] != -1
                        and distance[n_row, n_col] < best_distance):
                    best_action, best_distance = action_id, distance[n_row, n_col]
            assert best_action is not None, "a reachable free cell must have a closer neighbour"
            actions[row, col] = best_action
    return actions


def is_reachable(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """True if the goal can be reached from the start."""
    return bfs_distance_field(grid, goal)[start] != -1


def optimal_path_length(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Shortest path length from start to goal, or -1 if unreachable."""
    return int(bfs_distance_field(grid, goal)[start])


if __name__ == "__main__":
    # quick sanity check on a 5x5 grid with a wall that has one gap
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1:4, 2] = 1   # vertical wall
    grid[1, 2] = 0     # gap in the wall
    start, goal = (4, 0), (4, 4)
    print("Grid:\n", grid)
    print("Reachable:", is_reachable(grid, start, goal))
    print("Optimal path length:", optimal_path_length(grid, start, goal))
    print("Optimal action field:\n", optimal_action_field(grid, goal))
