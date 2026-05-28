"""
Ground-truth oracle: optimal per-cell action policy via BFS from the goal.

This is infrastructure. If the labels are wrong,every downstream result is 
meaningless. It is implemented and tested here so you can trust it.
The rerun visual check in build_dataset.py is your second line of defense.

Action encoding:
    0 = North (row - 1)
    1 = South (row + 1)
    2 = East  (col + 1)
    3 = West  (col - 1)
    4 = stay  (goal cell, or unreachable/obstacle — masked out of the loss)
"""
from collections import deque
import numpy as np

# (d_row, d_column) for each action id 0..3
_MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]
STAY = 4
IGNORE_INDEX = -100  # standard PyTorch cross-entropy ignore value

def bfs_distance_field(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """
    Breadth-first distance-to-goal for every free cell (4-connected, unit cost)

    Args:
        grid: (H, W) array, 0 = free, 1 = obstacle.
        goal: (row, column) of the goal cell (must be free).
    
    Returns:
        dist: (H, W) int array. Distance to goal in steps; unreachable / obstacle cells are -1.
    """
    height, width = grid.shape
    distance = np.full(shape=(height, width), fill_value=-1, dtype=np.int32)
    goal_row, goal_column = goal
    assert grid[goal_row, goal_column] == 0, "Goal cell must be free"
    distance[goal_row, goal_column] = 0
    q = deque([(goal_row, goal_column)])
    while q:
        row, column = q.popleft()
        for d_row, d_column in _MOVES:
            n_row, n_column = row + d_row, column + d_column
            if (0 <= n_row < height and 0 <= n_column < width and
                grid[n_row, n_column] == 0 and distance[n_row, n_column] == -1):
                distance[n_row, n_column] = distance[row, column] + 1
                q.append((n_row, n_column))
    return distance


def optimal_action_field(grid: np.ndarray, goal: tuple[int, int]) -> np.ndarray:
    """
    Optimal action for each cell, given the grid and goal.

    Args:
        grid: (H, W) array, 0 = free, 1 = obstacle.
        goal: (row, column) of the goal cell (must be free).
    
    Returns:
        actions: (H, W) int array.
            - free reachable cells: action id in {0, 1, 2, 3} for N/S/E/W
            - goal cell: 4 (STAY)
            - unreachable/obstacle cells: -100 (IGNORE_INDEX, masked out of the loss)
    """
    height, width = grid.shape
    distance = bfs_distance_field(grid, goal)
    actions = np.full(shape=(height, width), fill_value=IGNORE_INDEX, dtype=np.int64)
    goal_row, goal_column = goal

    for row in range(height):
        for column in range(width):
            if grid[row, column] == 1:
                continue  # obstacle
            if (row, column) == goal:
                actions[row, column] = STAY
                continue
            if distance[row, column] == -1:
                continue  # unreachable
            # Find the neighbor with the smallest distance to the goal
            best_action = None
            best_distance = distance[row, column]
            for action_id, (d_row, d_column) in enumerate(_MOVES):
                n_row, n_column = row + d_row, column + d_column
                if (0 <= n_row < height and 0 <= n_column < width and
                    distance[n_row, n_column] != -1 and distance[n_row, n_column] < best_distance):
                    best_distance = distance[n_row, n_column]
                    best_action = action_id
            assert best_action is not None, "There should be a valid action for reachable free cells"
            actions[row, column] = best_action
    return actions

def is_reachable(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> bool:
    """
    Check if the goal is reachable from the start cell.

    Args:
        grid: (H, W) array, 0 = free, 1 = obstacle.
        start: (row, column) of the start cell (must be free).
        goal: (row, column) of the goal cell (must be free).
    
    Returns:
        True if the goal is reachable from the start, False otherwise.
    """
    return bfs_distance_field(grid, goal)[start] != -1

def optimal_path_length(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """
    Compute the length of the optimal path from start to goal.

    Args:
        grid: (H, W) array, 0 = free, 1 = obstacle.
        start: (row, column) of the start cell (must be free).
        goal: (row, column) of the goal cell (must be free).

    Returns:
        The length of the optimal path from start to goal, or -1 if not reachable.
    """
    return int(bfs_distance_field(grid, goal)[start])

if __name__ == "__main__":
    # Quick test on a small grid
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1:4, 2] = 1  # vertical wall
    grid[1, 2] = 0  # gap in the wall
    start, goal = (4, 0), (4, 4)
    print("Grid:\n", grid)
    print("Reachable:", is_reachable(grid, start, goal))
    print("Optimal path length:", optimal_path_length(grid, start, goal))
    print("Optimal action field:\n", optimal_action_field(grid, goal))