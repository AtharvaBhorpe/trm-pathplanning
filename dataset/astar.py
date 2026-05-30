"""
A* shortest-path planner — the classical baseline for the latency comparison.

Where `solver.py` runs BFS *outward from the goal* to label **every** cell (the dense
action field used as training labels), A* answers a single **start -> goal** query and
expands far fewer nodes thanks to the Manhattan-distance heuristic. That makes it the
fair, strong classical reference for "give me one path, how fast?".

A* is optimal by construction here (4-connected unit-cost grid + admissible Manhattan
heuristic), so in the comparison its success rate and optimality ratio are trivially
1.0 — its only interesting axis is **latency**, and unlike the neural models it cannot
batch: N grids cost N independent searches.

Action / move encoding matches solver.py: 0=N, 1=S, 2=E, 3=W.
"""
import heapq

import numpy as np

_MOVES = [(-1, 0), (1, 0), (0, 1), (0, -1)]  # N, S, E, W


def _manhattan(a, b):
    """Admissible heuristic for a 4-connected unit-cost grid."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def astar(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]):
    """
    Shortest path from start to goal with A* (Manhattan heuristic).

    grid:  (H, W) array, 0 = free, 1 = obstacle.
    start: (row, col), must be free.
    goal:  (row, col), must be free.

    Returns (path, nodes_expanded):
        path           : list of (row, col) from start to goal inclusive, or None if
                         unreachable.
        nodes_expanded : number of nodes popped from the frontier (a cost proxy).
    """
    height, width = grid.shape
    if grid[start] == 1 or grid[goal] == 1:
        return None, 0

    # frontier entries: (f = g + h, g, cell). g is the cost-so-far.
    frontier = [(_manhattan(start, goal), 0, start)]
    came_from = {start: None}
    best_g = {start: 0}
    nodes_expanded = 0

    while frontier:
        _, g, cell = heapq.heappop(frontier)
        nodes_expanded += 1
        if cell == goal:
            return _reconstruct(came_from, goal), nodes_expanded
        # skip stale frontier entries left over from a since-improved g
        if g > best_g[cell]:
            continue
        row, col = cell
        for d_row, d_col in _MOVES:
            n_row, n_col = row + d_row, col + d_col
            if not (0 <= n_row < height and 0 <= n_col < width):
                continue
            if grid[n_row, n_col] == 1:
                continue
            neighbour = (n_row, n_col)
            new_g = g + 1
            if new_g < best_g.get(neighbour, np.inf):
                best_g[neighbour] = new_g
                came_from[neighbour] = cell
                f = new_g + _manhattan(neighbour, goal)
                heapq.heappush(frontier, (f, new_g, neighbour))

    return None, nodes_expanded  # goal never popped -> unreachable


def _reconstruct(came_from, goal):
    """Walk the came_from chain back from goal to start, then reverse."""
    path = [goal]
    while came_from[path[-1]] is not None:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def astar_path_length(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> int:
    """Shortest start->goal path length (number of steps), or -1 if unreachable."""
    path, _ = astar(grid, start, goal)
    return len(path) - 1 if path is not None else -1


if __name__ == "__main__":
    # same sanity grid as solver.py: a vertical wall with one gap
    grid = np.zeros((5, 5), dtype=np.int8)
    grid[1:4, 2] = 1
    grid[1, 2] = 0
    start, goal = (4, 0), (4, 4)
    path, expanded = astar(grid, start, goal)
    print("Grid:\n", grid)
    print("Path:", path)
    print("Length:", astar_path_length(grid, start, goal), "| nodes expanded:", expanded)
