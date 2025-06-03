# Maze Generation and Multi-Node Routing Project

This project evolves across three versions, focusing on maze generation, pathfinding, and real-world multi-node routing using various algorithms. Below is an overview of each version, their features, usage, and dependencies.

## Version 1: Maze Generation and Solving
- **File**: `mazegeneration.py`
- **Purpose**: Generates and solves mazes using a modified Prim's algorithm and Depth-First Search (DFS) with a partial Breadth-First Search (BFS) implementation.
- **Key Features**:
  - Creates a maze of user-defined odd size using a modified Prim's algorithm.
  - Visualizes the maze and solution path using Pygame (white for paths, red for solution).
  - Implements DFS for maze solving; BFS is incomplete.
  - Tracks visited nodes and branches for pathfinding.
- **Issues**:
  - BFS implementation is incomplete and non-functional.
  - Time complexity measurement using `timeit` yields inconsistent results.
  - GUI lacks start/end markers and may not scale well for large mazes.
- **Usage**:
  - Run `mazegeneration.py`, input an odd number for maze size.
  - Maze is generated, solved with DFS, and displayed via Pygame.
  - Close Pygame window to exit.
- **Dependencies**: Python, NumPy, Pygame, time, timeit.

## Version 2: Advanced Maze Pathfinding with RL
- **Files**:
  - `pathfind.py`: Implements Ant Colony Optimization (ACO) and Genetic Algorithm (GA) without Reinforcement Learning (RL).
  - `pathfind_RL.py`: Implements ACO and GA with RL enhancements.
  - `pathfinder.py`: Combines both algorithms, compares performance, and logs results.
  - `pyamaze.py`: Library for maze generation and visualization (by Learning Orbis).
- **Purpose**: Solves mazes using ACO and GA, with RL variants to improve pathfinding, visualized with Pygame.
- **Key Features**:
  - Generates random mazes with configurable size and 50% loop percentage using `pyamaze`.
  - Solves mazes with ACO (green), ACORL (cyan), GA (red), and GARL (yellow).
  - RL variants adjust move probabilities based on path cost.
  - Logs runtime and path length to `data.csv` for analysis.
- **Issues**:
  - RL implementation in `pathfind_RL.py` has an incomplete `set_probabilities` method (references undefined `RLrewardsystem`).
  - Particle Swarm Optimization (PSO) section is present but incomplete.
  - CSV logging may append duplicate headers.
- **Usage**:
  - Run `pathfinder.py`, input maze dimensions (rows, columns).
  - Maze is generated, solved with all four algorithms, and visualized.
  - Results saved to `data.csv`.
- **Dependencies**: Python, NumPy, Pygame, Pyamaze, Matplotlib, CSV.

## Version 3: Multi-Node Routing on Real-World Graphs
- **File**: `V3.3.py`
- **Purpose**: Finds optimal routes visiting multiple nodes on a real-world road network using A* and Dijkstra’s algorithms with ACO and GA for optimization.
- **Key Features**:
  - Fetches road network data using OSMnx within a bounding box defined by node coordinates.
  - Computes shortest paths with A* and Dijkstra’s algorithms using precomputed distance matrices.
  - Applies ACO and GA to solve the Traveling Salesman Problem for multi-node routes.
  - Visualizes routes with Matplotlib (ACO: red, GA: blue) on the road network.
- **Issues**:
  - `length_of_path` function is defined but not implemented.
  - Uses hardcoded node coordinates; dynamic node input code is commented out.
  - Limited error handling for disconnected graphs or invalid nodes.
- **Usage**:
  - Run `V3.3.py` with predefined node coordinates.
  - Generates road network, computes routes using A* and Dijkstra’s with ACO and GA, and displays two plots.
  - Prints path lengths and runtimes for each algorithm.
- **Dependencies**: Python, OSMnx, NetworkX, NumPy, Matplotlib.

## General Usage
- **Version 1**: Run `mazegeneration.py` for basic maze generation and DFS solving.
- **Version 2**: Run `pathfinder.py` for advanced maze solving with ACO, GA, and RL variants.
- **Version 3**: Run `V3.3.py` for real-world multi-node routing.
- Ensure all dependencies are installed (`pip install osmnx networkx numpy matplotlib pygame`).

## Future Improvements
- **Version 1**: Complete BFS implementation, fix time measurement, add GUI start/end markers.
- **Version 2**: Fix RL probability setup, implement PSO, improve CSV logging to avoid duplicate headers.
- **Version 3**: Implement `length_of_path`, enable dynamic node input, add robust error handling.
- Optimize algorithms across all versions for larger inputs and enhance visualization.

## Notes
- Each version builds on the previous, transitioning from grid-based mazes to real-world routing.
- Ensure proper installation of dependencies, especially OSMnx for Version 3, which requires geospatial libraries.
- Commented-out code in all versions indicates experimental or incomplete features for future development.