import osmnx as ox
import networkx as nx
import numpy as np
import random
import heapq
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import time
import csv

# --- Routing Classes ---

class MultiNodeRouterAstar:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.distance_matrix = self.precompute_distances()

    def haversine(self, node1, node2):
        R = 6371  # Earth radius in km
        lat1, lon1 = self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x']
        lat2, lon2 = self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    def algo(self, start, end):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.haversine(start, end)}
        visited = set()
        while open_set:
            _, current = heapq.heappop(open_set)
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            visited.add(current)
            for neighbor in self.graph.successors(current):
                edge_data = self.graph[current][neighbor][0]
                tentative_g = g_score[current] + edge_data['length']
                if neighbor in visited and tentative_g >= g_score.get(neighbor, float('inf')):
                    continue
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self.haversine(neighbor, end)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        return None

    def precompute_distances(self):
        distance_matrix = {}
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    path = self.algo(i, j)
                    if path:
                        distance_matrix[(i, j)] = sum(self.graph[u][v][0]['length']
                                                      for u, v in zip(path[:-1], path[1:]))
        return distance_matrix

class MultiNodeRouterDjikstra:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.distance_matrix = self.precompute_distances()

    def haversine(self, node1, node2):
        R = 6371
        lat1, lon1 = self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x']
        lat2, lon2 = self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))

    def precompute_distances(self):
        distance_matrix = {}
        for i in self.nodes:
            for j in self.nodes:
                if i != j:
                    path = self.algo(i, j)
                    if path:
                        distance_matrix[(i, j)] = sum(
                            self.graph[u][v][0]['length']
                            for u, v in zip(path[:-1], path[1:])
                        )
        return distance_matrix

    def algo(self, start, end):
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        visited = set()
        while open_set:
            current_cost, current = heapq.heappop(open_set)
            if current == end:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                return path[::-1]
            if current in visited:
                continue
            visited.add(current)
            for neighbor in self.graph.successors(current):
                edge_data = self.graph[current][neighbor][0]
                tentative_g = g_score[current] + edge_data['length']
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, neighbor))
        return None

# --- Metaheuristics ---

class EnhancedACO:
    def __init__(self, router, num_ants=20, iterations=50, alpha=1, beta=3, evaporation=0.3):
        self.router = router
        self.num_ants = num_ants
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromone = {(i, j): 1.0 for i in router.nodes for j in router.nodes if i != j}
        self.best_length = 0

    def run(self):
        best_path = None
        best_length = float('inf')
        for _ in range(self.iterations):
            paths = [self.construct_path() for _ in range(self.num_ants)]
            self.update_pheromones(paths)
            current_best = min(paths, key=lambda x: x[1])
            if current_best[1] < best_length:
                best_path, best_length = current_best
                self.best_length = best_length
        return self.reconstruct_full_path(best_path)

    def construct_path(self):
        unvisited = set(self.router.nodes)
        current = random.choice(self.router.nodes)
        unvisited.remove(current)
        path = [current]
        total_length = 0
        while unvisited:
            next_node = self.select_next(current, unvisited)
            total_length += self.router.distance_matrix[(current, next_node)]
            path.append(next_node)
            unvisited.remove(next_node)
            current = next_node
        return path, total_length

    def select_next(self, current, unvisited):
        probabilities = []
        for node in unvisited:
            pheromone = self.pheromone[(current, node)] ** self.alpha
            heuristic = (1 / self.router.distance_matrix[(current, node)]) ** self.beta
            probabilities.append(pheromone * heuristic)
        total = sum(probabilities)
        probabilities = [p / total for p in probabilities]
        return random.choices(list(unvisited), weights=probabilities, k=1)[0]

    def update_pheromones(self, paths):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation)
        for path, length in paths:
            deposit = 1 / length
            for i in range(len(path) - 1):
                self.pheromone[(path[i], path[i + 1])] += deposit

    def reconstruct_full_path(self, node_order):
        full_path = []
        for i in range(len(node_order) - 1):
            segment = self.router.algo(node_order[i], node_order[i + 1])
            full_path += segment[:-1]
        full_path.append(node_order[-1])
        return full_path

class EnhancedGA:
    def __init__(self, router, pop_size=50, generations=100, mutation_rate=0.1, elite_size=5):
        self.router = router
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.best_length = 0

    def run(self):
        pop = self.initial_population()
        for _ in range(self.generations):
            pop = self.evolve(pop)
        best = min(pop, key=lambda x: self.fitness(x))
        self.best_length = self.fitness(best)
        return self.reconstruct_full_path(best)

    def initial_population(self):
        return [random.sample(self.router.nodes, len(self.router.nodes))
                for _ in range(self.pop_size)]

    def fitness(self, individual):
        return sum(self.router.distance_matrix[(individual[i], individual[i + 1])]
                   for i in range(len(individual) - 1))

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while child[ptr] is not None:
                    ptr += 1
                child[ptr] = gene
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve(self, pop):
        graded = sorted(pop, key=lambda x: self.fitness(x))
        elites = graded[:self.elite_size]
        children = []
        while len(children) < self.pop_size - self.elite_size:
            p1, p2 = random.choices(graded[:20], k=2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            children.append(child)
        return elites + children

    def reconstruct_full_path(self, node_order):
        full_path = []
        for i in range(len(node_order) - 1):
            segment = self.router.algo(node_order[i], node_order[i + 1])
            full_path += segment[:-1]
        full_path.append(node_order[-1])
        return full_path

# --- Hybrid ACO-GA ---

class HybridACO_GA:
    def __init__(self, router, pop_size=50, generations=100, mutation_rate=0.1, elite_size=5, aco_ants=20, aco_iters=50):
        self.router = router
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        self.aco_ants = aco_ants
        self.aco_iters = aco_iters
        self.best_length = 0

    def run(self):
        pop = self.initial_population()
        for _ in range(self.generations):
            pop = self.evolve(pop)
        best = min(pop, key=lambda x: self.fitness(x))
        self.best_length = self.fitness(best)
        return self.reconstruct_full_path(best)

    def initial_population(self):
        aco = EnhancedACO(self.router, num_ants=self.aco_ants, iterations=self.aco_iters)
        # Use ACO to generate candidate solutions
        aco_paths = [aco.construct_path()[0] for _ in range(self.pop_size)]
        unique_paths = []
        seen = set()
        for path in aco_paths:
            t = tuple(path)
            if t not in seen:
                unique_paths.append(path)
                seen.add(t)
            if len(unique_paths) == self.pop_size:
                break
        # If not enough unique, fill with random permutations
        while len(unique_paths) < self.pop_size:
            p = random.sample(self.router.nodes, len(self.router.nodes))
            if tuple(p) not in seen:
                unique_paths.append(p)
                seen.add(tuple(p))
        return unique_paths

    def fitness(self, individual):
        return sum(self.router.distance_matrix[(individual[i], individual[i + 1])]
                   for i in range(len(individual) - 1))

    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        ptr = 0
        for gene in parent2:
            if gene not in child:
                while child[ptr] is not None:
                    ptr += 1
                child[ptr] = gene
        return child

    def mutate(self, individual):
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(individual)), 2)
            individual[i], individual[j] = individual[j], individual[i]
        return individual

    def evolve(self, pop):
        graded = sorted(pop, key=lambda x: self.fitness(x))
        elites = graded[:self.elite_size]
        children = []
        while len(children) < self.pop_size - self.elite_size:
            p1, p2 = random.choices(graded[:20], k=2)
            child = self.crossover(p1, p2)
            child = self.mutate(child)
            children.append(child)
        return elites + children

    def reconstruct_full_path(self, node_order):
        full_path = []
        for i in range(len(node_order) - 1):
            segment = self.router.algo(node_order[i], node_order[i + 1])
            full_path += segment[:-1]
        full_path.append(node_order[-1])
        return full_path

# --- Main Program ---

# --- Load graph from bounding box ---
BBOX_WEST = 77.30
BBOX_SOUTH = 28.45
BBOX_EAST = 77.50
BBOX_NORTH = 28.67
print("Downloading graph data from bounding box...")
graph = ox.graph_from_bbox((BBOX_WEST, BBOX_SOUTH, BBOX_EAST, BBOX_NORTH), network_type='drive')

# --- Interactive Node Selection ---
import matplotlib.pyplot as plt
clicked_points = []

def onclick(event):
    if event.inaxes is not None and event.button == 1:
        x, y = event.xdata, event.ydata
        clicked_points.append((x, y))
        event.inaxes.scatter(x, y, c='red', s=50, zorder=3)
        event.inaxes.annotate(f"{len(clicked_points)}", (x, y), textcoords="offset points", xytext=(5, 5), ha='left', fontsize=10, color='red', weight='bold')
        plt.draw()
        print(f"Point {len(clicked_points)}: ({x:.6f}, {y:.6f})")

print("\nSelect nodes on the map (left-click, close window when done)...")
fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5, show=False, close=False)
ax.set_title("Left-click to add nodes, then close window when done", fontsize=12)
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.tight_layout()
plt.show()
fig.canvas.mpl_disconnect(cid)

node_array = np.array(clicked_points) if clicked_points else np.array([]).reshape(0, 2)
if len(node_array) == 0:
    print("No points were selected.")
    exit()

# Map to nearest OSMnx nodes
nodes = list(map(lambda x: ox.distance.nearest_nodes(graph, X=x[0], Y=x[1]), node_array))
seen = set()
nodes = [x for x in nodes if not (x in seen or seen.add(x))]

print(f"\nTotal points selected: {len(node_array)}")
print("Selected points (longitude, latitude):")
for i, pt in enumerate(node_array):
    print(f"Point {i+1}: ({pt[0]:.6f}, {pt[1]:.6f})")

# --- Routers ---
print("Initializing routers...")
sr1 = time.time()
router1 = MultiNodeRouterAstar(graph, nodes)
er1 = time.time()
print("A* router initialized in", er1 - sr1, "seconds")

sr2 = time.time()
router2 = MultiNodeRouterDjikstra(graph, nodes)
er2 = time.time()
print("Dijkstra router initialized in", er2 - sr2, "seconds")

routers = [router1, router2]

# --- Solvers ---
Aco = []
Ga = []
Hybrid = []

for i in range(2):
    print("\nRouter:", "A*" if i == 0 else "Dijkstra")

    s1 = time.time()
    aco_solver = EnhancedACO(routers[i])
    aco_path = aco_solver.run()
    aco_path_length = aco_solver.best_length
    e1 = time.time()
    Aco.append(aco_path)
    print("ACO path length =", aco_path_length, "time taken =", e1 - s1)

    s2 = time.time()
    ga_solver = EnhancedGA(routers[i])
    ga_path = ga_solver.run()
    ga_path_length = ga_solver.best_length
    e2 = time.time()
    Ga.append(ga_path)
    print("GA path length =", ga_path_length, "time taken =", e2 - s2)

    s3 = time.time()
    hybrid_solver = HybridACO_GA(routers[i])
    hybrid_path = hybrid_solver.run()
    hybrid_path_length = hybrid_solver.best_length
    e3 = time.time()
    Hybrid.append(hybrid_path)
    print("Hybrid ACO-GA path length =", hybrid_path_length, "time taken =", e3 - s3)

# --- Visualization ---

x = [pt[0] for pt in node_array]
y = [pt[1] for pt in node_array]

fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5, show=False, close=False)
ax.scatter(x, y, c='green', alpha=0.5, s=50, zorder=3, label='locations')
ox.plot_graph_route(graph, Ga[0], route_color='red', ax=ax, route_linewidth=2, orig_dest_size=0)
ox.plot_graph_route(graph, Aco[0], route_color='blue', ax=ax, route_linewidth=2, orig_dest_size=0)
ox.plot_graph_route(graph, Hybrid[0], route_color='magenta', ax=ax, route_linewidth=2, orig_dest_size=0)
ax.legend()
fig.suptitle("A* Routers: GA (red), ACO (blue), Hybrid (magenta)")

fig2, ax2 = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5, show=False, close=False)
ax2.scatter(x, y, c='green', alpha=0.5, s=50, zorder=3, label='locations')
ox.plot_graph_route(graph, Ga[1], route_color='red', ax=ax2, route_linewidth=2, orig_dest_size=0)
ox.plot_graph_route(graph, Aco[1], route_color='blue', ax=ax2, route_linewidth=2, orig_dest_size=0)
ox.plot_graph_route(graph, Hybrid[1], route_color='magenta', ax=ax2, route_linewidth=2, orig_dest_size=0)
ax2.legend()
fig2.suptitle("Dijkstra Routers: GA (red), ACO (blue), Hybrid (magenta)")

plt.show()

# --- Benchmark Initializer Runtimes and Write to CSV ---


# --- Benchmark Full Pipeline (Routers + ACO/GA/Hybrid) and Write to CSV ---
full_results = []
for run in range(10):
    print(f"\n--- Full Pipeline Run {run + 1} ---")
    nodes = list(map(lambda x: ox.distance.nearest_nodes(graph, X=x[0], Y=x[1]), node_array))
    seen = set()
    nodes = [x for x in nodes if not (x in seen or seen.add(x))]

    s1 = time.time()
    router1 = MultiNodeRouterAstar(graph, nodes)
    e1 = time.time()
    astar_time = e1 - s1

    s2 = time.time()
    router2 = MultiNodeRouterDjikstra(graph, nodes)
    e2 = time.time()
    dijkstra_time = e2 - s2

    s3 = time.time()
    aco_astar = EnhancedACO(router1)
    aco_astar_path = aco_astar.run()
    aco_astar_length = aco_astar.best_length
    e3 = time.time()
    aco_astar_time = e3 - s3

    s4 = time.time()
    ga_astar = EnhancedGA(router1)
    ga_astar_path = ga_astar.run()
    ga_astar_length = ga_astar.best_length
    e4 = time.time()
    ga_astar_time = e4 - s4

    s5 = time.time()
    hybrid_astar = HybridACO_GA(router1)
    hybrid_astar_path = hybrid_astar.run()
    hybrid_astar_length = hybrid_astar.best_length
    e5 = time.time()
    hybrid_astar_time = e5 - s5

    s6 = time.time()
    aco_dijkstra = EnhancedACO(router2)
    aco_dijkstra_path = aco_dijkstra.run()
    aco_dijkstra_length = aco_dijkstra.best_length
    e6 = time.time()
    aco_dijkstra_time = e6 - s6

    s7 = time.time()
    ga_dijkstra = EnhancedGA(router2)
    ga_dijkstra_path = ga_dijkstra.run()
    ga_dijkstra_length = ga_dijkstra.best_length
    e7 = time.time()
    ga_dijkstra_time = e7 - s7

    s8 = time.time()
    hybrid_dijkstra = HybridACO_GA(router2)
    hybrid_dijkstra_path = hybrid_dijkstra.run()
    hybrid_dijkstra_length = hybrid_dijkstra.best_length
    e8 = time.time()
    hybrid_dijkstra_time = e8 - s8

    print(f"Astar: ACO={aco_astar_length:.2f} ({aco_astar_time:.2f}s), GA={ga_astar_length:.2f} ({ga_astar_time:.2f}s), Hybrid={hybrid_astar_length:.2f} ({hybrid_astar_time:.2f}s)")
    print(f"Dijkstra: ACO={aco_dijkstra_length:.2f} ({aco_dijkstra_time:.2f}s), GA={ga_dijkstra_length:.2f} ({ga_dijkstra_time:.2f}s), Hybrid={hybrid_dijkstra_length:.2f} ({hybrid_dijkstra_time:.2f}s)")

    full_results.append({
        'run': run + 1,
        'astar_time': astar_time,
        'dijkstra_time': dijkstra_time,
        'aco_astar_length': aco_astar_length,
        'aco_astar_time': aco_astar_time,
        'ga_astar_length': ga_astar_length,
        'ga_astar_time': ga_astar_time,
        'hybrid_astar_length': hybrid_astar_length,
        'hybrid_astar_time': hybrid_astar_time,
        'aco_dijkstra_length': aco_dijkstra_length,
        'aco_dijkstra_time': aco_dijkstra_time,
        'ga_dijkstra_length': ga_dijkstra_length,
        'ga_dijkstra_time': ga_dijkstra_time,
        'hybrid_dijkstra_length': hybrid_dijkstra_length,
        'hybrid_dijkstra_time': hybrid_dijkstra_time
    })

with open('full_benchmark.csv', 'w', newline='') as csvfile:
    fieldnames = [
        'run', 'astar_time', 'dijkstra_time',
        'aco_astar_length', 'aco_astar_time',
        'ga_astar_length', 'ga_astar_time',
        'hybrid_astar_length', 'hybrid_astar_time',
        'aco_dijkstra_length', 'aco_dijkstra_time',
        'ga_dijkstra_length', 'ga_dijkstra_time',
        'hybrid_dijkstra_length', 'hybrid_dijkstra_time'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in full_results:
        writer.writerow(row)
print("\nFull pipeline benchmark results written to full_benchmark.csv")



'''Point 1: (77.339795, 28.623858)
Point 2: (77.404198, 28.619487)
Point 3: (77.324865, 28.593261)
Point 4: (77.393366, 28.589918)
Point 5: (77.335697, 28.577062)
Point 6: (77.387804, 28.562664)
Point 7: (77.353554, 28.561378)
Point 8: (77.444010, 28.580662)
Point 9: (77.416200, 28.604317)
Point 10: (77.422640, 28.576291)
Point 11: (77.419127, 28.550322)
Point 12: (77.378729, 28.546465)
Point 13: (77.355603, 28.546208)
Point 14: (77.355310, 28.546208)
Point 15: (77.371996, 28.625915)
Point 16: (77.454841, 28.657540)
Point 17: (77.419127, 28.648798)'''