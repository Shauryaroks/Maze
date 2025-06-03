import osmnx as ox
import networkx as nx
import numpy as np
import random
import heapq
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import time

class MultiNodeRouterAstar:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.distance_matrix = self.precompute_distances()
        
    def haversine(self, node1, node2):
        R = 6371  # Earth radius
        lat1, lon1 = self.graph.nodes[node1]['y'], self.graph.nodes[node1]['x']
        lat2, lon2 = self.graph.nodes[node2]['y'], self.graph.nodes[node2]['x']
        
        dlat = radians(lat2 - lat1)
        dlon = radians(lon2 - lon1)
        a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
        return R * 2 * atan2(sqrt(a), sqrt(1-a))
    
    def algo(self, start, end):
        open_set = [(0, start)]  #fscore, node
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
                    distance_matrix[(i,j)] = sum(self.graph[u][v][0]['length'] 
                                              for u,v in zip(path[:-1], path[1:]))
        return distance_matrix

class MultiNodeRouterDjikstra:
    def __init__(self, graph, nodes):
        self.graph = graph
        self.nodes = nodes
        self.distance_matrix = self.precompute_distances()  # Fixed typo
    
    def haversine(self, node1, node2):
        R = 6371  # Earth radius
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
                        distance_matrix[(i,j)] = sum(
                            self.graph[u][v][0]['length'] 
                            for u,v in zip(path[:-1], path[1:])
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

        





class EnhancedACO:
    def __init__(self, router, num_ants=20, iterations=50, 
                 alpha=1, beta=3, evaporation=0.3):
        self.router = router
        self.num_ants = num_ants
        self.iterations = iterations 
        self.alpha = alpha
        self.beta = beta
        self.evaporation = evaporation
        self.pheromone = {(i,j):1.0 for i in router.nodes for j in router.nodes if i != j}
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
        probabilities = [p/total for p in probabilities]
        return random.choices(list(unvisited), weights=probabilities, k=1)[0]
    
    def update_pheromones(self, paths):
        for edge in self.pheromone:
            self.pheromone[edge] *= (1 - self.evaporation)
            
        for path, length in paths:
            deposit = 1 / length
            for i in range(len(path)-1):
                self.pheromone[(path[i], path[i+1])] += deposit
                
    def reconstruct_full_path(self, node_order):
        full_path = []
        for i in range(len(node_order)-1):
            segment = self.router.algo(node_order[i], node_order[i+1])
            full_path += segment[:-1]
        full_path.append(node_order[-1])
        return full_path


class EnhancedGA:
    def __init__(self, router, pop_size=50, generations=100, 
                 mutation_rate=0.1, elite_size=5):
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
        return sum(self.router.distance_matrix[(individual[i], individual[i+1])] 
                   for i in range(len(individual)-1))
    
    def crossover(self, parent1, parent2):
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None]*len(parent1)
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
        for i in range(len(node_order)-1):
            segment = self.router.algo(node_order[i], node_order[i+1])
            full_path += segment[:-1]
        full_path.append(node_order[-1])
        return full_path


def length_of_path(route):
    pass


node_array = []
'''cmd = ""
startY = float(input("enter the beginning node lat"))
startX = float(input("enter the beginning node lon"))
node_array.append([startX,startY])
while(cmd!="exit"):
    cmd = input("enter command
                1. exit if you are done adding nodes
                2. add if you wanna add a node
                ")
    match(cmd):
        
        case "add":
            lat = float(input("enter latitude"))
            lon = float(input("enter longitude"))
            node_array.append([lon,lat])
        case "exit":
            pass
        '''
node_array = [[ 77.3210,28.5672],[ 77.3332,28.5448],[77.3649, 28.6289],[77.3193 ,28.5704],[77.3546,28.5743],[77.4525,28.6092]]
            
    
north = max(node_array,key=lambda x:x[1])[1]+0.02
south = min(node_array,key=lambda x:x[1])[1]-0.02
east = max(node_array,key=lambda x:x[0])[0]+0.02
west = min(node_array,key=lambda x:x[0])[0]-0.02

graph = ox.graph_from_bbox((west,south,east,north),network_type='drive')

nodes = list(map(lambda x: ox.distance.nearest_nodes(graph, X=x[0], Y=x[1]), node_array))

sr1 = time.time()
router1 = MultiNodeRouterAstar(graph,nodes)
er1 = time.time()
print("time taken to initialize Astar router", er1-sr1)

sr2 = time.time()
router2 = MultiNodeRouterDjikstra(graph,nodes)
er2 = time.time()
print("time taken to initialize Djikstra router", er2-sr2)

routers = [router1,router2]


Aco = []
Ga = []

for i in range(2):    
    if i==0:
        print("A*")
    else:
        print("Djisktra")
    s1 = time.time()
    aco_solver = EnhancedACO(routers[i])
    aco_path = aco_solver.run()
    aco_path_length = aco_solver.best_length
    e1 = time.time()
    Aco.append(aco_path) 
    print("Ant Colony Optimization path length = ",aco_path_length," time taken=",e1-s1)

    
    s2 = time.time()
    ga_solver = EnhancedGA(routers[i])
    ga_path = ga_solver.run()
    ga_path_length = ga_solver.best_length
    e2 = time.time()
    Ga.append(ga_path)
    print("Genetic Algorithm path length = ",ga_path_length," time taken=",e2-s2)


#experimental code

#end

x = []
y = []
for i in range(len(node_array)):
    x.append(node_array[i][0])
    y.append(node_array[i][1])


fig_ga, ax_ga = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5, show=False, close=False)
ax_ga.scatter(x, y, c='green', alpha=0.5, s=50, zorder=3, label='locations')
ox.plot_graph_route(graph, Ga[0], route_color='red', ax=ax_ga, route_linewidth=2)
ox.plot_graph_route(graph, Ga[1], route_color='blue', ax=ax_ga, route_linewidth=2)
ax_ga.legend()
fig_ga.suptitle("Genetic Algorithm Routes")

fig_aco, ax_aco = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5, show=False, close=False)
ax_aco.scatter(x, y, c='green', alpha=0.5, s=50, zorder=3, label='locations')
ox.plot_graph_route(graph, Aco[0], route_color='red', ax=ax_aco, route_linewidth=2)
ox.plot_graph_route(graph, Aco[1], route_color='blue', ax=ax_aco, route_linewidth=2)
ax_aco.legend()
fig_aco.suptitle("Ant Colony Optimization Routes")

plt.show()
