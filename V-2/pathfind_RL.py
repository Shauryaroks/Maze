import random
from pyamaze import maze, agent, COLOR
import numpy as np
import time

'''rows = int(input("no of rows"))
columns = int(input("no of columnss"))

m = maze(rows,columns)
m.CreateMaze(loopPercent=50)#saveMaze=True'''
#m.CreateMaze(loadMaze="maze--2025-02-04--18-25-56.csv")


#        __     __
#       (o \,-/ o)
#       <  . Y .  >
#        \  ---  /
#        /       \
#       / |     | \
#      *  |     |  *
#         `'`'`'`
#  === ANT COLONY ALGORITHM ===


class AntColonySolverRL:
    def __init__(self, maze, start, end, rows, cols, num_ants=40, num_iterations=10, alpha=1.0, beta=2.0, evaporation_rate=0.2, pheromone_deposit=1.0):
        self.maze = maze
        self.start = start
        self.maze = maze
        self.start = start
        self.end = end
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha  
        self.beta = beta    
        self.evaporation_rate = evaporation_rate
        self.pheromone_deposit = pheromone_deposit
        self.rows = rows
        self.cols = cols

        self.pheromone = np.ones((self.rows,self.cols))

        self.moves = ['E','W','S','N']
        self.RLprobabilities = maze

    def set_probabilities(self):
        for i in self.RLprobabilities:
            for j in self.RLprobabilities[i]:
                    if self.RLprobabilities[i][j] == 1:
                        if (j=='N') or (j=='W'):
                            self.RLrewardsystem[i][j] *=0.95
                        if (j=='E') or (j=='S'):
                            self.RLrewardsystem[i][j] *= 1
    
    def reinforce(self,path):
        n = len(path)
        for i in range(len(path)-1):
            x,y = path[i]
            cost = (len(path)/(self.rows+self.cols))*0.01
            if cost>0.05:
                increment_factor = 1
            else:            
                increment_factor = 1.05-cost
            for j in self.moves:
                valid, x,y = self.next_coordinates(x,y,j)
                if valid and (x,y) ==  path[i+1]:
                    self.RLprobabilities[path[i]][j] *= increment_factor

    
                            
    
    #markov decision making
    def heuristic(self,x,y):
        return np.sqrt((self.end[0]-(x-1))**2 + (self.end[1]-(y-1))**2)
    
    def probs(self):
        return self.RLprobabilities
    

    #to compute if a direction is open or not
    def next_coordinates(self,x,y,direction):
        if self.maze[(x,y)][direction]:
            if direction == "E":
                return(True, x,y+1)
            if direction == "W":
                return(True, x, y-1)
            if direction == "S":
                return(True, x+1, y)
            if direction == "N":
                return(True, x-1, y)        
            
        else:
            return(False,x,y)

    #simulates one ant through the whole maze    
    def move_ant(self):
        current_position = self.start
        visited = [current_position]
        path = [current_position]

        while current_position != self.end:
            x,y = current_position
            probabilities = []
            valid_moves = []

            for direction in self.moves:
                is_valid, next_x,next_y = self.next_coordinates(x,y,direction)
                if is_valid and (next_x,next_y) not in visited:
                    valid_moves.append((next_x,next_y))
                    pheremone_level = self.pheromone[next_x-1][next_y-1]**self.alpha
                    heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)**self.beta
                    probabilities.append(pheremone_level*heuristic_level*self.RLprobabilities[(x,y)][direction])
            #backtrack 
            if valid_moves == []:
                path.pop()
                current_position = path[-1]             
            else:
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]

                next_position = random.choices(valid_moves, weights=probabilities, k=1)[0]
                path.append(next_position)
                visited.append(next_position)
                current_position = next_position

        return path
    #updation of pheromones
    def update_pheromones(self,path):    
        self.pheromone *= (1-self.evaporation_rate)

        
        if path is not None:
            path_length = len(path)
            for (x, y) in path:
                self.pheromone[x-1][y-1] += self.pheromone_deposit/path_length
    #main solve function
    def solve(self):
        best_path = None
        best_path_length = float('inf')
        self.set_probabilities #RL

        for iteration in range(self.num_iterations):
            all_paths = []

            for _ in range(self.num_ants):
                path =  self.move_ant()
                if path:
                    all_paths.append(path)
                    if len(path) < best_path_length:
                        best_path  = path
                        best_path_length = len(path)
                self.update_pheromones(path)    
                print(f'best path found {best_path_length} in time ')
                self.reinforce(best_path)
        ACO_solution_dict = {}
        for i in range(1, len(best_path)):  # Start from 1 to avoid out-of-range
            ACO_solution_dict[best_path[i]] = best_path[i - 1]
        return ACO_solution_dict


#    G  E  N  E  T  I  C     A  L  G  O  R  I  T  H  M  
#    
#       ||        ||  
#       ||        ||  
#     (||)      (||)  
#    ((  ))    ((  ))  
#     (||)      (||)  
#      ||        ||  
#      ||        ||  
#     (||)      (||)  
#    ((  ))    ((  ))  
#     (||)      (||)  
#      ||        ||  
#      ||        ||  
#  
#  === GENETIC ALGORITHM ===




class GeneticAlgorithmSolverRL:
    def __init__(self, maze, start, end, rows, cols, mutation_rate= 0.1, population_size = 40, generations = 10):
        self.maze = maze
        self.start = start
        self.end = end
        self.rows = rows
        self.cols = cols
        self.mutation_rate = mutation_rate
        self.population_size = population_size
        self.generations = generations
        self.directions = ['E', 'S','N', 'W']

        self.parent_population = []
        self.initial_population = []
        self.RLprobabilities = maze

    def set_probabilities(self):
        for i in self.RLprobabilities:
            for j in self.RLprobabilities[i]:
                    if self.RLprobabilities[i][j] == 1:
                        if (j=='N') or (j=='W'):
                            self.RLrewardsystem[i][j] *=0.95
                        if (j=='E') or (j=='S'):
                            self.RLrewardsystem[i][j] *= 1

    def reinforce(self,path):
        n = len(path[1])
        for i in range(len(path[1])-1):
            x,y = path[1][i]
            cost = (len(path[1])/(self.rows+self.cols))*0.01
            if cost>0.05:
                increment_factor = 1
            else:            
                increment_factor = 1.05-cost
            for j in self.directions:
                valid, x,y = self.next_coordinates(x,y,j)
                if valid and (x,y) ==  path[1][i+1]:
                    self.RLprobabilities[path[1][i]][j] *= increment_factor


    def next_coordinates(self,x,y,direction):
        if self.maze[(x,y)][direction]:
            if direction == "E":
                return(True, x,y+1)
            if direction == "W":
                return(True, x, y-1)
            if direction == "S":
                return(True, x+1, y)
            if direction == "N":
                return(True, x-1, y)

        else:
            return(False,x,y)

    def heuristic(self,x,y):
        return np.sqrt((self.end[0]-(x-1))**2 + (self.end[1]-(y-1))**2)

    def move(self):
        current_position = self.start
        path = [self.start]
        visited = [self.start]
        while current_position != self.end:
            x,y = current_position
            probabilities = []
            valid_moves = []
            for direction in self.directions:
                is_valid,next_x,next_y = self.next_coordinates(x,y,direction)
                if is_valid and ((next_x,next_y) not in visited) and ((next_x,next_y) ):
                    valid_moves.append((next_x,next_y))
                    heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)
                    probabilities.append(heuristic_level*self.RLprobabilities[x,y][direction])
            #backtrack
            if valid_moves == []:
                path.pop()
                current_position = path[-1]             
            else:
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]

                next_position = random.choices(valid_moves, weights=probabilities, k=1)[0]
                path.append(next_position)
                visited.append(next_position)
                current_position = next_position
        return path 


    def crossover(self,p1,p2):
        current_position = self.start
        visited = [self.start]
        path = [self.start]
        while current_position != self.end:
            x,y = current_position
            probabilities = []
            valid_moves = []
            valid_inherited_moves = []
            probabilities_inherited = []
            for direction in self.directions:
                is_valid,next_x,next_y = self.next_coordinates(x,y,direction)
                if is_valid and ((next_x,next_y) not in visited) and ((next_x,next_y)):
                    if ((next_x,next_y) not in p1) and ((next_x,next_y) not in p2):
                        valid_moves.append((next_x,next_y))
                        heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)
                        probabilities.append(heuristic_level*self.RLprobabilities[x,y][direction])
                    else:
                        if ((next_x,next_y) in p1) and ((next_x,next_y) in p2):
                            valid_inherited_moves.append((next_x,next_y))
                            heuristic_level = heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)
                            probabilities_inherited.append((min(len(p1)-p1.index((next_x,next_y)),len(p2)-p2.index((next_x,next_y))))*heuristic_level*self.RLprobabilities[x,y][direction])
                        else:
                            if (next_x,next_y) in p1:
                                valid_inherited_moves.append((next_x,next_y))
                                heuristic_level = heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)
                                probabilities_inherited.append((len(p1)-p1.index((next_x,next_y)))*heuristic_level*self.RLprobabilities[x,y][direction])
                            else:
                                valid_inherited_moves.append((next_x,next_y))
                                heuristic_level = heuristic_level = (1/self.heuristic(next_x,next_y)+1e-5)
                                probabilities_inherited.append((len(p2)-p2.index((next_x,next_y)))*heuristic_level*self.RLprobabilities[x,y][direction])
            
            if valid_moves == [] and valid_inherited_moves == []:
                path.pop()
                current_position = path[-1]
            elif valid_moves ==[]:
                total = sum(probabilities_inherited)
                probabilities = [p / total for p in probabilities_inherited]
                next_position = random.choices(valid_inherited_moves, weights=probabilities_inherited, k=1)[0]
                path.append(next_position)
                visited.append(next_position)
                current_position = next_position
            elif valid_inherited_moves == []:
                total = sum(probabilities)
                probabilities = [p / total for p in probabilities]
                next_position = random.choices(valid_moves, weights=probabilities, k=1)[0]
                path.append(next_position)
                visited.append(next_position)
                current_position = next_position 
            else:
                mutation = random.uniform(0,1)
                if mutation < self.mutation_rate:
                    total = sum(probabilities)
                    probabilities = [p / total for p in probabilities]
                    next_position = random.choices(valid_moves, weights=probabilities, k=1)[0]
                    path.append(next_position)
                    visited.append(next_position)
                    current_position = next_position
                else:
                    total = sum(probabilities_inherited)
                    probabilities = [p / total for p in probabilities_inherited]
                    next_position = random.choices(valid_inherited_moves, weights=probabilities_inherited, k=1)[0]
                    path.append(next_position)
                    visited.append(next_position)
                    current_position = next_position
        return path




                    


    def solve(self):
        #initial population generation
        current_pop = []
        parent_pop = []
        self.set_probabilities #RL
        for i in range(self.population_size):
            agent = self.move()
            current_pop.append(((len(agent) + 1e-5), agent))
        
        for i in range(self.generations):
            parent_pop = current_pop
            parent_pop.sort(key =lambda x: x[0])
            current_pop = []
            current_pop.append(parent_pop[0])
            current_pop.append(parent_pop[1])
            best_path = len(current_pop[0][1])
            for i in range(self.population_size-1):
                a = random.sample([0,1,2],2)
                parent1 =  parent_pop[a[0]][1]
                parent2 =  parent_pop[a[1]][1]
                child = self.crossover(parent1,parent2)
                current_pop.append(((len(child) + 1e-5),child))
                best_path = min(best_path,len(current_pop[-1][1]))
                print(f'path found of {best_path} length')
                print(len(child))
            self.reinforce(parent_pop[0])
            self.reinforce(parent_pop[1])
            self.reinforce(parent_pop[2])

        current_pop.sort(key =lambda x: x[0])
        GA_solution_dict = {}
        for i in range(len(current_pop[0][1])-1):
            GA_solution_dict[current_pop[0][1][i+1]] =  current_pop[0][1][i]

        return GA_solution_dict

                    


#   P  A  R  T  I  C  L  E    S  W  A  R  M    O  P  T  I  M  I  Z  A  T  I  O  N  
#   
#            *      o      *      o      *  
#      o      \   |   /      *       |       o  
#         *    -- ( ) --     o      ( )      *  
#      o       /   |   \      *       |       o  
#            *      o      *      o      *  
#   
#    === PARTICLE SWARM OPTIMIZATION ===





'''start = (1,1)
end = (rows, columns)

s1 = time.time()
solution = AntColonySolverRL(m.maze_map,start,end, rows,columns)
solution_path = solution.solve()
e1 = time.time()
a = agent(m,footprints=True,filled=True,color = COLOR.green,shape = "square")


m.tracePath({a:solution_path}, delay = 100)
s2 = time.time()
GA_solution = GeneticAlgorithmSolverRL(m.maze_map,start,end,rows,columns)
GA_solution_path = GA_solution.solve()
e2 = time.time()
b = agent(m,footprints=True,filled=True,color = COLOR.blue, shape = "circle")
GA_solution_dict = {}
for i in range(len(GA_solution_path[1])-1):
    GA_solution_dict[GA_solution_path[1][i+1]] =  GA_solution_path[1][i]

print("\n")
print(len(GA_solution_path[1]), "GA time taken = ", e2-s2)
print("\n")
print(len(solution_path), "ACO time taken = ", e1-s1)

m.tracePath({b:GA_solution_dict}, delay = 100)

#print(solution.probs())

m.run()'''
