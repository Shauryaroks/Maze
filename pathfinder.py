from pathfind import AntColonySolver, GeneticAlgorithmSolver
from pathfind_RL import GeneticAlgorithmSolverRL, AntColonySolverRL
import random
from pyamaze import maze, agent, COLOR
import numpy as np
import time
import csv
import matplotlib.pyplot as plt


def add_data(algo,runtime,path):
    global rows
    global columns
    global Data
    dict = {}
    dict["algorithm"] = algo
    dict["Rows"] = rows
    dict["columns"] = columns
    dict["runtime"] = runtime
    dict["PathLength"] = path
    Data.append(dict)

#data = ["algorithm","Rows", "columns","runtime"]

rows = int(input("no of rows"))
columns = int(input("no of columnss"))
start = (1,1)
end = (rows, columns)
maze = maze(rows,columns)
maze.CreateMaze(loopPercent=50)
num_agents = 20 
generations = 20

Data = []
Dict = {}

#checking ACO WITHOUT RL AND WITH RL FIRST

s1 = time.time()
ACOsolution = AntColonySolver(maze.maze_map,start,end, rows,columns,num_ants=num_agents,num_iterations=generations)
ACOpath = ACOsolution.solve()
e1 = time.time()
a = agent(maze,footprints=True,filled=True,color = COLOR.green,shape = "square")
add_data("ACO",e1-s1,len(ACOpath))

maze.tracePath({a:ACOpath}, delay = 100)

s2 = time.time()
ACOsolutionRL = AntColonySolverRL(maze.maze_map,start,end, rows,columns,num_ants=num_agents,num_iterations=generations)
ACOpathRL = ACOsolutionRL.solve()
e2 = time.time()
b = agent(maze,footprints=True,filled=False,color = COLOR.cyan,shape = "square")
add_data("ACORL",e2-s2,len(ACOpathRL))

maze.tracePath({b:ACOpathRL}, delay = 100)

s3 = time.time()
GA_solution = GeneticAlgorithmSolver(maze.maze_map,start,end,rows,columns,population_size=num_agents,generations=generations)
GApath = GA_solution.solve()
e3 = time.time()
c = agent(maze,footprints=True,filled=True,color = COLOR.red,shape = "square")
add_data("GA",e3-s3,len(GApath))

maze.tracePath({c:GApath}, delay = 100)

s4 = time.time()
GApathRL = GeneticAlgorithmSolverRL(maze.maze_map,start,end,rows,columns,population_size=num_agents,generations=generations)
GApathRL = GApathRL.solve()
e4 = time.time()
d = agent(maze,footprints=True,filled=False,color = COLOR.yellow, shape = "square" )
add_data("GARL",e4-s4,len(GApathRL))

maze.tracePath({d:GApathRL}, delay = 100)

with open('data.csv', 'a+', newline='') as csvfile:
    fieldnames = ["algorithm","Rows", "columns","runtime","PathLength"]
    writer = csv.DictWriter(csvfile,fieldnames = fieldnames)
    writer.writeheader()
    writer.writerows(Data)

maze.run()