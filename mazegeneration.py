import random
import numpy
import pygame
import time
import timeit

def isbranch(x,y):
    global path_stack
    global branch_stack
    temp_stack = []
    if 0<=x-1<n and 0<=y<n and array[y][x-1]==1 and not (is_visited(x-2,y)):
        temp_stack.append("l")
    if 0<=x+1<n and 0<=y<n and array[y][x+1]==1 and not (is_visited(x+2,y)):
        temp_stack.append("r")
    if 0<=x<n and 0<=y+1<n and array[y+1][x]==1 and not (is_visited(x,y+2)):
        temp_stack.append("d")
    if 0<=x<n and 0<=y-1<n and array[y-1][x]==1 and not (is_visited(x,y-2)):
        temp_stack.append("u")
    if [x,y,temp_stack] not in branch_stack: branch_stack.append([x,y,temp_stack])
    

def findnode(x,y,queue):
    array[y][x] ==2
    global retrace
    retrace = {}
    temp = []
    if 0<=x-1<n and 0<=y<n and array[y][x-1]==1 and not (is_visited(x-2,y)):
        queue.append((x,y-2))
        temp.append((x,y-2))
    if 0<=x+1<n and 0<=y<n and array[y][x+1]==1 and not (is_visited(x+2,y)):
        queue.append((x+2,y))
        temp.append((x+2,y))
    if 0<=x<n and 0<=y+1<n and array[y+1][x]==1 and not (is_visited(x,y+2)):
        queue.append((x,y+2))
        temp.append((x,y+2))
    if 0<=x<n and 0<=y-1<n and array[y-1][x]==1 and not (is_visited(x,y-2)):
        queue.append((x,y-2))
        temp.append((x,y-2))
    visited.append(queue.pop(0))
    retrace[(x,y)]=temp
    return queue



    





def traverse(x,y):
    global path_stack
    global branch_stack
    array[y][x] = 2
    for i in (branch_stack):
        if i[0] ==x and i[1]==y:
        
            
            try:
                command = i[2][0]
                loc = branch_stack.index(i)
                del(branch_stack[loc][2][0])
                match command:
                    case 'l':
                        if not is_visited(x-2,y):
                            if((x-2),y) not in path_stack: path_stack.append(((x-2),y))
                    case 'r':
                        if not is_visited(x+2,y):
                            if ((x+2,y)) not in path_stack: path_stack.append((x+2,y))
                    case 'd':
                        if not is_visited(x,(y+2)):
                            if ((x,(y+2))) not in path_stack: path_stack.append((x,(y+2)))
                    case 'u':
                        if not is_visited(x,y-2):
                            if ((x,(y-2))) not in path_stack: path_stack.append((x,(y-2)))
            except IndexError:
                branch_stack.remove(i)
                path_stack.pop()

            
                
                 
            

def is_visited(x,y):
    return(array[y][x]==2)



    



#prims algorithm
#generating matrix
def matrix(n):
    matrix = numpy.zeros((n,n))
    return matrix

def addfront(x,y):
    global frontier
    if 0<=(x+2)<n and array[x+2][y]==0 and ((x+2,y) not in frontier):
        frontier.append((x+2,y))
    if 0<=(x-2)<n and array[x-2][y]==0 and ((x-2,y) not in frontier):
        frontier.append((x-2,y))
    if 0<=(y-2)<n and array[x][y-2]==0 and ((x,y-2) not in frontier):
        frontier.append((x,y-2))
    if 0<=(y+2)<n and array[x][y+2]==0 and ((x,y+2) not in frontier):
        frontier.append((x,y+2))

def connect(x,y):
    array[x][y] = 1
    frontier.remove((x,y)) 
    connection = []
    if 0<=(x+2)<n and array[x+2][y]==1:
        connection.append(1)
    if 0<=(x-2)<n and array[x-2][y]==1:
        connection.append(2)
    if 0<=(y-2)<n and array[x][y-2]==1:
        connection.append(3)
    if 0<=(y+2)<n and array[x][y+2]==1:
        connection.append(4)
    c = connection[random.randrange(0,len(connection))]
    match c:
        case 1:
            array[x+1][y] = 1
        case 2:
            array[x-1][y] = 1
        case 3:
            array[x][y-1] = 1
        case 4:
            array[x][y+1] = 1
    


def prims(array):
    #first we select our starting block which is array[0][0]
    global frontier 
    frontier =[]
    x = 0
    y = 0
    array[x][y] = 1
    
    
    

    return array
    
#start of maze generation
n = int(input("order of maze(odd)"))
global frontier
global array    
array = matrix(n)
x = 0
y = 0
array = prims(array)

solution = numpy.copy(array)
addfront(x,y)
while(True):
    
    addfront(x,y)
    
    x,y = frontier[random.randrange(0,len(frontier))] 
    connect(x,y)
    if frontier == []:
        break
print(array)
solution = numpy.copy(array)
#now creating GUI


def redraw(window):
    global size
    global array
    
    
    n = len(array)
    window.fill((0,0,0))

    for i in range(len(array)):
        for j in range(len(array)):
            if array[j][i]== 1:
                pygame.draw.rect(sc, (255,255,255), pygame.Rect((i*dist),(j*dist),dist+1,dist+1))
    grid(window,size,n)
    
    pygame.display.update()   
    
def redraw_solved(window):
    global size
    global array
    
    
    n = len(array)
    window.fill((0,0,0))

    for i in range(len(array)):
        for j in range(len(array)):
            if array[j][i]== 1:
                pygame.draw.rect(sc, (255,255,255), pygame.Rect((i*dist),(j*dist),dist+1,dist+1))
            elif array[j][i]==2:
                pygame.draw.rect(sc, (255,0,0), pygame.Rect((i*dist),(j*dist),dist+1,dist+1))
    grid(window,size,n)
    
    pygame.display.update()   



def displaytext(text,x,y):
    global dist
 #   img = font.render(text,True,(0,0,0))
  #  sc.blit(img,(x,y))




def grid(window,size,rows):
    global dist
    dist = size /rows
    x=0
    y=0

    for l in range(rows):
        x+= dist
        y+=dist
        pygame.draw.line(window,(255,255,255), (x,0),(x,size))
        pygame.draw.line(window,(255,255,255),(0,y),(size,y))
    
       



#execution of GUI
pygame.init()
global size
global dist
size = 700
dist = size/n
sc = pygame.display.set_mode((size,size))

#font = pygame.font.SysFont("Helvetica",int(size//n)-30)



 

'''                 SEARCHING ALGORITHMS                '''
def DFS():
    global branch_stack
    branch_stack = []
    global path_stack
    path_stack = []
    global visited
    visited = []
    
    x = 0 
    y = 0
    traverse(x,y)
    path_stack.append((x,y))
    isbranch(x,y)
    
    while True:

        xnew = path_stack[(len(path_stack)-1)][0]
        ynew = path_stack[(len(path_stack)-1)][1]
        
        if(x!=xnew or y!=ynew):
            x = xnew
            y = ynew
            isbranch(x,y)
        else:
            pass


        
        
        
        if (x==(n-1)) and (y==(n-1)):
            break
        else:
            pass

        traverse(x,y)
    
        
      
        
print(solution)
        
s = timeit.timeit()
DFS()
e = timeit.timeit()

for i in path_stack:
    solution[i[1]][i[0]] = 2
array = solution

        
print(path_stack)
print(s-e)



'''
global endy
global endx
endy = random.randint(0,((n-1)//2))
endx = random.randint(0,((n-1)//2))
endy *=2
endx *=2

def BFS():
    global endy
    global endx
    global retrace
    x,y = 0,0
    global array
    n = len(array)
    queue = []
    queue.append((x,y))
    while queue:
        x = queue[0][0]
        y = queue[0][1]
        if array[endy][endx] ==2:
            break
        queue = findnode(x,y,queue)
        time.sleep(3)
        print(queue)
            


global visited
visited = []
print(BFS())
print(array)
print(visited)

    



'''
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    redraw_solved(sc)
