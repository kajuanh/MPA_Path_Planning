from concurrent.futures import thread
import math
from typing import List
from xmlrpc.client import Boolean
import matplotlib
# from typing_extensions import Self
import numpy as np
import time
from threading import Thread
import multiprocessing
import concurrent.futures as con
import pygame
import sys
import random as rd
def distance(c1:list,c2:list):
    return math.sqrt((c1[0]- c2[0])**2 + (c1[1] - c2[0]) ** 2)

class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None,
                args=(), kwargs={}, Verbose=None):
        Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None

    def run(self):
        # print(type(self._target))
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self, *args):
        Thread.join(self, *args)
        return self._return


def cdn_of_2axis(a: int, b: int, st: List, en: List, environment):
    coordinates = []

    if abs(a) == abs(b):
        if a*b > 0:
            if a > 0:
                if environment[st[0]][st[1]] == 1:
                    coordinates.append([st[0], st[1]])
            else:
                if environment[en[0]][en[1]] == 1:
                    coordinates.append([en[0], en[1]])
        else:
            if a < 0:
                if environment[st[0]-1][st[1]] == 1:
                    coordinates.append([st[0]-1, st[1]])
            else:
                if environment[en[0]-1][en[1]] == 1:
                    coordinates.append([en[0]-1, en[1]])

        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            y = math.floor(((x-st[0])/a)*b + st[1])
            if a*b < 0:
                if environment[x-1][y] == 1:
                    coordinates.append([x-1, y])
            else:
                if environment[x][y] == 1:
                    coordinates.append([x, y])

    else:
        # with xaxis
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            test_y = ((x-st[0])/a)*b + st[1]
            y = math.floor(test_y)
            if environment[x][y] == 1 and test_y % 1 != 0:
                coordinates.append([x, y])
            if (test_y % 1 != 0) and environment[x-1][y] == 1:
                coordinates.append([x-1, y])
        # with yaxis
        y_array = (st[1]+1, en[1]) if st[1] < en[1] else (en[1]+1, st[1])
        for y in range(*y_array):
            test_x = (((y-st[1])/b)*a + st[0])
            x = math.floor(test_x)
            if environment[x][y] == 1 and test_x % 1 != 0:
                coordinates.append([x, y])
            if (test_x % 1 != 0) and environment[x][y-1] == 1:
                coordinates.append([x, y-1])
    return list(set(tuple(x) for x in coordinates))


def check_collision_of_2axis(a: int, b: int, st: List, en: List, environment, check: bool):
    if check[0]:
        return check

    if abs(a) == abs(b):
        if a*b > 0:
            if a > 0:
                if environment[st[0]][st[1]] == 1:
                    check[0] = True

            else:
                if environment[en[0]][en[1]] == 1:
                    check[0] = True

        else:
            if a < 0:
                if environment[st[0]-1][st[1]] == 1:
                    check[0] = True

            else:
                if environment[en[0]-1][en[1]] == 1:
                    check[0] = True

        if check[0]:
            return check

        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            y = math.floor(((x-st[0])/a)*b + st[1])
            if a*b < 0:
                if environment[x-1][y] == 1:
                    check[0] = True
                    break

            else:
                if environment[x][y] == 1:
                    check[0] = True
                    break

    else:
        # with xaxis
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            test_y = ((x-st[0])/a)*b + st[1]
            y = math.floor(test_y)
            if environment[x][y] == 1 and test_y % 1 != 0:
                check[0] = True
                break
            if (test_y % 1 != 0) and environment[x-1][y] == 1:
                check[0] = True
                break
        # with yaxis
        y_array = (st[1]+1, en[1]) if st[1] < en[1] else (en[1]+1, st[1])
        for y in range(*y_array):
            test_x = (((y-st[1])/b)*a + st[0])
            x = math.floor(test_x)
            if environment[x][y] == 1 and test_x % 1 != 0:
                check[0] = True
                break
            if (test_x % 1 != 0) and environment[x][y-1] == 1:
                check[0] = True
                break
    return check

class Node:
    def __init__(self, x: int, y: int) -> None:
        '''a node have coordinates (x,y)
        is a square as :
        (x,y)___________(x_r,y_r)
            |                 |
            |    x_c,y_c      |
        (x_b,y_b)________(x_br,y_br)
        '''
        self.x = x
        self.y = y
        self.list =[x,y]
        self.three_corners = [(x+1, y), (x, y+1), (x+1, y+1)]
        self.center = (x+0.5, y+0.5)

    def linear_equations_to(self,other:'Node'):
        '''linear_equations AB:
        |x = xA+a*t (a = xB - xA)
        |y = yA+b*t (b = yB - yA)

        if t = 1 : x = xB, y = yB => AB ~ (0<delta_t<1)  
        '''     
        a = other.x-self.x
        b = other.y-self.y
        return a, b
    


    def check_collision(self, other: 'Node', environment: np) -> Boolean:
        check = False

        if environment[self.x][self.y] == 1 or environment[other.x][other.y] == 1:
            check = True
            return check

        a, b = self.linear_equations_to(other)
        if a == 0:
            y_array = (self.y+1, other.y) if self.y < other.y else (other.y+1, self.y)
            for y in range(*y_array):
                if environment[self.x][y] == 1:
                    check = True
                    return check

        elif b == 0:
            x_array = (self.x+1, other.x) if self.x < other.x else (other.x+1, self.x)
            for x in range(*x_array):
                if environment[x][self.y] == 1:
                    check = True
                    return check
        else:
            thread_check =[False]
            list_thread = []
            thread = Thread(
                name='Thread_00',
                target=check_collision_of_2axis,
                args=(a, b, self.list, other.list, environment,thread_check,  )
            )
            thread.start()
            list_thread.append(thread)
            for i in range(3):
                point_a = self.three_corners[i]
                point_b = other.three_corners[i]
                thread = Thread(
                    name='Thread_%d' % i,
                    target=check_collision_of_2axis,
                    args=(a, b, point_a, point_b, environment,thread_check, ))
                thread.start()
                list_thread.append(thread)
            for thread in list_thread:
                thread.join()
                check=thread_check[0]
        return check
        
    def collision_coordinates(self, other: 'Node', environment: np) -> List[List]:
        coordinates = []
        a, b = self.linear_equations_to(other)
        if a == 0:
            y_array = (self.y+1, other.y) if self.y < other.y else (other.y+1, self.y)
            for y in range(*y_array):
                if environment[self.x][y] == 1:
                    coordinates.append([self.x,y])
            return coordinates

        elif b == 0:
            x_array = (self.x+1, other.x) if self.x < other.x else (other.x+1, self.x)
            for x in range(*x_array):
                if environment[x][self.y] == 1:
                    coordinates.append([x, self.y])
            return coordinates
        else:
            list_thread = []
            thread = ThreadWithReturnValue(
                name='Thread_00',
                target=cdn_of_2axis,
                args=(a, b, self.list, other.list, environment, )
            )
            thread.start()
            list_thread.append(thread)
            for i in range(3):
                point_a = self.three_corners[i]
                point_b = other.three_corners[i]
                thread = ThreadWithReturnValue(
                    name='Thread_%d' % i,
                    target=cdn_of_2axis,
                    args=(a, b, point_a, point_b, environment, ))
                thread.start()
                list_thread.append(thread)
            for thread in list_thread:
                coordinates.extend(thread.join())
        return coordinates


class GridMap:
    def __init__(self,map_size=None,data=[],obstacles=[],empty=[]):
        self.map_size = map_size
        self.data = data
        self.obstacles = []
        self.empty = []

    def collision_coordinates(self, c_1:list,c_2: list):
        node_1 = Node(*c_1)
        node_2 = Node(*c_2)
        collision_coordinates = node_1.collision_coordinates(node_2,self.data)
        return collision_coordinates, node_1, node_2

    def check_collision(self, c_1:list, c_2: list):
        node_1 = Node(*c_1)
        node_2 = Node(*c_2)
        check = node_1.check_collision(node_2,self.data)
        return check

    def display_matplotlib(self,points:list[list],time = 100):
        import matplotlib.pyplot as plt

        for c1,c2 in points:
            temp_data = self.data.copy()
            collision_coordinates, node_1, node_2 = self.collision_coordinates(c1,c2)
            for index in collision_coordinates:
                temp_data [index[0]][index[1]] = 2
            fig, ax = plt.subplots()
            fig.set_size_inches(10,10)
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.set_xlim(0, self.map_size+1)
            ax.set_ylim(0, self.map_size+1)
            plt.xticks([*range(self.map_size+1)])
            plt.yticks([*range(self.map_size+1)])
            plt.imshow(temp_data.transpose(),origin='lower',extent = (0,self.map_size,0,self.map_size))
            plt.plot((node_1.list[0],node_2.list[0]),(node_1.list[1],node_2.list[1]),'--')
            for i in range(3):
                point_a = node_1.three_corners[i]
                point_b = node_2.three_corners[i]
                plt.plot((point_a[0],point_b[0]),(point_a[1],point_b[1]),'--')
            plt.grid()
            ax.invert_yaxis()

            plt.show(block=False)
            plt.pause(time)
            plt.close()

    def display_pygame(self,points:list[list]):
        BLACK = (0, 0, 0)
        WHITE = (200, 200, 200)
        YELLOW = (215, 225, 88)
        GRAY = (150, 150, 100)
        BLOCKSIZE = 40
        WINDOW_WIDTH = WINDOW_HEIGHT = self.map_size*BLOCKSIZE
        pygame.init()
        SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        CLOCK = pygame.time.Clock()
        SCREEN.fill(BLACK)
        point_index = 0
        limit = len(points)
        while True:
            SCREEN.fill(BLACK)
            temp_data = self.data.copy()
            c1 = c2 =None
            if point_index<limit:
                c1 , c2 = points[point_index]
                temp_data [c1[0]][c1[1]] = 3
                temp_data [c2[0]][c2[1]] = 3
                
                collision_coordinates, node_1, node_2 = self.collision_coordinates(c1,c2)
                if not (check:= self.check_collision(c1,c2)):
                    print(check)
                for index in collision_coordinates:
                    temp_data [index[0]][index[1]] = 2

            for x in range(self.map_size):
                for y in range(self.map_size):
                    rect = pygame.Rect(x*BLOCKSIZE, y*BLOCKSIZE,
                                    BLOCKSIZE-1, BLOCKSIZE-1)
                    if temp_data[x][y] == 1:
                        pygame.draw.rect(SCREEN, GRAY, rect)
                    elif temp_data[x][y]==2:
                        pygame.draw.rect(SCREEN, YELLOW, rect)
                    elif temp_data[x][y]==3:
                        pygame.draw.rect(SCREEN, (0,240,0), rect)
                    else:
                        pygame.draw.rect(SCREEN, WHITE, rect, 1)
            
            if c1 is not None and c2 is not None:
                pygame.draw.line(SCREEN,(230,0,0),np.array(c1)*BLOCKSIZE,np.array(c2)*BLOCKSIZE,2)
                for i in range(3):
                    point_a = np.array(node_1.three_corners[i])*BLOCKSIZE
                    point_b = np.array(node_2.three_corners[i])*BLOCKSIZE
                    pygame.draw.line(SCREEN,(230,0,0),point_a,point_b,2)
            pygame.display.flip()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            point_index += 1
            CLOCK.tick(5)
        



            
class Path:
    def __init__(self,environment: GridMap,path:list[list[list]] = None):
        self.path = path
        if path is not None:
            self.amount = len(path)  
            self.distance = self.func_distance(path)
        else:
            self.amount = self.distance = 0
        self.environment = environment
    def func_distance(self,path:list[list[list]]):
        dis = 0
        for index in range(len(path)-1):
            dis += distance(path[index],path[index+1])
        return dis
    
    def check_collision(self, c_1: list, c_2: list):
        return self.environment.check_collision(c_1, c_2)
    
    def shorten(self,origin_sol:list[list[list]],start:list[list],end:list[list]):
        reduce_sol = [start]
        index = 0
        l_origin_sol = len(origin_sol)
        temp_sol = None
        while self.check_collision(reduce_sol[-1], end):
            for i in range(index, l_origin_sol):
                if not self.check_collision(origin_sol[i], reduce_sol[-1]):
                    temp_sol = origin_sol[i]
                    index = i+1
            reduce_sol.append(temp_sol)
        origin_sol = reduce_sol
        reduce_sol = [end]
        index = len(origin_sol)-1
        while self.check_collision(reduce_sol[-1], start):
            for i in range(index, 0, -1):
                if not self.check_collision(origin_sol[i], reduce_sol[-1]):
                    temp_sol = origin_sol[i]
                    index = i-1
            reduce_sol.append(temp_sol)
        reduce_sol.reverse()
        reduce_sol.insert(0, start)
        return reduce_sol

    def random_space(self, s: int, center: list, apply_map = None):
        if apply_map is None:
            apply_map = self.environment.data
        random_space = []
        for i in range(-s, s+1):
            for j in range(-s, s+1):
                if i == j == 0:
                    continue
                x = center[0]+i
                y = center[1]+j
                if self.environment.map_size <= x or x < 0 or self.environment.map_size <= y or y < 0:
                    continue
                if apply_map[x,y] != 0:
                    continue
                random_space.append([x, y])
        return random_space


    def random_init(self, start: list[int, int], end: list[int, int]):
        origin_sol = [start]
        temp_map = self.environment.data.copy()
        if temp_map[start[0],start[1]]==1 or temp_map[end[0],end[1]]==1 :
            print('start is not node empty')
            return
        else:
            temp_map[start[0],start[1]] = 3
        while(self.check_collision(origin_sol[-1],end)):
            rd_values = self.random_space(2,origin_sol[-1],temp_map)
            while True:
                if len(rd_values) == 0:
                    origin_sol.pop()
                    break
                selected = rd.choice(rd_values)
                rd_values.remove(selected)

                if not self.check_collision(selected, origin_sol[-1]):
                    origin_sol.append(selected)
                    temp_map[selected[0],selected[1]] = 3
                    break

            if len(origin_sol) == 0:
                origin_sol = [start]
                temp_map.data = self.environment.data
                temp_map.data[start[0],start[1]] = 3
        reduce_sol = self.shorten(origin_sol,start,end)
        return reduce_sol

class MPAs:
    def __init__(self,environment:GridMap):
        self.environment = environment
        self.class_path = Path(self.environment)
        
    def check_collision(self, c_1, c_2):
        return self.environment.check_collision(c_1, c_2)

    def init_population(self, n_child: int, start, end):
        origin_sol = []
        for _ in range(n_child):
            origin_sol.append(self.class_path.random_init(start, end))
        return origin_sol

    def calculator(self, f_sol, f_st, f_dst):
        s = 0
        is_dst = False
        pre_x = f_st
        for f_x in f_sol:
            if self.check_collision(pre_x, f_x):
                return 1, s
            s += distance(pre_x, f_x)
            if not self.check_collision(f_x, f_dst):
                s += distance(f_x, f_dst)
                is_dst = True
                break
            pre_x = f_x
        if not is_dst:
            if self.check_collision(f_sol[-1], f_dst):
                return 1, s
            s += distance(f_sol[-1], f_dst)
        return 0, s

    def way(self, st, dst):
        n_child = self.environment.map_size
        min_s = math.inf
        prey = []
        best_prey = []
        old_s = []
        max_d = 0

        if not self.environment.check_collision(st, dst):
            return distance(st, dst), [st, dst]
        origin_sol = self.init_population(n_child, st, dst)
        for iPrey in origin_sol:
            sol_d = len(iPrey)
            if max_d < sol_d:
                max_d = sol_d
            v, dis_prey = self.calculator(iPrey, st, dst)
            old_s.append(dis_prey)
            prey.append(iPrey)
            if min_s > dis_prey:
                min_s = dis_prey
                best_prey = list(iPrey)
            for i in prey:
                i.extend([dst for _ in range(max_d-len(i))])
            a_prey = np.array(prey)
        print('best_prey:', best_prey)
        print('a_prey:', a_prey.T)

data = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]

environment = GridMap(15,np.array(data).transpose())
'''list_point = []
list_right = []
list_left = []
list_top = []
list_bottom = []

for i in range(15):
    list_right.append([[6, 6], [14, i]])
    list_left.append([[6, 6], [0, 14-i]])
    list_top.append([[6, 6], [i, 0]])
    list_bottom.append([[6, 6], [14-i, 14]])

list_point.extend(list_right)
list_point.extend(list_bottom)
list_point.extend(list_left)
list_point.extend(list_top)

# environment.display_matplotlib(list_point)
# environment.display_pygame(list_point)'''
# environment.display_matplotlib([[[5,5],[6,6]],
#                                 [[6,6],[5,5]],
#                                 [[8,7],[9,6]],
#                                 [[9,6],[8,7]]])
import matplotlib.pyplot as plt
def draw_line(points:list[list],plt:matplotlib.pyplot):
    xl = []
    yl = []
    for point in points:
        xl.append(point[0]+0.5)
        yl.append(point[1]+0.5)
    plt.plot(xl,yl,'-')

# path = Path(environment)
temp_data = environment.data.copy()
# print(temp_data.T)
# print(temp_data.T)
st = time.time()
mpa_obj = MPAs(environment)
origin_sol = mpa_obj.init_population(environment.map_size,[1,2],[11,11])
en = time.time()
print(en-st)
print(len(origin_sol))
fig, ax = plt.subplots()
fig.set_size_inches(10,10)

ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
ax.set_xlim(0, environment.map_size+1)
ax.set_ylim(0, environment.map_size+1)
plt.xticks([*range(environment.map_size+1)])
plt.yticks([*range(environment.map_size+1)])
for points in origin_sol:
    draw_line(points,plt)
plt.imshow(temp_data.transpose(),origin='lower',extent = (0,environment.map_size,0,environment.map_size))
plt.grid()
ax.invert_yaxis()

plt.show(block=False)
plt.pause(10)
st = time.time()
mpa_obj.way([1, 2], [11, 11])
en = time.time()
print(en-st)

