from concurrent.futures import thread
import math
from tabnanny import check
import threading
from typing import List
from xmlrpc.client import Boolean
# from typing_extensions import Self
import numpy as np
import time
from threading import Thread
import multiprocessing
import concurrent.futures as con
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
    if abs(a)==abs(b):
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            y = math.floor(((x-st[0])/a)*b + st[1])
            if a*b<0:
                if environment[x-1][y] == 1:
                    coordinates.append([x-1, y])
                if a<b:
                    if environment[st[0]-1][st[1]] == 1:
                        coordinates.append([st[0]-1, st[1]])
            else:
                if environment[x][y] == 1:
                    coordinates.append([x, y])
                if a>0:
                    if environment[st[0]+1][st[1]] == 1:
                        coordinates.append([st[0]+1, st[1]])
                    if environment[st[0]][st[1]+1] == 1:    
                        coordinates.append([st[0], st[1]+1])
                else:
                    if environment[en[0]][en[1]+1] == 1:    
                        coordinates.append([en[0],en[1]+1])
                    if environment[en[0]+1][en[1]] == 1:    
                        coordinates.append([en[0]+1,en[1]])

                    
    else:           
        # with xaxis
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            test_y = ((x-st[0])/a)*b + st[1]
            y = math.floor(test_y)
            if environment[x][y] == 1 and test_y%1!=0:
                coordinates.append([x, y])
            if (test_y%1!=0) and environment[x-1][y] ==1:
                coordinates.append([x-1, y])
        # with yaxis
        y_array = (st[1]+1, en[1]) if st[1] < en[1] else (en[1]+1, st[1])
        for y in range(*y_array):
            test_x = (((y-st[1])/b)*a + st[0])
            x = math.floor(test_x)
            if environment[x][y] == 1 and test_x%1!=0:
                coordinates.append([x, y])
            if (test_x%1!=0) and environment[x][y-1] ==1:
                coordinates.append([x, y-1])
    return list(set(tuple(x) for x in coordinates))

def check_collision_of_2axis(a: int, b: int, st: List, en: List, environment, check: bool):
    if check[0]: 
        return check

    if abs(a)!=abs(b):
        # with xaxis
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            test_y = ((x-st[0])/a)*b + st[1]
            y = math.floor(test_y)
            if environment[x][y] == 1 and test_y%1!=0:
                check[0] = True
                break
            if (test_y%1!=0) and environment[x-1][y] ==1:
                check[0] = True
                break
        # with yaxis
        y_array = (st[1]+1, en[1]) if st[1] < en[1] else (en[1]+1, st[1])
        for y in range(*y_array):
            test_x = (((y-st[1])/b)*a + st[0])
            x = math.floor(test_x)
            if environment[x][y] == 1 and test_x%1!=0:
                check[0]= True
                break
            if (test_x%1!=0) and environment[x][y-1] ==1:
                check[0] = True
                break
    else:
        x_array = (st[0]+1, en[0]) if st[0] < en[0] else (en[0]+1, st[0])
        for x in range(*x_array):
            y = int(((x-st[0])/a)*b + st[1])
            if a*b<0:
                if environment[x-1][y] == 1:
                    check[0] = True
                    break
            else:
                if environment[x][y] == 1:
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
        self.max_size = map_size
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

    def display_matplotlib(self,points:list[list]):
        import matplotlib.pyplot as plt

        for c1,c2 in points:
            temp_data = self.data.copy()
            collision_coordinates, node_1, node_2 = self.collision_coordinates(c1,c2)
            for index in collision_coordinates:
                temp_data [index[0]][index[1]] = 2
            fig, ax = plt.subplots()
            fig.set_size_inches(10,10)
            ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
            ax.set_xlim(0, self.max_size+1)
            ax.set_ylim(0, self.max_size+1)
            plt.xticks([*range(self.max_size+1)])
            plt.yticks([*range(self.max_size+1)])
            plt.imshow(temp_data.transpose(),origin='lower',extent = (0,self.max_size,0,self.max_size))
            plt.plot((node_1.list[0],node_2.list[0]),(node_1.list[1],node_2.list[1]),'--')
            for i in range(3):
                point_a = node_1.three_corners[i]
                point_b = node_2.three_corners[i]
                plt.plot((point_a[0],point_b[0]),(point_a[1],point_b[1]),'--')
            plt.grid()
            ax.invert_yaxis()

            plt.show(block=False)
            plt.pause(1)
            plt.close()        
class Path:
    def __init__(self,path:list[list] = None):
        self.path = path
        self.amount = len(path)
        self.distance = self.func_distance(path)

    def func_distance(self):
        dis = 0
        for index in range(self.amount-1):
            dis += distance(self.path[index],self.path[index+1])
        return dis
    
    def random_init(self,environment : GridMap, start:list,end:list):
        pass
        # while ()        

data = [
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0],
    [1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
]

environment = GridMap(15,np.array(data).transpose())
list_point = []
list_right = []
list_left = []
list_top = []
list_bottom = []

for i in range(15):
    list_right.append([[6, 6], [14, i]])
    list_left.append([[6, 6], [0, i]])
    list_top.append([[6, 6], [i, 0]])
    list_bottom.append([[6, 6], [i, 14]])

list_point.extend(list_right)
list_point.extend(list_bottom)
list_point.extend(list_left)
list_point.extend(list_top)

environment.display_matplotlib(list_point)
