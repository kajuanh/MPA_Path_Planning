from threading import Thread
import random as rd
import numpy as np
import math

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

def distance(c1: list[int] | tuple[int], c2: list[int] | tuple[int]):
    '''function to calculate distance between 2 points'''
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def check_collision_of_2axis(a: int, b: int, st: list[int] | tuple[int], en: list[int] | tuple[int], environment: np, check: bool):
    '''function check collision by calculating the intersection points of the line segment to the lines:
        x=i, x=i+1, x=i+2 ... x=j (i=st_x, j=en_x) and y=h, y=h+1, ...y=k (h=st_y, k=en_y)'''

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

        if check[0]:
            return check

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


def remove_duplicate(path: list[list[int]] | list[tuple[int, int]]):
    '''function remove duplicate points in a path'''
    duplicate = True
    while duplicate:
        length = len(path)
        index_l = index_r = None
        for cdn in path:
            if path.count(cdn)>1:
                index_l = path.index(cdn)
                index_r = length - 1 - path[::-1].index(cdn)
                break
        if index_l is not None and index_r is not None:
            temp = path[:index_l]
            temp.extend(path[index_r:])
            path = temp
        else:
            duplicate = False
    return path

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
        # self.center = (x+0.5, y+0.5)

    def linear_equations_to(self, other: 'Node'):
        '''linear_equations AB:
        |x = xA+a*t (a = xB - xA)
        |y = yA+b*t (b = yB - yA)

        if t = 1 : x = xB, y = yB => AB ~ (0<delta_t<1)  
        '''     
        a = other.x-self.x
        b = other.y-self.y
        return a, b

    def check_collision(self, other: 'Node', environment: np) -> bool:
        '''check collision between 2 node with environment by collision test 4 straight
        lines connecting 4 corners of node '''
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
            thread_check = [False]
            list_thread = []
            thread = Thread(
                name='Thread_00',
                target=check_collision_of_2axis,
                args=(a, b, self.list, other.list, environment, thread_check,)
            )
            thread.start()
            list_thread.append(thread)
            for i in range(3):
                point_a = self.three_corners[i]
                point_b = other.three_corners[i]
                thread = Thread(
                    name='Thread_%d' % i,
                    target=check_collision_of_2axis,
                    args=(a, b, point_a, point_b, environment, thread_check, ))
                thread.start()
                list_thread.append(thread)
            for thread in list_thread:
                thread.join()
                check = thread_check[0]
        return check


class GridMap:
    def __init__(self, map_size=None, data=[], obstacles=[], empty=[]):
        self.map_size = map_size
        self.data = data
        self.obstacles = []
        self.empty = []
        self.map_collision = np.zeros((map_size,map_size,map_size,map_size))
        self.dict_distance = {}
        self.d_min = 2
    
    def distance_s(self, c1: list[int] | tuple[int], c2: list[int] | tuple[int]):
        '''function to calculate distance between 2 points have save in dict_distance'''
        if type(c1) != type([]):
            c1 = list(c1)
        if type(c2) != type([]):
            c2 = list(c2)

        if (have := self.dict_distance.get('%s%s' % (str(c1), str(c2)))) is not None:
            return have
        else:
            have = distance(c1, c2)
            self.dict_distance['%s%s' % (str(c1), str(c2))] = have
            return have

    def check_outside(self,x:int)->bool:
        if x<0 or x>=self.map_size:
            return True
        else:
            return False

    def check_list_outside(self,xs:list)->bool:
        for x in xs:
            if self.check_outside(x):
                return True
        return False

    def check_collision(self, c_1: list, c_2: list):

        '''case c1 or c2 outside map'''
        if self.check_list_outside([c_1[0], c_1[1], c_2[0], c_2[1]]):
            return True

        '''case c1 == c2'''
        if c_1[0] == c_2[0] and c_1[1] == c_2[1]:
            if self.data[c_1[0], c_1[1]] == 1:
                return True
            return False

        '''case have save in map_collision'''
        if self.map_collision[c_1[0], c_1[1]][c_2[0], c_2[1]] == 1:
            return True
        elif self.map_collision[c_1[0], c_1[1]][c_2[0], c_2[1]] == 2:
            return False

        '''case no have save in map_collision'''
        node_1 = Node(*c_1)
        node_2 = Node(*c_2)
        check = node_1.check_collision(node_2, self.data)

        '''save to map_collision'''
        if check:
            self.map_collision[c_1[0], c_1[1]][c_2[0], c_2[1]] = 1
        else:
            self.map_collision[c_1[0], c_1[1]][c_2[0], c_2[1]] = 2

        return check

    def remove_no_connection(self, path: list[list], end: list[int]):
        '''remove point no have connection with points near it'''
        len_list = len(path)
        remove_index = []
        for i in range(1, len_list-1):
            if self.check_collision(path[i], path[i-1]) and self.check_collision(path[i], path[i+1]):
                remove_index.append(i)
        if len_list>=2:
            if self.check_collision(path[-1], end) or self.check_collision(path[len_list-1], path[len_list-2]):
                remove_index.append(len_list-1)
        remove_index.reverse()
        for i in remove_index:
            path.pop(i)
        return path


    def random_space(self, s: int, center: list, apply_map=None):
        '''return node can chose in apply_map with center on s cells'''
        if apply_map is None:
            apply_map = self.data.copy()
        random_space = []
        for i in range(-s, s+1):
            for j in range(-s, s+1):
                if i == j == 0:
                    continue
                x = center[0]+i
                y = center[1]+j
                if self.map_size <= x or x < 0 or self.map_size <= y or y < 0:
                    continue
                if apply_map[x][y] != 0:
                    continue
                random_space.append([x, y])
        return random_space

