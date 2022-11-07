from threading import Thread
from scipy.stats import levy
from typing import List
import random as rd
import numpy as np
import math


# def display_array(array):
#     for i in array:
#         print(i.tolist())

def distance(c1: list, c2: list):
    return math.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[0]) ** 2)

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

def remove_duplicate(path :list[list]):
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

    def linear_equations_to(self,other:'Node'):
        '''linear_equations AB:
        |x = xA+a*t (a = xB - xA)
        |y = yA+b*t (b = yB - yA)

        if t = 1 : x = xB, y = yB => AB ~ (0<delta_t<1)  
        '''     
        a = other.x-self.x
        b = other.y-self.y
        return a, b
    


    def check_collision(self, other: 'Node', environment: np) -> bool:
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
        if c_1[0] == c_2[0] and c_1[1]==c_2[1]:
            if self.data[c_1[0],c_1[1]] == 1:
                return True
            return False

        if self.check_list_outside([c_1[0], c_1[1], c_2[0], c_2[1]]):
            return True

        if self.map_collision[c_1[0],c_1[1]][c_2[0],c_2[1]] == 1:
            return True
        elif self.map_collision[c_1[0],c_1[1]][c_2[0],c_2[1]] == 2:
            return False
        node_1 = Node(*c_1)
        node_2 = Node(*c_2)
        check = node_1.check_collision(node_2, self.data)
        if check:
            self.map_collision[c_1[0],c_1[1]][c_2[0],c_2[1]] = 1
        else:
            self.map_collision[c_1[0],c_1[1]][c_2[0],c_2[1]] = 2
        return check
    
    def remove_no_connection(self,path :list[list],end:list[int]):
        len_list = len(path)
        remove_index = []
        for i in range(1,len_list-1):
            if self.check_collision(path[i],path[i-1]) and self.check_collision(path[i],path[i+1]):
                remove_index.append(i)
        if self.check_collision(path[-1],end) and self.check_collision(path[-1],path[-2]):
                remove_index.append(len_list-1)
        remove_index.reverse()
        for i in remove_index:
            path.pop(i)
        return path


    # def shorten(self, origin_sol: list[list[list]], start: list[list], end: list[list]):
    #     reduce_sol = [start]
    #     index = 0
    #     l_origin_sol = len(origin_sol)
    #     temp_sol = None
    #     while self.check_collision(reduce_sol[-1], end):
    #         for i in range(index, l_origin_sol):
    #             if not self.check_collision(origin_sol[i], reduce_sol[-1]):
    #                 temp_sol = origin_sol[i]
    #                 index = i+1
    #         reduce_sol.append(temp_sol)
    #     origin_sol = reduce_sol
    #     reduce_sol = [end]
    #     index = len(origin_sol)-1
    #     while self.check_collision(reduce_sol[-1], start):
    #         for i in range(index, 0, -1):
    #             if not self.check_collision(origin_sol[i], reduce_sol[-1]):
    #                 temp_sol = origin_sol[i]
    #                 index = i-1
    #         reduce_sol.append(temp_sol)
    #     reduce_sol.reverse()
    #     # reduce_sol.insert(0, start)
    #     return reduce_sol
    def shorten(self, path: list[list[int,int]],start: list[int,int],end: list[int,int]):
        len_list = len(path)
        index_l = 0
        index_r = len_list-1
        for i in range(len_list):
            if path[i][0]==start[0]and path[i][1]==start[1]:
                index_l = i+1
        for j in range(index_r,-1,-1):
            if path[j][0]==end[0] and path[j][1]==end[1]:
                index_r = j-1
        path_trip = path[index_l:index_r+1]
        path_trip = self.remove_no_connection(path_trip,end)
        path_trip = remove_duplicate(path_trip)
        len_list = len(path_trip)
        index_l = tem_id = 0
        index_r = len_list-1
        left = [start]
        right = [end]
        t=0
        while t<100 and self.check_collision(left[-1],right[-1]):
            far_l = near_r = None
            for i in range(index_l, index_r+1):
                if not self.check_collision(left[-1], path_trip[i]) and not (left[-1][0] == path_trip[i][0] and left[-1][1] == path_trip[i][1]):
                    far_l = path_trip[i]
                    tem_id = i+1
            if far_l is not None:
                left.append(far_l)
                index_l = tem_id

            if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1] or not self.check_collision(left[-1],right[-1]):
                break

            for j in range(index_r,index_l-1,-1):
                if not self.check_collision(right[-1], path_trip[j]) and not (right[-1][0] == path_trip[j][0] and right[-1][1] == path_trip[j][1]) :
                    near_r = path_trip[j]
                    tem_id = j-1
            if near_r is not None:
                right.append(near_r)
                index_r = tem_id
            
            if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1]or not self.check_collision(left[-1],right[-1]):
                break
            t+=1
        if t==100:
            print(path)
            raise('loi')
        if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1]:
            right.pop()
        left.pop(0)
        right.reverse()
        left.extend(right)
        left = remove_duplicate(left)
        return left

    def random_space(self, s: int, center: list, apply_map=None):
        if apply_map is None:
            apply_map = self.data
        random_space = []
        for i in range(-s, s+1):
            for j in range(-s, s+1):
                if i == j == 0:
                    continue
                x = center[0]+i
                y = center[1]+j
                if self.map_size <= x or x < 0 or self.map_size <= y or y < 0:
                    continue
                if apply_map[x, y] != 0:
                    continue
                random_space.append([x, y])
        return random_space

    def random_init(self, start: list[int, int], end: list[int, int]):
        origin_sol = [start]
        temp_map = self.data.copy()
        limit = 1000
        lim = 0
        if temp_map[start[0], start[1]] == 1 or temp_map[end[0], end[1]] == 1:
            print('start or end is not node empty')
            return
        else:
            temp_map[start[0], start[1]] = 3
        while (self.check_collision(origin_sol[-1], end)):
            # print('running')
            rd_values = self.random_space(2, origin_sol[-1], temp_map)
            # print(rd_values)
            while True:
                if len(rd_values) == 0:
                    origin_sol.pop()
                    break
                selected = rd.choice(rd_values)
                rd_values.remove(selected)

                if not self.check_collision(selected, origin_sol[-1]):
                    origin_sol.append(selected)
                    temp_map[selected[0], selected[1]] = 3
                    break
                
            if len(origin_sol) == 0 or lim>limit:
                origin_sol = [start]
                temp_map = self.data.copy()
                temp_map[start[0], start[1]] = 3
                lim = 0
            lim += 1
        reduce_sol = self.shorten(origin_sol, start, end)
        return reduce_sol

def read_file(filepath):
    with open(filepath, "r") as file:
        #area = map_size*map_size (map_size located on line 1 of the file)
        map_size = int(file.readline())

        #number of goal (num_goal located on line 2 of the file)
        num_goal = int(file.readline())

        #get coordinates of goal on (num_goal) line next and reverse
        list_goal = []
        for i in range(num_goal):
            goal = file.readline().strip()
            goal_coordinates = list(map(int, goal.split()))
            list_goal.append((goal_coordinates[1], goal_coordinates[0]))

        #get map from file
        np_map = np.zeros((map_size, map_size), int)
        for line in range(map_size):
            np_map[line] = (file.readline()).strip().split()

        #convert to window coordinate system
        np_map = np_map.transpose()

        #get list obstacle from map
        list_obstacle = []
        list_empty = []
        for i in range(map_size):
            for j in range(map_size):
                if np_map[i][j] == 1:
                    list_obstacle.append((i, j))
                else:
                    list_empty.append((i, j))
                    
        #change node goal to empty
        for goal in list_goal:
            np_map[int(goal[0])][int(goal[1])] = 0

        #save data about map
        environment = GridMap(map_size,np_map,list_obstacle,list_empty)
        print('SUCCESS read file')
        return environment, list_goal
class MPAs(GridMap):
    x_min = 1
    x_max = 5
    d_min = 2
    origin = 0.4

    def init_population(self, n_child: int, start, end):
        origin_sol = []
        for _ in range(n_child):
            origin_sol.append(self.random_init(start, end))
            print('random_init: %d/%d'%(_+1,n_child),end='\r')
        print()
        return origin_sol

    def check(self, f_X1, f_X2):
        f_x = f_X1[0]
        f_y = f_X1[1]
        pre_x = f_X2[0]
        pre_y = f_X2[1]
        if f_x > self.map_size or f_x<0:
            f_x = pre_x
        if f_y > self.map_size or f_y<0:
            f_y = pre_y
        return int(f_x), int(f_y)

    def normal_search(self, f_sol):
        f_ns_sol = []
        for f_i in f_sol:
            randoms = self.random_space(1, f_i)
            if len(randoms) > 0:
                selected = rd.choice(randoms)
                f_ns_sol.append(selected)
            else:
                f_ns_sol.append(f_i)
        return f_ns_sol

    def evolution(self, f_father, f_mother, f_st, f_dst):
        f_child = [f_st]
        for f_i in range(len(f_father)):
            if self.check_collision(f_child[-1], f_father[f_i]):
                f_child.append(f_mother[f_i])
            elif self.check_collision(f_child[-1], f_mother[f_i]):
                f_child.append(f_father[f_i])
            else:
                if distance(f_child[-1], f_father[f_i]) < distance(f_child[-1], f_mother[f_i]):
                    f_child.append(f_father[f_i])
                else:
                    f_child.append(f_mother[f_i])
            if not self.check_collision(f_child[-1], f_dst):
                break
        f_child.remove(f_st)
        for f_i in range(len(f_father) - len(f_child)):
            f_child.append(f_dst)
        return f_child


    def  calculator(self, f_sol, f_st, f_dst):
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
        n_child = self.map_size
        min_s = math.inf
        prey = []
        best_prey = []
        old_s = []
        max_d = 0

        if not self.check_collision(st, dst):
            print('no have collision (start-end) ')
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

        best_prey.extend([dst for _ in range(max_d-len(best_prey))])
        d = max_d
        X_min = np.array([[self.x_min, self.x_min] for _ in range(d)])
        X_max = np.array([[self.x_max, self.x_max] for _ in range(d)])
        prey = np.array(prey)
        old_prey = np.array(prey)
        loop = 50
        levy.a = 1.5
        levy.b = 1
        P = np.array([[0.5, 0.5] for _ in range(d)])

        for index in range(loop):
            for i in range(n_child):
                new_v, new_dis = self.calculator(prey[i], st, dst)
                if new_v == 0:
                    if new_dis < old_s[i]:
                        old_s[i] = new_dis
                    else:
                        prey[i] = np.array(old_prey[i])
                    if new_dis < min_s:
                        min_s = new_dis
                        print(min_s)
                        best_prey = np.array(prey[i])
                else:
                    prey[i] = np.array(old_prey[i])
            Elite = np.array([list(best_prey) for _ in range(n_child)])
            old_prey = np.array(prey)

            cf = math.pow(1 - index / loop, 2 * index / loop)
            CF = np.array([[cf, cf] for _ in range(d)])
            for i in range(n_child):
                pre_prey = []
                rBx = np.random.normal(0, 1, d)
                rBy = np.random.normal(0, 1, d)
                Rb = np.array([[rBx[j], rBy[j]] for j in range(d)])
                rLx = levy.rvs(0, 1, d)
                rLy = levy.rvs(0, 1, d)
                Rl = np.array([[rLx[j], rLy[j]] for j in range(d)])
                rx = np.random.uniform(0, 1, d)
                ry = np.random.uniform(0, 1, d)
                R = np.array([[rx[j], ry[j]] for j in range(d)])
                if index < loop / 3:
                    step = Rb * (Elite[i] - Rb * prey[i])
                    pre_prey = prey[i] + P * R * step
                elif loop / 3 <= index < 2 * loop / 3:
                    if i < self.map_size / 2:
                        step = Rl * (Elite[i] - Rl * prey[i])
                        pre_prey = prey[i] + P * R * step
                    else:
                        step = Rb * (Rb * Elite[i] - prey[i])
                        pre_prey = Elite[i] + P * CF * step
                elif index >= 2 * loop / 3:
                    step = Rl * (Rl * Elite[i] - prey[i])
                    pre_prey = Elite[i] + P * CF * step
                for j in range(d):
                    pre_prey[j] = self.check(pre_prey[j], prey[i][j])
                prey[i] = np.array(pre_prey)
            # print('prey:')
            # display_array(prey)

            for i in range(n_child):
                new_v, new_dis = self.calculator(prey[i], st, dst)
                if new_v == 0:
                    if new_dis < old_s[i]:
                        old_s[i] = new_dis
                    else:
                        prey[i] = np.array(old_prey[i])
                    if new_dis < min_s:
                        min_s = new_dis
                        best_prey = np.array(prey[i])
                        print(min_s)
                else:
                    prey[i] = np.array(old_prey[i])

                child = self.normal_search(prey[i])
                new_v, new_dis = self.calculator(child, st, dst)
                if new_v == 0:
                    if new_dis < old_s[i]:
                        old_s[i] = new_dis
                        prey[i] = np.array(child)
                    if new_dis < min_s:
                        min_s = new_dis
                        best_prey = np.array(child)
                        print(min_s)

                ga_child = self.evolution(child, prey[i], st, dst)
                new_v, new_dis = self.calculator(ga_child, st, dst)
                if new_v == 0:
                    if new_dis < old_s[i]:
                        old_s[i] = new_dis
                        prey[i] = np.array(ga_child)
                    if new_dis < min_s:
                        min_s = new_dis
                        best_prey = np.array(ga_child)
                        print(min_s)

            old_prey = np.array(prey)

            r_x = rd.random()
            fad = 0.2 * (1 - r_x) + r_x
            Fad = np.array([[fad, fad] for _ in range(d)])
            ux = np.random.randint(0, 1, d)
            uy = np.random.randint(0, 1, d)
            U = np.array([[ux[j], uy[j]] for j in range(d)])

            for i in range(n_child):
                rx = np.random.uniform(0, 1, d)
                ry = np.random.uniform(0, 1, d)
                R = np.array([[rx[j], ry[j]] for j in range(d)])
                pre_prey = prey[i]
                if r_x < 0.2:
                    pre_prey = pre_prey + CF * (X_min + R * (X_max - X_min)) * U
                else:
                    pre_prey = pre_prey + Fad * (prey[rd.randint(0, n_child - 1)] - prey[rd.randint(0, n_child - 1)])
                for j in range(d):
                    pre_prey[j] = self.check(pre_prey[j], prey[i][j])
                prey[i] = pre_prey
            print('(no start) loop %d: %.4f'%(index+1,min_s), end='\r')
        print()
        final_sol = self.shorten(list(map(list,best_prey)),st,dst)
        final_sol.insert(0, st)
        v_sol, dis_sol = self.calculator(final_sol, st, dst)
        print('best with start: ', dis_sol, final_sol)
        return dis_sol, final_sol, a_prey

import matplotlib
import matplotlib.pyplot as plt
def draw_line(points:list[list],plt:matplotlib.pyplot):
    xl = []
    yl = []
    for point in points:
        xl.append(point[0]+0.5)
        yl.append(point[1]+0.5)
    plt.plot(xl,yl,'-')

def display(temp_data,map_size,final_sol,plt):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlim(0, map_size)
    ax.set_ylim(0, map_size)
    plt.xticks([*range(map_size)])
    plt.yticks([*range(map_size)])
    draw_line(final_sol,plt)
    plt.imshow(temp_data.transpose(),origin='lower',extent = (0,map_size,0,map_size))
    plt.grid()
    ax.invert_yaxis()

    plt.show(block=False)
    plt.pause(10)

