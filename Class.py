from GripMap import GridMap, remove_duplicate
from scipy.stats import levy
import random as rd
import numpy as np
import math
# from new_shorten import shorten

# def display_array(array):
#     for i in array:
#         print(i.tolist())
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
        # environment = GridMap(map_size,np_map,list_obstacle,list_empty)
        print('SUCCESS read file')
        return map_size,np_map,list_obstacle,list_empty, list_goal

class MPAs(GridMap):
    x_min = 1
    x_max = 5
    d_min = 2
    origin = 0.4

    def init_population(self, n_child: int, start, end):
        origin_sol = []
        for _ in range(n_child):
            origin_sol.append(self.random_init(start, end))
        #     print('random_init: %d/%d' % (_+1, n_child), end='\r')
        # print()
        return origin_sol

    def random_init(self, start: list[int, int], end: list[int, int]):
        '''random init path'''
        origin_sol = [start]
        temp_map = self.data.copy()
        limit = 1000
        lim = 0
        if temp_map[start[0], start[1]] == 1 or temp_map[end[0], end[1]] == 1:
            # print('start or end is not node empty')
            return
        else:
            temp_map[start[0], start[1]] = 3
        while (self.check_collision(origin_sol[-1], end)):
            rd_values = self.random_space(2, origin_sol[-1], temp_map)
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

            if len(origin_sol) == 0 or lim > limit:
                origin_sol = [start]
                temp_map = self.data.copy()
                temp_map[start[0], start[1]] = 3
                lim = 0
            lim += 1
        origin_sol = self.shorten_t(origin_sol,start,end)
        return origin_sol


    def check(self, f_X1, f_X2):
        f_x = f_X1[0]
        f_y = f_X1[1]
        pre_x = f_X2[0]
        pre_y = f_X2[1]
        a_y = (f_y - pre_y) / (f_x - pre_x) if (f_x - pre_x) != 0 else 0
        a_x = (f_x - pre_x) / (f_y - pre_y) if (f_y - pre_y) != 0 else 0

        if f_x < self.origin:
            f_a = a_y
            f_b = f_y - f_a * f_x
            f_x = self.origin
            f_y = f_a * f_x + f_b
        elif f_x >= self.map_size:
            f_a = a_y
            f_b = f_y - f_a * f_x
            f_x = self.map_size - self.origin
            f_y = f_a * f_x + f_b
        elif f_y < self.origin:
            f_a = a_x
            f_b = f_x - f_a * f_y
            f_y = self.origin
            f_x = f_a * f_y + f_b
        elif f_y >= self.map_size:
            f_a = a_x
            f_b = f_x - f_a * f_y
            f_y = self.map_size - self.origin
            f_x = f_a * f_y + f_b
        if f_x >= self.map_size - self.origin:
            f_x = self.map_size - self.origin
        if f_y >= self.map_size - self.origin:
            f_y = self.map_size - self.origin
        if f_x <= self.origin:
            f_x = self.origin
        if f_y <= self.origin:
            f_y = self.origin

        return round(f_x, 2), round(f_y, 2)

    def normal_search(self, f_sol):
        f_ns_sol = []
        for f_i in f_sol:
            f_x = rd.randint(-1, 1)
            f_y = rd.randint(-1, 1)
            f_ns_sol.append([f_i[0] + f_x, f_i[1] + f_y])
        return f_ns_sol        

    def evolution(self, f_father, f_mother, f_st, f_dst):
        f_child = [f_st]
        for f_i in range(len(f_father)):
            if self.check_collision(f_child[-1], f_father[f_i]):
                f_child.append(f_mother[f_i])
            elif self.check_collision(f_child[-1], f_mother[f_i]):
                f_child.append(f_father[f_i])
            else:
                if self.distance_s(f_child[-1], f_father[f_i]) < self.distance_s(f_child[-1], f_mother[f_i]):
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
            s += self.distance_s(pre_x, f_x)
            if not self.check_collision(f_x, f_dst):
                s += self.distance_s(f_x, f_dst)
                is_dst = True
                break
            pre_x = f_x
        if not is_dst:
            if self.check_collision(f_sol[-1], f_dst):
                return 1, s
            s += self.distance_s(f_sol[-1], f_dst)
        return 0, s

    def shorten(self, path: list[list[int,int]],start: list[int,int],end: list[int,int]):
        '''function to shorten the path'''
        remove_st_en = []
        for i,p in enumerate (path,0):
            if p == start or p == end:
                remove_st_en.append(i)
        remove_st_en.reverse()
        for i in remove_st_en:
            path.pop(i)
        path_trip = path
        # path_trip = self.remove_no_connection(path_trip,end)
        path_trip = remove_duplicate(path_trip)
        len_list = len(path_trip)
        reduce_sol = [end]
        t=0
        index_l = len_list-1
        while t<100 and self.check_collision(reduce_sol[-1],start):
            temp = None
            for i in range(index_l,-1,-1):
                    if not self.check_collision(reduce_sol[-1], path_trip[i]) :
                        temp = path_trip[i]
                        tem_id = i-1
            if temp is not None:
                reduce_sol.append(temp)
                index_l = tem_id
            # print(reduce_sol)
            t+=1
        if t>=100:
            # print(start,end)
            # print(path_trip)
            # print(path)
            raise ('loi')
        reduce_sol.pop(0)
        reduce_sol.reverse()
        return reduce_sol
        # return path_trip


    def shorten_t(self, origin_sol: list[list[int, int]], start: list[int, int], end: list[int, int]):
        reduce_sol = [start]
        while self.check_collision(reduce_sol[-1], end):
            i = len(origin_sol) - 1
            while self.check_collision(reduce_sol[-1], origin_sol[i]):
                i -= 1
            reduce_sol.append(origin_sol[i])
        pre_sol = start
        reduce_sol.pop(0)
        reduce_sol.append(end)
        iPrey = []
        for sol in reduce_sol:
            loop = int(self.distance_s(pre_sol, sol) / self.d_min)
            for i in range(1, loop):
                x = round(pre_sol[0] + (sol[0] - pre_sol[0]) * i / loop)
                y = round(pre_sol[1] + (sol[1] - pre_sol[1]) * i / loop)
                if self.check_collision(pre_sol, [x, y]) or self.check_collision([x, y], sol):
                    continue
                iPrey.append([x, y])
            iPrey.append(sol)
            pre_sol = sol
        iPrey.pop()
        return iPrey

    def best_shorten(self, best_prey: list[list[int, int]], start: list[int, int], end: list[int, int]):
        start = list(start)
        end = list(end)
        best_reduce = []
        for i in best_prey:
            best_reduce.append(list(i))
            if not self.check_collision(i, end):
                break

        final_sol = [start]
        while self.check_collision(final_sol[-1], end):
            i = len(best_reduce) - 1
            while self.check_collision(final_sol[-1], best_reduce[i]):
                i -= 1
            final_sol.append(best_reduce[i])
        final_sol.append(end)
        return final_sol

    def way(self, st, dst):
        n_child = self.map_size*3
        min_s = math.inf
        prey = []
        best_prey = []
        old_s = []
        max_d = 0

        if not self.check_collision(st, dst):
            # print('no have collision (start-end) ')
            return self.distance_s(st, dst), [st, dst]

        origin_sol = self.init_population(n_child, st, dst)
        print(len(origin_sol))
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
        prey = np.array(prey,dtype=np.int64)
        old_prey = np.array(prey,dtype=np.int64)
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
                        # print(min_s)
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
                        # print(min_s)
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
                        # print(min_s)

                ga_child = self.evolution(child, prey[i], st, dst)
                new_v, new_dis = self.calculator(ga_child, st, dst)
                if new_v == 0:
                    if new_dis < old_s[i]:
                        old_s[i] = new_dis
                        prey[i] = np.array(ga_child)
                    if new_dis < min_s:
                        min_s = new_dis
                        best_prey = np.array(ga_child)
                        # print(min_s)

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
        #     print('(no start) loop %d: %.4f'%(index+1,min_s), end='\r')
        # print()
        final_sol = self.best_shorten(best_prey,st,dst)
        v_sol, dis_sol = self.calculator(final_sol, st, dst)
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

