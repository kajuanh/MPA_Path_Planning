import math
from scipy.stats import levy
import numpy as np
import random as rd


class MPA:
    environment = []
    list_dst = []
    n = 0
    x_min = 1
    x_max = 5
    d_min = 2
    origin = 0.4

    def __init__(self, filename):
        fp = open(filename, "r")
        f_n = int(fp.readline())
        f_l = int(fp.readline())
        l_dst = []
        f_map = []
        for f_i in range(f_l):
            s = fp.readline()
            s_l = s[:-1].split(" ")
            l_dst.append([int(s_l[0]) + 0.5, int(s_l[1]) + 0.5])
        for f_i in range(f_n):
            s = fp.readline()
            s_l = s[0:-2].split(" ")
            f_map.append(list(map(int, s_l)))
        for f_i in l_dst:
            f_map[int(f_i[0])][int(f_i[1])] = 0
        self.n = f_n
        self.list_dst = list(l_dst)
        self.environment = []
        for f_i in f_map:
            self.environment.append(list(f_i))
        fp.close()

    def check_collision(self, f_X1, f_X2):
        x1 = int(f_X1[0])
        y1 = int(f_X1[1])
        x2 = int(f_X2[0])
        y2 = int(f_X2[1])

        if self.environment[x1][y1] == 1 or self.environment[x2][y2] == 1:
            return True
        else:
            if x1 > x2:
                x1 = int(f_X2[0])
                x2 = int(f_X1[0])
            if y1 > y2:
                y1 = int(f_X2[1])
                y2 = int(f_X1[1])
            list_obstacle = []
            x1 -= 1
            if x1 < 0:
                x1 = 0
            x2 += 1
            if x2 >= self.n:
                x2 = self.n - 1
            y1 -= 1
            if y1 < 0:
                y1 = 0
            y2 += 1
            if y2 >= self.n:
                y2 = self.n - 1
            for f_i in range(x1, x2 + 1):
                for f_j in range(y1, y2 + 1):
                    if self.environment[f_i][f_j] == 1:
                        list_obstacle.append([f_i, f_j])

            if len(list_obstacle) == 0:
                return False
            if f_X1[0] == f_X2[0]:
                for obs in list_obstacle:
                    x_left = obs[0] - self.origin
                    x_right = obs[0] + 1 + self.origin
                    y_top = obs[1] - self.origin
                    y_bot = obs[1] + 1 + self.origin
                    y_low = f_X1[1]
                    y_high = f_X2[1]
                    if f_X1[1] > f_X2[1]:
                        y_low = f_X2[1]
                        y_high = f_X1[1]
                    if x_left <= f_X1[0] <= x_right and (y_low <= y_top <= y_high or y_low <= y_bot <= y_high):
                        return True
            elif f_X1[1] == f_X2[1]:
                for obs in list_obstacle:
                    x_left = obs[0] - self.origin
                    x_right = obs[0] + 1 + self.origin
                    y_top = obs[1] - self.origin
                    y_bot = obs[1] + 1 + self.origin
                    x_low = f_X1[0]
                    x_high = f_X2[0]
                    if f_X1[0] > f_X2[0]:
                        x_low = f_X2[0]
                        x_high = f_X1[0]
                    if y_top <= f_X1[1] <= y_bot and (x_low <= x_left <= x_high or x_low <= x_right <= x_high):
                        return True
            else:
                f_a = (f_X2[1] - f_X1[1]) / (f_X2[0] - f_X1[0])
                f_b = f_X1[1] - f_a * f_X1[0]
                for obs in list_obstacle:
                    x_left = obs[0] - self.origin
                    x_right = obs[0] + 1 + self.origin
                    y_top = obs[1] - self.origin
                    y_bot = obs[1] + 1 + self.origin

                    x_top = (y_top - f_b) / f_a
                    x_bot = (y_bot - f_b) / f_a
                    y_left = f_a * x_left + f_b
                    y_right = f_a * x_right + f_b

                    x_low = f_X1[0]
                    x_high = f_X2[0]
                    if f_X1[0] > f_X2[0]:
                        x_low = f_X2[0]
                        x_high = f_X1[0]

                    y_low = f_X1[1]
                    y_high = f_X2[1]
                    if f_X1[1] > f_X2[1]:
                        y_low = f_X2[1]
                        y_high = f_X1[1]

                    if (x_left <= x_top <= x_right and (y_low <= y_top <= y_high or y_low <= y_bot <= y_high)) or \
                            (x_left <= x_bot <= x_right and (y_low <= y_top <= y_high or y_low <= y_bot <= y_high)) or \
                            (y_top <= y_left <= y_bot and (x_low <= x_left <= x_high or y_low <= x_right <= y_high)) or \
                            (y_top <= y_right <= y_bot and (x_low <= x_left <= x_high or y_low <= x_right <= y_high)):
                        return True
            return False

    @staticmethod
    def distance(f_X1, f_X2):
        return math.sqrt((f_X1[0] - f_X2[0]) * (f_X1[0] - f_X2[0]) + (f_X1[1] - f_X2[1]) * (f_X1[1] - f_X2[1]))

    def calculator(self, f_sol, f_st, f_dst):
        s = 0
        is_dst = False
        pre_x = f_st
        for f_x in f_sol:
            if self.check_collision(pre_x, f_x):
                return 1, s
            s += self.distance(pre_x, f_x)
            if not self.check_collision(f_x, f_dst):
                s += self.distance(f_x, f_dst)
                is_dst = True
                break
            pre_x = f_x
        if not is_dst:
            if self.check_collision(f_sol[-1], f_dst):
                return 1, s
            s += self.distance(f_sol[-1], f_dst)
        return 0, s

    def check(self, f_X1, f_X2):
        f_x = f_X1[0]
        f_y = f_X1[1]
        pre_x = f_X2[0]
        pre_y = f_X2[1]

        if f_x < self.origin:
            f_a = (f_y - pre_y) / (f_x - pre_x)
            f_b = f_y - f_a * f_x
            f_x = self.origin
            f_y = f_a * f_x + f_b
        elif f_x >= self.n:
            f_a = (f_y - pre_y) / (f_x - pre_x)
            f_b = f_y - f_a * f_x
            f_x = self.n - self.origin
            f_y = f_a * f_x + f_b
        if f_y < self.origin:
            f_a = (f_x - pre_x) / (f_y - pre_y)
            f_b = f_x - f_a * f_y
            f_y = self.origin
            f_x = f_a * f_y + f_b
        elif f_y >= self.n:
            f_a = (f_x - pre_x) / (f_y - pre_y)
            f_b = f_x - f_a * f_y
            f_y = self.n - self.origin
            f_x = f_a * f_y + f_b
        if f_x >= self.n - self.origin:
            f_x = self.n - self.origin
        if f_y >= self.n - self.origin:
            f_y = self.n - self.origin
        if f_x <= self.origin:
            f_x = self.origin
        if f_y <= self.origin:
            f_y = self.origin

        return round(f_x, 2), round(f_y, 2)

    def normal_search(self, f_sol):
        f_ns_sol = []
        for f_i in f_sol:
            f_x = self.d_min * rd.random() * (rd.randint(0, 2) - 1)
            f_y = self.d_min * rd.random() * (rd.randint(0, 2) - 1)
            f_ns_sol.append(self.check([f_i[0] + f_x, f_i[1] + f_y], f_i))
        return f_ns_sol

    def evolution(self, f_father, f_mother, f_st, f_dst):
        f_child = [f_st]
        for f_i in range(len(f_father)):
            if self.check_collision(f_child[-1], f_father[f_i]):
                f_child.append(f_mother[f_i])
            elif self.check_collision(f_child[-1], f_mother[f_i]):
                f_child.append(f_father[f_i])
            else:
                if self.distance(f_child[-1], f_father[f_i]) < self.distance(f_child[-1], f_mother[f_i]):
                    f_child.append(f_father[f_i])
                else:
                    f_child.append(f_mother[f_i])
            if not self.check_collision(f_child[-1], f_dst):
                break
        f_child.remove(f_st)
        for f_i in range(len(f_father) - len(f_child)):
            f_child.append(f_dst)
        return f_child

    def way(self, st, dst):
        if not self.check_collision(st, dst):
            return self.distance(st, dst), [list(st), list(dst)]
        n_child = self.n
        min_s = math.inf
        prey = list([])
        best_prey = list([])
        old_s = list([])
        max_d = 0
        for index in range(n_child):
            origin_sol = list([list(st)])
            mp = []
            for env in self.environment:
                mp.append(list(env))
            while self.check_collision(origin_sol[-1], dst):
                limit = 0
                while True:
                    limit += 1
                    x0 = (self.x_min + rd.random() * (self.x_max - self.x_min)) * (rd.randint(0, 2) - 1)
                    y0 = (self.x_min + rd.random() * (self.x_max - self.x_min)) * (rd.randint(0, 2) - 1)
                    x, y = self.check([origin_sol[-1][0] + x0, origin_sol[-1][1] + y0], origin_sol[-1])
                    if not self.check_collision([int(x) + 0.5, int(y) + 0.5], origin_sol[-1]):
                        x = int(x) + 0.5
                        y = int(y) + 0.5
                    if mp[int(x)][int(y)] == 0 and not self.check_collision([x, y], origin_sol[-1]):
                        origin_sol.append([x, y])
                        mp[int(x)][int(y)] = 3
                        break
                    if limit > self.x_max * self.x_max * 40:
                        origin_sol.pop()
                        break
                if len(origin_sol) == 0:
                    origin_sol = list([list(st)])
                    mp.clear()
                    for env in self.environment:
                        mp.append(list(env))
            reduce_sol = list([list(st)])
            while self.check_collision(reduce_sol[-1], dst):
                i = len(origin_sol) - 1
                while self.check_collision(reduce_sol[-1], origin_sol[i]):
                    i -= 1
                reduce_sol.append(origin_sol[i])
            pre_sol = st
            reduce_sol.pop(0)
            reduce_sol.append(list(dst))
            iPrey = list([])
            for sol in reduce_sol:
                loop = int(self.distance(pre_sol, sol) / self.d_min)
                for i in range(1, loop):
                    x = round(pre_sol[0] + (sol[0] - pre_sol[0]) * i / loop, 2)
                    y = round(pre_sol[1] + (sol[1] - pre_sol[1]) * i / loop, 2)
                    if self.check_collision(pre_sol, [x, y]) or self.check_collision([x, y], sol):
                        continue
                    iPrey.append([x, y])
                iPrey.append(sol)
                pre_sol = sol
            iPrey.pop()
            sol_d = len(iPrey)
            if max_d < sol_d:
                max_d = sol_d
            v, dis_prey = self.calculator(iPrey, st, dst)
            old_s.append(dis_prey)
            prey.append(iPrey)
            if min_s > dis_prey:
                min_s = dis_prey
                best_prey = list(iPrey)
        d = max_d
        for i in range(n_child):
            for j in range(d - len(prey[i])):
                prey[i].append(list(dst))
        for i in range(d - len(best_prey)):
            best_prey.append(list(dst))

        X_min = np.array([[self.x_min, self.x_min] for _ in range(d)])
        X_max = np.array([[self.x_max, self.x_max] for _ in range(d)])
        prey = np.array(prey)
        old_prey = np.array(prey)
        loop = 1
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
                    if i < self.n / 2:
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

        best_reduce = []
        for i in best_prey:
            best_reduce.append(i)
            if not self.check_collision(i, dst):
                break

        final_sol = list([list(st)])
        while self.check_collision(final_sol[-1], dst):
            i = len(best_reduce) - 1
            while self.check_collision(final_sol[-1], best_reduce[i]):
                i -= 1
            final_sol.append(list(best_reduce[i]))
        final_sol.append(list(dst))
        v_sol, dis_sol = self.calculator(final_sol, st, dst)
        return dis_sol, final_sol
