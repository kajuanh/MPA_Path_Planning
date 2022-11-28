import math
import numpy as np
import random as rd
from GripMap import GridMap
from mytempcode_MPA.levy import levy as levys

def display_data(data: np):
    for i in data:
        display_line_array(i)


def display_line_array(line: np):
    s = ''
    for j in line:
        s += '[%-2s %-2s]  ' % (j[0], j[1])
    print(s)


class MPA(GridMap):
    d_min = 2
    
    def shorten(self, origin_sol: list[list[int, int]], start: list[int, int], end: list[int, int]):
        '''shorten the path'''
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
        '''shorten best_prey '''
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

    def random_init(self, start: list[int, int], end: list[int, int]):
        '''random init individuals (path)'''
        origin_sol = [start]
        temp_map = self.data.copy()
        limit = 10
        lim = 0
        c = math.ceil(self.distance_s(start,end)/5)
        if temp_map[start[0], start[1]] == 1 or temp_map[end[0], end[1]] == 1:
            raise Exception('start or end is not node empty')
            return
        else:
            temp_map[start[0], start[1]] = 3
        while (self.check_collision(origin_sol[-1], end)):
            rd_values = self.random_space(c, origin_sol[-1], temp_map)
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

            if len(origin_sol) == 0:
                origin_sol = [start]
                temp_map = self.data.copy()
                temp_map[start[0], start[1]] = 3
                lim += 1
            if lim > limit:
                raise Exception('start or end is stuck')
        origin_sol = self.shorten(origin_sol, start, end)
        return origin_sol

    def full_data(self, data: list[list[list]], start: list[int], end: list[int]):
        '''full data by end to create numpy'''
        max_len = 0

        for line in data:
            if line[0] == start:
                line.pop(0)
            if line[-1] == end:
                line.pop(-1)

            if (len_line := len(line)) > max_len:
                max_len = len_line

        for line in data:
            line.extend([end for _ in range(max_len-len(line))])
        return data, max_len
    
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

    def calculator(self, f_sol:list[list[int]], f_st:list[int], f_dst:list[int]):
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

    def init_population(self, n_child: int, start: list[int], end: list[int]):
        '''init population with n_child individuals'''
        population = []
        for _ in range(n_child):
            population.append(self.random_init(start, end))
        #     print('random_init: %d/%d' % (_+1, n_child), end='\r')
        # print()
        return population

    def way(self, start: list[int], end: list[int]):
        n_child = self.map_size*2
        CONST_INF = 9999.0
        if not self.check_collision(start, end):
            # print('no have collision (start-end) ')
            return self.distance_s(start, end), [start, end]

        population = self.init_population(n_child, start, end)
        data, dim = self.full_data(population, start, end)
        Prey = np.array(data, dtype=np.int64)
        SearchAgents_no = Prey.shape[0]
        lb = 0
        ub = self.map_size-1
        Max_iter = 50

        Xmin = np.ones((SearchAgents_no, dim), dtype=int)*lb
        Xmax = np.ones((SearchAgents_no, dim), dtype=int)*ub

        Top_predator_pos = np.zeros((dim, 2), dtype=int)
        Top_predator_fit = CONST_INF

        Convergence_curve = np.zeros((1, Max_iter))

        stepsize_x = np.zeros((SearchAgents_no, dim))
        stepsize_y = np.zeros((SearchAgents_no, dim))

        fitness = np.full((SearchAgents_no, 1), CONST_INF)

        FADs = 0.2
        P = 0.5
        for Iter in range(50):
            for i in range(SearchAgents_no):

                Flag4ubx = (Prey[i, :,0] > ub).astype(int)
                Flag4lbx = (Prey[i, :,0] < lb).astype(int)
                Flag4uby = (Prey[i, :,1] > ub).astype(int)
                Flag4lby = (Prey[i, :,1] < lb).astype(int)

                Prey[i, :, 0] = (Prey[i, :, 0]*(np.logical_not(Flag4ubx +Flag4lbx).astype(int))+ub*Flag4ubx+lb*Flag4lbx)
                Prey[i, :, 1] = (Prey[i, :, 1]*(np.logical_not(Flag4uby +Flag4lby).astype(int))+ub*Flag4uby+lb*Flag4lby)
                v, dis_prey = self.calculator(Prey[i, :].tolist(), start, end)

                if v == 0:
                    fitness[i, 0] = dis_prey
                elif v == 1:
                    fitness[i, 0] = CONST_INF
                else:
                    print(v, dis_prey, Prey[i, :])
                if (fitness[i, 0] < Top_predator_fit):
                    Top_predator_fit = fitness[i, 0]

                    Top_predator_pos = Prey[i].copy()
                    # print(fitness[i, 0])
            # print(Top_predator_pos.T)
            if Iter == 0:
                fit_old = fitness.copy()
                Prey_old = Prey.copy()

            Inx = np.zeros((fitness.shape[0],1))
            for i in range(fitness.shape[0]):
                if(fit_old[i] < fitness[i]):
                    Inx[i] = 0
                else:
                    Inx[i] = 1

            Indx = np.full((Inx.shape[0], dim), Inx, dtype=int)
            Prey[:, :, 0] = Indx*Prey_old[:, :, 0] + np.logical_not(Indx) * Prey[:, :, 0]
            Prey[:, :, 1] = Indx*Prey_old[:, :, 1] + np.logical_not(Indx) * Prey[:, :, 1]
            fitness =Inx*fit_old + np.logical_not(Inx).astype(int) * fitness

            fit_old = fitness.copy()
            Prey_old = Prey.copy()
                
            Elite = np.full((SearchAgents_no, *Top_predator_pos.shape), Top_predator_pos, dtype=np.int64)  # %(Eq. 10)

            CF = (1-Iter/Max_iter)**(2*Iter/Max_iter)

            # RLX = np.array(levy.rvs(0, 1, (SearchAgents_no, dim)))
            # RLY = np.array(levy.rvs(0, 1, (SearchAgents_no, dim)))
            RLX = 0.05*levys(SearchAgents_no, dim, 1.5)
            RLY = 0.05*levys(SearchAgents_no, dim, 1.5)
            # RBX = np.random.normal(0,1,(SearchAgents_no, dim))
            # RBY = np.random.normal(0,1,(SearchAgents_no, dim))
            RBX = np.random.randn(SearchAgents_no, dim)
            RBY = np.random.randn(SearchAgents_no, dim)

            for i in range(SearchAgents_no):
                for j in range(dim):
                    R = rd.uniform(0, 1)
                    #  %------------------ Phase 1 (Eq.12) -------------------
                    if Iter < Max_iter/3:
                        stepsize_x[i, j] = RBX[i, j] * (Elite[i, j, 0]-RBX[i, j]*Prey[i, j, 0])
                        stepsize_y[i, j] = RBY[i, j] * (Elite[i, j, 1]-RBY[i, j]*Prey[i, j, 1])

                        Prey[i, j, 0] = Prey[i, j, 0] + ((P*R*stepsize_x[i, j]) % self.map_size) if stepsize_x[i, j] > 0 else - ((P*R*abs(stepsize_x[i, j])) % self.map_size)
                        Prey[i, j, 1] = Prey[i, j, 1] + ((P*R*stepsize_y[i, j]) % self.map_size) if stepsize_y[i, j] > 0 else - ((P*R*abs(stepsize_y[i, j])) % self.map_size)

                    # %--------------- Phase 2 (Eqs. 13 & 14)----------------
                    elif (Iter > Max_iter/3) and (Iter < 2*Max_iter/3):
                        if i > Prey.shape[0]/2:
                            stepsize_x[i, j] = RBX[i, j] * (Elite[i, j, 0]-RBX[i, j]*Prey[i, j, 0])
                            stepsize_y[i, j] = RBY[i, j] * (Elite[i, j, 1]-RBY[i, j]*Prey[i, j, 1])

                            Prey[i, j, 0] = Elite[i, j, 0] + ((P*CF*stepsize_x[i, j]) % self.map_size) if stepsize_x[i, j] > 0 else - (
                                (P*CF*abs(stepsize_x[i, j])) % self.map_size)

                            Prey[i, j, 1] = Elite[i, j, 1] + ((P*CF*stepsize_y[i, j]) % self.map_size) if stepsize_y[i, j] > 0 else - (
                                (P*CF*abs(stepsize_y[i, j])) % self.map_size)
                        else:
                            stepsize_x[i, j] = (RLX[i, j] * (Elite[i, j, 0]-RLX[i, j]*Prey[i, j, 0]))
                            stepsize_y[i, j] = (RLY[i, j] * (Elite[i, j, 1]-RLY[i, j]*Prey[i, j, 1]))

                            Prey[i, j, 0] = Prey[i, j, 0] + ((P*R*stepsize_x[i, j]) % self.map_size) if stepsize_x[i, j] > 0 else - (
                                (P*R*abs(stepsize_x[i, j])) % self.map_size)

                            Prey[i, j, 1] = Prey[i, j, 1] + ((P*R*stepsize_y[i, j]) % self.map_size) if stepsize_y[i, j] > 0 else - (
                                (P*R*abs(stepsize_y[i, j])) % self.map_size)

                    #  %----------------- Phase 3 (Eq. 15)-------------------
                    else:
                        stepsize_x[i, j] = (RLX[i, j] * (RLX[i, j]*Elite[i, j, 0]-Prey[i, j, 0]))
                        stepsize_y[i, j] = (RLX[i, j] * (RLX[i, j]*Elite[i, j, 1]-Prey[i, j, 1]))

                        Prey[i, j, 0] = Elite[i, j, 0] + ((P*CF*stepsize_x[i, j]) % self.map_size) if stepsize_x[i, j] > 0 else - (
                            (P*CF*abs(stepsize_x[i, j])) % self.map_size)

                        Prey[i, j, 1] = Elite[i, j, 1] + ((P*CF*stepsize_y[i, j]) % self.map_size) if stepsize_y[i, j] > 0 else - (
                            (P*CF*abs(stepsize_y[i, j])) % self.map_size)

                child = self.normal_search(Prey[i])
                new_v, new_dis = self.calculator(child, start, end)
                if new_v == 0 and new_dis < fitness[i]:
                    Prey[i] = child
                
                ga_child = self.evolution(child, Prey[i], start, end)
                new_v, new_dis = self.calculator(ga_child, start, end) 
                if new_v == 0 and new_dis < fitness[i]:
                    Prey[i] = ga_child


            # display_data(Prey)
            for i in range(SearchAgents_no):
                # dis_prey = np.inf
                Flag4ubx = (Prey[i, :,0] > ub).astype(int)
                Flag4lbx = (Prey[i, :,0] < lb).astype(int)
                Flag4uby = (Prey[i, :,1] > ub).astype(int)
                Flag4lby = (Prey[i, :,1] < lb).astype(int)

                Prey[i, :, 0] = (Prey[i, :, 0]*(np.logical_not(Flag4ubx + Flag4lbx).astype(int))+ub*Flag4ubx+lb*Flag4lbx)
                Prey[i, :, 1] = (Prey[i, :, 1]*(np.logical_not(Flag4uby + Flag4lby).astype(int))+ub*Flag4uby+lb*Flag4lby)
                v,dis_prey = self.calculator(Prey[i, :].tolist(), start, end) 

                if v == 0:
                    fitness[i, 0] = dis_prey
                    
                elif v==1:
                    fitness[i, 0] = CONST_INF
                else:
                    print(v, dis_prey, Prey[i, :])

                if (fitness[i, 0] < Top_predator_fit):
                    Top_predator_fit = fitness[i, 0].copy()
                    Top_predator_pos = Prey[i].copy()
                    # print(fitness[i, 0])

            # print('Top_predator_pos:')
            # display_line_array(Top_predator_pos)
            # print('fitness',fitness)
            if Iter == 0:
                fit_old = fitness.copy()
                Prey_old = Prey.copy()

            Inx = np.zeros((fitness.shape[0],1),dtype=int)
            for i in range(fitness.shape[0]):
                if(fit_old[i] < fitness[i]):
                    Inx[i] = 0
                else:
                    Inx[i] = 1

            Indx = np.full((Inx.shape[0], dim), Inx, dtype=int)

            Prey[:, :, 0] = Indx*Prey_old[:, :, 0] + np.logical_not(Indx) * Prey[:, :, 0]
            Prey[:, :, 1] = Indx*Prey_old[:, :, 1] + np.logical_not(Indx) * Prey[:, :, 1]

            fitness =Inx*fit_old + np.logical_not(Inx) * fitness
            Prey_old = Prey.copy()
            #%---------- Eddy formation and FADs� effect (Eq 16) -----------
            if rd.uniform(0, 1) < FADs:
                U = np.random.rand(SearchAgents_no, dim) < FADs
                stepsize_x = CF * ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
                stepsize_y = CF * ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
                Prey[:, :, 0] = Prey[:, :, 0] + stepsize_x 
                Prey[:, :, 1] = Prey[:, :, 1] + stepsize_y 
            else:
                r = rd.uniform(0, 1)
                Rs = Prey.shape[0]
                stepsize_x = (FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :, 0]-Prey[np.random.permutation(Rs), :, 0])
                stepsize_y = (FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :, 1]-Prey[np.random.permutation(Rs), :, 1])
                Prey[:, :, 0] = Prey[:, :, 0] + stepsize_x
                Prey[:, :, 1] = Prey[:, :, 1] + stepsize_y

            Convergence_curve[:, Iter] = Top_predator_fit
        
        final_sol = self.best_shorten(Top_predator_pos.tolist(), start, end)
        v_sol, dis_sol = self.calculator(final_sol, start, end)
        # print(dis_sol, final_sol)
        return dis_sol, final_sol

