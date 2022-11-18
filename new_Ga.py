import sys, os, math
import random as rd
import numpy as np
import time
from datetime import datetime
'''from Class import MPAs,display, read_file

# mutation_rate = 0.3
# map_tsp = [
#     [0, 27.485281374238575, 29.128990204491963, 17.064495102245978],
#     [27.485281374238575, 0, 10.828427124746192, 10.242640687119286],
#     [29.128990204491963, 10.828427124746192, 0, 12.064495102245981],
#     [17.064495102245978, 10.242640687119286, 12.064495102245981, 0]
# ]
# n_population = 10
# n_cities = len(map_tsp)
# print(n_cities)
def genesis(city_list, n_population, n_cities):

    # population_set = []
    # for i in range(n_population):
    #     #Randomly generating a new solution
    #     sol_i = city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
    #     population_set.append(sol_i)
    # # print(population_set)
    # return np.array(population_set)
    population_set = []
    population_set.append(city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)])
    population_set_np = np.unique(population_set,axis=0)
    while population_set_np.shape[0]< n_population:

        population_set.append(city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)])
        population_set_np = np.unique(population_set,axis=0)
    return population_set_np
# names_list =np.array([0,1,2,3])
# population_set = genesis(names_list, n_population)
# print(population_set)
def fitness_eval(city_list, map_tsp, n_cities):
    total = 0
    for i in range(n_cities-1):
        a = city_list[i]
        b = city_list[i+1]
        total += map_tsp[a][b]
    return total
def get_all_fitnes(population_set, map_tsp, n_population,n_cities):
    fitnes_list = np.zeros(n_population)
    #Looping over all solutions computing the fitness for each solution
    # print(n_population)
    for i in range(n_population):
        # print(population_set[i])
        # print(map_tsp)
        # print(fitness_eval(population_set[i], map_tsp))

        fitnes_list[i] = fitness_eval(population_set[i], map_tsp,n_cities)

    return fitnes_list

# fitnes_list = get_all_fitnes(population_set,map_tsp)
# print(fitnes_list)
# print(np.random.choice(list(range(3)), 3, replace=False))
def progenitor_selection(population_set,fitnes_list):
    total_fit = fitnes_list.sum()
    prob_list = fitnes_list/total_fit
    
    #Notice there is the chance that a progenitor. mates with oneself
    progenitor_list_a = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    progenitor_list_b = np.random.choice(list(range(len(population_set))), len(population_set),p=prob_list, replace=True)
    
    progenitor_list_a = population_set[progenitor_list_a]
    progenitor_list_b = population_set[progenitor_list_b]
    
    
    return np.array([progenitor_list_a,progenitor_list_b])


# progenitor_list = progenitor_selection(population_set,fitnes_list)
# print(progenitor_list[0][2])
def mate_progenitors(prog_a, prog_b):
    offspring = prog_a[0:5]

    for city in prog_b:

        if not city in offspring:
            offspring = np.concatenate((offspring,[city]))

    return offspring
            

def mate_population(progenitor_list):
    new_population_set = []
    for i in range(progenitor_list.shape[1]):
        prog_a, prog_b = progenitor_list[0][i], progenitor_list[1][i]
        offspring = mate_progenitors(prog_a, prog_b)
        new_population_set.append(offspring)
        
    return new_population_set

# new_population_set = mate_population(progenitor_list)
# print(new_population_set[0])
def mutate_offspring(offspring,n_cities,mutation_rate):
    for q in range(int(n_cities*mutation_rate)):
        a = np.random.randint(0,n_cities)
        b = np.random.randint(0,n_cities)

        offspring[a], offspring[b] = offspring[b], offspring[a]

    return offspring
    
    
def mutate_population(new_population_set,n_cities,mutation_rate):
    mutated_pop = []
    for offspring in new_population_set:
        mutated_pop.append(mutate_offspring(offspring,n_cities,mutation_rate))
    return mutated_pop

# mutated_pop = mutate_population(new_population_set)
# print(mutated_pop[0])

best_solution = [-1,np.inf,np.array([])]

for i in range(10000):
    # if i%100==0: print(i, fitnes_list.min(), fitnes_list.mean(), datetime.now().strftime("%d/%m/%y %H:%M"))
    fitnes_list = get_all_fitnes(mutated_pop,map_tsp)
    
    #Saving the best solution
    if fitnes_list.min() < best_solution[1]:
        best_solution[0] = i
        best_solution[1] = fitnes_list.min()
        best_solution[2] = np.array(mutated_pop)[fitnes_list.min() == fitnes_list]
    
    progenitor_list = progenitor_selection(population_set,fitnes_list)
    new_population_set = mate_population(progenitor_list)
    
    mutated_pop = mutate_population(new_population_set)
print(best_solution)

import curses
import time

# def report_progress(filename, progress,stdscr):
#     # stdscr.addstr(0, 0, "Moving file: {0}".format(filename))
#     for i in range(3):
#         stdscr.addstr(i, 0, "Total progress {2}: [{1:10}] {0}%".format(progress * 10, "#" * progress,i))
#     stdscr.refresh()

# if __name__ == "__main__":
#     stdscr = curses.initscr()
#     curses.noecho()
#     curses.cbreak()

#     try:
#         for i in range(10):
#             report_progress("file_{0}.txt".format(i), i+1,stdscr)
#             time.sleep(0.5)
#     finally:
#         curses.echo()
#         curses.nocbreak()
#         curses.endwin()
'''

class GA_TSP:
    def __init__(self, map_tsp, k=20, max_gen=5, mutation_prob=0.1, index_s=None, index_e=None):
        self.mutation_prob = mutation_prob
        self.num_t = len(map_tsp)
        self.map_tsp = map_tsp
        self.max_gen = max_gen
        self.index_s = index_s
        self.index_e = index_e
        self.map_cost = {}
        self.k = k

    def compute_fitness(self, s):
        if (l := self.map_cost.get(str(s))) is not None:
            return l
        else:
            l = 0
            for i in range(self.num_t-1):
                l += self.map_tsp[s[i]][s[i+1]]
            if all(v is not None for v in [self.index_s, self.index_e]):
                l += self.map_tsp[self.index_s][s[0]]
                l += self.map_tsp[s[-1]][self.index_e]
            self.map_cost[str(s)] = l
        # l += G[s[len(s)-1]][s[0]]
        return l

    
    def selection(self, gen, k):
        gen = sorted(gen, key=lambda s: self.compute_fitness(s))
        return gen[:k]

    def crossover(self, s1, s2, index):
        m_1, f_2 = s1.copy(), s2.copy()

        c1 = f_2.copy()
        for i in range(index, len(m_1)): c1.remove(m_1[i])
        for i in range(index, len(m_1)): c1.append(m_1[i])

        c2 = m_1.copy()
        for i in range(index, len(f_2)): c2.remove(f_2[i])
        for i in range(index, len(f_2)): c2.append(f_2[i])
        return (c1, c2)

    def mutation(self, s, m, n):
        i, j = min(m, n), max(m, n)
        cs = s.copy()
        while i < j:
            cs[i], cs[j] = cs[j], cs[i]
            i += 1
            j -= 1
        return cs

    def init_population(self):
        path = list(range(self.num_t))
        if all(v is not None for v in [self.index_s,self.index_e]):
            path.pop(max(self.index_s,self.index_e))
            path.pop(min(self.index_s,self.index_e))
            self.num_t -=2
        n_p = self.k
        gen = [path]
        # value_index = {}
        # value_index[str(path)]=0
        while len(gen) < n_p:
            copy_path = path.copy()
            np.random.shuffle(copy_path)
            # try:
            #     gen.index(copy_path)
            # except Exception:
            #     gen.append(copy_path)
            if copy_path not in gen:
                gen.append(copy_path)
        return gen

    def solve(self):
        ''' return best_cost, best '''
        gen = self.init_population()
        for trial in range(self.max_gen):
            gen = self.selection(gen, self.k)
            next_gen = []
            for i in range(len(gen)):
                for j in range(i+1, len(gen)):
                    c1, c2 = self.crossover(gen[i], gen[j], np.random.randint(0, len(gen[i])))
                    next_gen.append(c1)
                    next_gen.append(c2)
                if np.random.rand() < self.mutation_prob:
                    m = np.random.randint(0, len(gen[i]))
                    while True:
                        n = np.random.randint(0, len(gen[i]))
                        if m != n:
                            break
                    c = self.mutation(gen[i], m, n)
                    next_gen.append(c)
            gen = next_gen

        best_gen = self.selection(gen, 1)[0]
        best = (round(self.compute_fitness(best_gen),3), best_gen)
        return best


# map_tsp = [[0, 12.602325267042627, 27.254661916137508, ],
#            [12.602325267042627, 0, 23.55093586115953],
#            [27.254661916137508, 23.55093586115953, 0]]
# ga_tsp = GA_TSP(map_tsp,6,5)
# print(ga_tsp.solve())
