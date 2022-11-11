import sys, os, math
sys.path.append(os.path.abspath(""))
from Class import MPAs,display, read_file
import random as rd
import numpy as np
import time
from datetime import datetime
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

    population_set = []
    for i in range(n_population):
        #Randomly generating a new solution
        sol_i = city_list[np.random.choice(list(range(n_cities)), n_cities, replace=False)]
        population_set.append(sol_i)
    return np.array(population_set)
names_list =np.array([0,1,2,3])
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
'''
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
'''
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
