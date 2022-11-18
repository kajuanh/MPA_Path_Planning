import sys
import os
import datetime
import shutil
from GripMap import ThreadWithReturnValue
from threading import Thread
from MPA import MPA#, read_file
import numpy as np
from VisualizeResult_copy import read_file, split_map, DisplayMatplotlib, plt
import curses
from new_Ga import GA_TSP

# env, m_size = read_file('Test/map15_4 _new.txt')
# print(m_size)
# print(env.T)
# goals, start, end, o_env = split_map(env)
# print(goals)
# print(start)
# print(end)
# print(o_env.T)
def solve(filepath, label):
    now = datetime.datetime.now()
    start_time = datetime.datetime.timestamp(now)

    env, m_size = read_file(filepath)
    goals, start, end, o_env, is_have_se = split_map(env)
    num_g = num_t = len(goals)
    targets = goals.copy()
    
    if is_have_se :
        se = '_se'
        num_t+=2
        targets.insert(0,start)
        targets.append(end)
        targets.sort()
        index_s = targets.index(start)
        index_e = targets.index(end)
    else:
        se = ''
        index_s = index_e = None

    k = (num_t)*(num_t-1)
    map_tsp = np.full((num_t,num_t),0.0, dtype=float)
    map_way = np.empty((num_t,num_t), dtype=object)

    file_name_sol = "Solutions/map" + (str(m_size)) + "_" + str(num_g) + se + "_sol.txt"
    os.makedirs(os.path.dirname(file_name_sol), exist_ok=True)
    shutil.copyfile(filepath, file_name_sol)
    
    mpa_obj = MPA(m_size, o_env)
    threads_ways = []
    dict_log = {}
    num_way = 0
    for i_s in range(num_t):
        for i_d in range(i_s + 1, num_t):
        
            if i_s == index_s and i_d == index_e:
                continue

            thread = ThreadWithReturnValue(
                name='Way%s%s' % (str(targets[i_s]), str(targets[i_d])),
                target=mpa_obj.way,
                args=(targets[i_s], targets[i_d]))
            threads_ways.append(thread)
            thread.start()
            dict_log[thread.name] = '..Running'
            num_way += 1

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    is_running = True
    while is_running:
        is_running = False
        for thread in threads_ways:
            if thread.is_alive():
                dict_log[thread.name] = '..Running'
                is_running = True
            else:
                dict_log[thread.name] = '.......OK'
        t = 1
        for key, item in dict_log.items():
            if t < 15:
                stdscr.addstr(t, 0, "{2} {0} : {1}".format(key, item, t))
            elif 15 < t < 30:
                stdscr.addstr(t-15, 40, "{2} {0} : {1}".format(key, item, t))
            elif t>30:
                stdscr.addstr(t-30, 80, "{2} {0} : {1}".format(key, item, t))
            # else:
            #     stdscr.addstr(t-30, 40, "{2} {0} : {1}".format(key, item, t))
            t += 1
        stdscr.refresh()
    curses.echo()
    curses.nocbreak()
    curses.endwin()

    t=0
    for i_s in range(num_t):
        for i_d in range(i_s + 1, num_t):
            if i_s == index_s and i_d == index_e:
                map_tsp[index_s][index_e] = map_tsp[index_e][index_s] = 9999
                continue
            dis_sol, final_sol = threads_ways[t].join()
            map_tsp[i_s][i_d] = dis_sol
            map_way[i_s][i_d] = final_sol
            re_way = final_sol.copy()
            re_way.reverse()
            map_tsp[i_d][i_s] = dis_sol
            map_way[i_d][i_s] = re_way
            t += 1

    # print(str(np.round(map_tsp,2).T))
    # print(str(map_way))
    ga_tsp = GA_TSP(map_tsp.tolist(), k, m_size,index_s=index_s,index_e=index_e)
    best_cost, best = ga_tsp.solve()
    # print(best_cost)
    # print(best)
    line = []
    if is_have_se:
        best.insert(0,index_s)
        best.append(index_e)
    for i in range(len(best)-1):
        line.extend(map_way[best[i]][best[i+1]])
    # print(line)
    display = DisplayMatplotlib(plt, m_size, line, o_env, start, end, goals)
    display.draw()


solve('Test/map_.txt',None)





