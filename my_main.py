import sys
import os
import datetime
import shutil
from GripMap import ThreadWithReturnValue
from threading import Thread
from MPA import MPA
import numpy as np
from VisualizeResult_copy import *
import curses
from new_Ga import GA_TSP
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from functools import partial
import glob
solutions_dir = os.path.abspath("Solutions")
# print(solutions_dir)
from pathlib import Path
import time
# env, m_size = read_file('Test/map15_4 _new.txt')
# print(m_size)
# print(env.T)
# goals, start, end, o_env = split_map(env)
# print(goals)
# print(start)
# print(end)
# print(o_env.T)
def change_label(label: Entry,str_display: str): 
    label.configure(state=NORMAL)
    label.delete(0, 100)  # delete
    label.insert(END, str_display) # insert
    label.configure(state=DISABLED)  # enable
    label.update()

def select_file(entry_input):
    file = filedialog.askopenfilename()
    entry_input.configure(state=NORMAL)
    entry_input.delete(0, 100)  # delete
    entry_input.insert(END, str(file))  # insert
    entry_input.configure(state=DISABLED)  # enable


def select_solve(select_button, solve_button):
    select_button.configure(state=NORMAL)
    solve_button.configure(state=NORMAL)

def select_show(select_combobox):
    select_combobox.configure(state=NORMAL)
    select_combobox['values'] = tuple(os.path.join('Solutions',file.name) for file in Path(solutions_dir).rglob('*.txt'))#tuple(glob.glob('*.txt'))
    select_combobox.current(0)
    select_combobox.configure(state='readonly')

def solve(file_input: Entry, label: Entry):
    filepath = file_input.get()
    # print('->>>>>',filepath:= file_input.get())
    start_time = time.time()
    env, m_size = read_file(filepath)
    goals, o_env, start, is_have_s = split_map(env)
    num_g = num_t = len(goals)
    targets = goals.copy()
    
    if is_have_s :
        s = '_s'
        num_t+=1
        targets.insert(0,start)
        # targets.sort()
        index_s = index_s = targets.index(start)
    else:
        s = ''
        index_s = index_e = None

    k = (num_t)*(num_t-1)
    map_tsp = np.full((num_t,num_t),0.0, dtype=float)
    map_way = np.empty((num_t,num_t), dtype=object)

    file_name_sol = "Solutions/map" + (str(m_size)) + "_" + str(num_g) + s + "_sol.txt"
    os.makedirs(os.path.dirname(file_name_sol), exist_ok=True)
    shutil.copyfile(filepath, file_name_sol)
    
    change_label(label, 'Success read file')

    mpa_obj = MPA(m_size, o_env)
    threads_ways = []
    dict_log = {}
    num_way = 0
    for i_s in range(num_t):
        for i_d in range(i_s + 1, num_t):

            thread = ThreadWithReturnValue(
                name='Way%s%s' % (str(targets[i_s]), str(targets[i_d])),
                target=mpa_obj.way,
                args=(targets[i_s], targets[i_d]))
            threads_ways.append(thread)
            thread.start()
            dict_log[thread.name] = '..Running'
            num_way += 1

    change_label(label, 'Start Find Way')

    stdscr = curses.initscr()
    curses.noecho()
    curses.cbreak()

    is_running = True
    while is_running:
        is_running = False
        per = 0
        for thread in threads_ways:
            if thread.is_alive():
                dict_log[thread.name] = '..Running'
                is_running = True
                
            else:
                per += 1
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

        change_label(label, str(per) + " / " + str(num_way)+' Way Done')

    curses.echo()
    curses.nocbreak()
    curses.endwin()

    change_label(label, 'Finish MPA')

    t=0
    for i_s in range(num_t):
        for i_d in range(i_s + 1, num_t):
            dis_sol, final_sol = threads_ways[t].join()
            map_tsp[i_s][i_d] = dis_sol
            map_way[i_s][i_d] = final_sol
            re_way = final_sol.copy()
            re_way.reverse()
            map_tsp[i_d][i_s] = dis_sol
            map_way[i_d][i_s] = re_way
            t += 1

    print('>'*5+'map_tsp\n', str_map_tsp := str(np.round(map_tsp, 2).T))
    # print(str(map_way))
    # print(goals)
    change_label(label, 'Finish Save Map_TSP')

    ga_tsp = GA_TSP(map_tsp.tolist(), k, m_size,index_s=index_s)
    best_cost, best = ga_tsp.solve()
    print('>'*5+'best_cost\n', best_cost)
    print(map_way)
    # print(best)
    line = []
    if is_have_s:
        best.insert(0,index_s)
        best.append(index_s)
    else:
        best.append(best[0])
    for i in range(len(best)-1):
        line.extend(map_way[best[i]][best[i+1]])
    print('>'*5+'best_cost\n', line)
    change_label(label, 'Finish GA_TSP')
    end_time = time.time()
    total_time =end_time - start_time
    with open(file_name_sol, 'a') as f:
        f.write(str(targets)+'\n')
        f.write(str(best)+'\n')
        f.write(str(best_cost)+'\n')
        f.write(str(line)+'\n')
        f.write(str(total_time)+'\n')
    change_label(label, '%s | Time: %.3f' % (file_name_sol, total_time))
    display = DisplayMatplotlib(m_size,o_env,line,start,[],goals,best_cost)
    display.draw()

# solve('Test/map_.txt',None)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = Tk()  # main win
    root.title("MPA")
    root.geometry('450x300')
    root.resizable(width=0, height=0)

    title_frame = Frame(root)
    title_frame.pack(fill="both")
    main_frame = Frame(root)
    main_frame.pack(fill="both")
    file_input_label = Label(main_frame, text="File")
    file_input_label.grid(row=0, column=0, pady=1)
    file_input = Entry(main_frame, width=50)
    file_input.grid(row=0, column=1, padx=5, pady=1)
    file_input.insert(END, "file_solve")
    file_input.configure(state=DISABLED)

    solve_button_frame = Frame(root)
    solve_button_frame.pack(fill="both")
    rs_label = Entry(solve_button_frame, width=40)
    rs_label.grid(row=1, column=2, padx=30, pady=1)
    rs_label.insert(END, "")
    rs_label.configure(state=DISABLED)

    s_file = Button(solve_button_frame, text="Select", command=partial(select_file, file_input))
    s_file.grid(row=1, column=0, padx=5, pady=1)
    s_file.configure(state=DISABLED)

    s_solve = Button(solve_button_frame, text="Solve", command=partial(solve, file_input, rs_label))
    s_solve.grid(row=1, column=1, padx=5, pady=1)
    s_solve.configure(state=DISABLED)
    Button(title_frame, text="Solve new solutions",
           command=partial(select_solve, s_file, s_solve)).grid(row=0, column=0, padx=5, pady=1)
    # Button(title_frame, text="Solve all solutions in program",
    #        command=partial(solve_all, file_input, rs_label)).grid(row=0, column=1, padx=5, pady=1)
    show_title_frame = Frame(root)
    show_title_frame.pack(fill="both")

    show_file_frame = Frame(root)
    show_file_frame.pack(fill="both")
    Label(show_file_frame, text='File').grid(row=0, column=0, pady=1)
    n = StringVar()
    list_file = Combobox(show_file_frame, width=35, textvariable=n)
    list_file['values'] = ('', '')
    list_file.grid(column=1, row=0, padx=5, pady=1)
    list_file.current(0)
    list_file.configure(state=DISABLED)

    Button(show_title_frame, text="Show solutions",
           command=partial(select_show, list_file)).grid(row=1, column=0, padx=5, pady=1)

    show_result_frame = Frame(root)
    show_result_frame.pack(fill="both")
    Label(show_result_frame, text='Size of Map').grid(row=0, column=0, pady=1)
    show_map = Entry(show_result_frame, width=20)
    show_map.grid(row=0, column=1, padx=5, pady=1)
    show_map.insert(END, "")
    show_map.configure(state=DISABLED)

    Label(show_result_frame, text='Destination').grid(row=1, column=0, pady=1)
    show_des = Entry(show_result_frame, width=20)
    show_des.grid(row=1, column=1, padx=5, pady=1)
    show_des.insert(END, "")
    show_des.configure(state=DISABLED)

    Label(show_result_frame, text='Distance').grid(row=2, column=0, pady=1)
    show_dis = Entry(show_result_frame, width=20)
    show_dis.grid(row=2, column=1, padx=5, pady=1)
    show_dis.insert(END, "")
    show_dis.configure(state=DISABLED)

    Label(show_result_frame, text='Time Solve').grid(row=3, column=0, pady=1)
    show_time = Entry(show_result_frame, width=20)
    show_time.grid(row=3, column=1, padx=5, pady=1)
    show_time.insert(END, "")
    show_time.configure(state=DISABLED)

    show_button_frame = Frame(root)
    show_button_frame.pack(fill="both")

    n_st = StringVar()
    list_st = Combobox(show_button_frame, width=5, textvariable=n_st)
    list_st['values'] = ('', '')
    list_st.grid(column=2, row=0, padx=5, pady=1)
    list_st.current(0)
    list_st.configure(state=DISABLED)

    n_dst = StringVar()
    list_dst = Combobox(show_button_frame, width=5, textvariable=n_dst)
    list_dst['values'] = ('', '')
    list_dst.grid(column=3, row=0, padx=5, pady=1)
    list_dst.current(0)
    list_dst.configure(state=DISABLED)

    # button_se = Button(show_button_frame, text="Environment",
    #                    command=partial(show_environment, list_file), width=13)
    # button_se.grid(row=0, column=0, padx=5, pady=1)

    # button_ptp = Button(show_button_frame, text="Point to Point",
    #                     command=partial(show_ptp, list_file, list_st, list_dst), width=13)
    # button_ptp.grid(row=0, column=1, padx=5, pady=1)

    # button_ss = Button(show_button_frame, text="Solutions",
    #                    command=partial(show_solve, list_file), width=13)
    # button_ss.grid(row=1, column=0, padx=5, pady=1)

    # button_dy = Button(show_button_frame, text="Animations",
    #                    command=partial(show_dynamic, list_file), width=13)
    # button_dy.grid(row=1, column=1, padx=5, pady=1)

    # s_show = Button(show_file_frame, text="Select",
    #                 command=partial(show_result, list_file, show_map, show_des, show_dis, show_time, list_st, list_dst))
    # s_show.grid(row=0, column=2, padx=5, pady=1)

    mainloop()





