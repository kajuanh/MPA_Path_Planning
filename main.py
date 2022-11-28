# This is a sample Python script.
import os
import time
import shutil
import curses
import numpy as np
from MPA import MPA
from tkinter import *
from GA import GA_TSP
from pathlib import Path
from tkinter.ttk import *
from VisualizeResult import *
from functools import partial
from tkinter import filedialog
from GripMap import ThreadWithReturnValue
solutions_dir = os.path.abspath("Solutions")

def change_label(label: Entry,str_display: str): 
    label.configure(state=NORMAL)
    label.delete(0, 100)  # delete
    label.insert(END, str_display) # insert
    label.configure(state=DISABLED)  # enable
    label.update()

def select_file(entry_input, list_ms):
    file = filedialog.askopenfilename()
    entry_input.configure(state=NORMAL)
    entry_input.delete(0, 100)  # delete
    entry_input.insert(END, str(os.path.relpath(file)))  # insert
    entry_input.configure(state=DISABLED,foreground='black')  # enable
    list_ms.configure(state=DISABLED)

def select_solve(select_button, solve_button, list_ms, new_map_button):
    select_button.configure(state=NORMAL)
    solve_button.configure(state=NORMAL)
    list_ms.configure(state=NORMAL)
    new_map_button.configure(state=NORMAL)

def select_show(select_combobox, s_show):
    select_combobox.configure(state=NORMAL)
    select_combobox['values'] = tuple(os.path.join('Solutions',file.name) for file in Path(solutions_dir).rglob('*.txt'))#tuple(glob.glob('*.txt'))
    select_combobox.current(0)
    select_combobox.configure(state='readonly')
    s_show.configure(state=NORMAL)

def solve(file_input: Entry, label: Entry):
    filepath = file_input.get()
    # print('->>>>>',filepath:= file_input.get())
    start_time = time.time()
    env, m_size = read_file(filepath)
    goals, o_env, start, is_have_s = split_map(env)
    num_g = num_t = len(goals)
    targets = goals.copy()

    if is_have_s:
        s = '_s'
        num_t += 1
        targets.insert(0, start)
        targets.sort()
        index_s = targets.index(start)
    else:
        s = ''
        index_s = None

    k = (num_t)*(num_t-1)
    map_tsp = np.full((num_t, num_t), 0.0, dtype=float)
    map_way = np.empty((num_t, num_t), dtype=object)

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
            if t < 20:
                stdscr.addstr(t, 0, "{2} {0} : {1}".format(key, item, t))
            elif 20 < t < 40:
                stdscr.addstr(t-20, 40, "{2} {0} : {1}".format(key, item, t))
            elif 40<t<60:
                stdscr.addstr(t-40, 80, "{2} {0} : {1}".format(key, item, t))
            elif 60<t:
                stdscr.addstr(t-60, 120, "{2} {0} : {1}".format(key, item, t))
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
        f.write(str(map_tsp.tolist())+'\n')
        f.write(str(map_way.tolist())+'\n')
    change_label(label, '%s | Time: %.3f' % (file_name_sol, total_time))
    display = DisplayMatplotlib(m_size,o_env,line,start,goals,best_cost)
    display.draw(arrow=False)

def show_environment(vslr: VisualizeResult):
    showEnvironment = DisplayMatplotlib(vslr.m_size, vslr.o_env, [], vslr.start, vslr.goals)
    showEnvironment.draw(arrow=False)

def show_result(select_combobox, s_map, s_des, show_o, s_dis,  
                s_time, l_st, l_dst, vslr: VisualizeResult,
                button_se, button_ptp, button_ss, button_dy):
    
    env, m_size, target, order, dis, result, time, map_tsp, map_way = read_file (select_combobox.get(),1)
    vslr.m_size = m_size
    vslr.env = env
    goals, o_env, start, is_have_s = split_map(env)
    vslr.goals = goals
    vslr.o_env = o_env
    vslr.start = start
    vslr.is_have_s = is_have_s
    vslr.target = target
    vslr.order = order
    vslr.dis = dis
    vslr.result = result
    vslr.time = time

    change_label(s_map,str(m_size))
    change_label(s_des,str(target))
    change_label(show_o,str(order))
    change_label(s_dis,str(dis))
    change_label(s_time,str(time))
    list_point = tuple([str(point) for point in range(len(target))])
    l_st.configure(state=NORMAL)
    l_st['values'] = list_point
    l_st.current(0)
    l_st.configure(state='readonly')

    l_dst.configure(state=NORMAL)
    l_dst['values'] = list_point
    l_dst.current(0)
    l_dst.configure(state='readonly')
    
    button_se.configure(state=NORMAL)
    button_ptp.configure(state=NORMAL)
    button_ss.configure(state=NORMAL)
    button_dy.configure(state=NORMAL)

def show_ptp(l_st, l_dst, vslr: VisualizeResult):
    print('----show_ptp-----')
    st = int(l_st.get())
    dst = int(l_dst.get())
    rev_result = vslr.result.copy()
    rev_result.reverse()
    index_st = vslr.result.index(vslr.target[st])
    index_dst = len(vslr.result) - rev_result.index(vslr.target[dst])
    print(vslr.target[st],vslr.target[dst])
    if index_st > index_dst:
        line = vslr.result[index_dst-1:index_st+1]
    else:
        line = vslr.result[index_st:index_dst]
    print(index_st,index_dst)
    print(line)

    show_ptp = DisplayMatplotlib(vslr.m_size, vslr.o_env, line, vslr.start, vslr.goals)
    show_ptp.draw(arrow=True)

def create_map(list_ms, file_input):
    print('-----create_map------')
    m_s = int(list_ms.get())
    file_path = file_input.get()
    if '.txt' not in file_path:
        empty_map = np.zeros((m_s,m_s),dtype=int)
        display_map = DisplayPygame(map_size=m_s,env=empty_map,BLOCKSIZE=60-m_s)
        display_map.create_map()
        pygame.quit()
    else:
        env, m_size = read_file(file_path)
        display_map = DisplayPygame(map_size=m_size,env=env,BLOCKSIZE=60-m_size)
        file_path = display_map.create_map()
        change_label(file_input,str(file_path))
    print('----end create-------')

def show_solve(vslr: VisualizeResult):
    showEnvironment = DisplayMatplotlib(vslr.m_size, vslr.o_env, vslr.result, vslr.start, vslr.goals, vslr.dis)
    showEnvironment.draw(arrow=False)

def show_dynamic(select_combobox):
    # visualizeResult = VisualizeResult(select_combobox.get())
    # visualizeResult.showSolutionDynamic()
    pass


if __name__ == '__main__':
    vslr = VisualizeResult
    root = Tk()  # main win
    root.title("MPA")
    root.geometry('500x400')
    root.resizable(width=0, height=0)

    title_frame = Frame(root)
    title_frame.pack(fill="both")
    main_frame = Frame(root)
    main_frame.pack(fill="both")
    file_input_label = Label(main_frame, text="File")
    file_input_label.grid(row=0, column=0, pady=5)
    file_input = Entry(main_frame, width=50 )
    file_input.grid(row=0, column=1, padx=5, pady=5)
    file_input.insert(END, "file_solve")
    file_input.configure(state=DISABLED)

    solve_button_frame = Frame(root)
    solve_button_frame.pack(fill="both")
    rs_label = Entry(solve_button_frame, width=40)
    rs_label.grid(row=1, column=2, padx=5, pady=5)
    rs_label.insert(END, "")
    rs_label.configure(state=DISABLED)
    
    n_m_size = StringVar()
    list_ms = Combobox(title_frame, width=5, textvariable=n_m_size)
    list_ms['values'] = [*range(10,31,5)]
    list_ms.grid(column=2, row=0, padx=5, pady=5)
    list_ms.current(0)
    list_ms.configure(state=DISABLED)



    s_file = Button(solve_button_frame, text="Select", command=partial(select_file, file_input, list_ms))
    s_file.grid(row=1, column=0, padx=5, pady=5)
    s_file.configure(state=DISABLED)

    s_solve = Button(solve_button_frame, text="Solve", command=partial(solve, file_input, rs_label))
    s_solve.grid(row=1, column=1, padx=5, pady=5)
    s_solve.configure(state=DISABLED)
    
    new_map = Button(title_frame, text="Create new map", command=partial(create_map, list_ms, file_input))
    new_map.grid(row=0, column=1, padx=5, pady=5)
    new_map.configure(state=DISABLED)

    Button(title_frame, text="Solve new solutions",
           command=partial(select_solve, s_file, s_solve, list_ms, new_map)).grid(row=0, column=0, padx=5, pady=5)
    
    
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


    show_result_frame = Frame(root)
    show_result_frame.pack(fill="both")
    Label(show_result_frame, text='Size of Map').grid(row=0, column=0, pady=1)
    show_map = Entry(show_result_frame, width=40, foreground='black')
    show_map.grid(row=0, column=1, padx=5, pady=1)
    show_map.insert(END, "")
    show_map.configure(state=DISABLED)

    Label(show_result_frame, text='Destination').grid(row=1, column=0, pady=1)
    show_des = Entry(show_result_frame, width=40, foreground='black')
    show_des.grid(row=1, column=1, padx=5, pady=1)
    show_des.insert(END, "")
    show_des.configure(state=DISABLED)

    Label(show_result_frame, text='Order').grid(row=2, column=0, pady=1)
    show_o = Entry(show_result_frame, width=40, foreground='black')
    show_o.grid(row=2, column=1, padx=5, pady=1)
    show_o.insert(END, "")
    show_o.configure(state=DISABLED)


    Label(show_result_frame, text='Distance').grid(row=3, column=0, pady=1)
    show_dis = Entry(show_result_frame, width=40, foreground='black')
    show_dis.grid(row=3, column=1, padx=5, pady=1)
    show_dis.insert(END, "")
    show_dis.configure(state=DISABLED)

    Label(show_result_frame, text='Time Solve').grid(row=4, column=0, pady=1)
    show_time = Entry(show_result_frame, width=40, foreground='black')
    show_time.grid(row=4, column=1, padx=5, pady=1)
    show_time.insert(END, "")
    show_time.configure(state=DISABLED)#,background='White',foreground='yellow')

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

    button_se = Button(show_button_frame, text="Environment", command=partial(show_environment, vslr), width=13)
    button_se.grid(row=0, column=0, padx=5, pady=1)
    button_se.configure(state=DISABLED)

    button_ptp = Button(show_button_frame, text="Point to Point", command=partial(show_ptp, list_st, list_dst, vslr), width=13)
    button_ptp.grid(row=0, column=1, padx=5, pady=1)
    button_ptp.configure(state=DISABLED)

    button_ss = Button(show_button_frame, text="Solutions", command=partial(show_solve, vslr), width=13)
    button_ss.grid(row=1, column=0, padx=5, pady=1)
    button_ss.configure(state=DISABLED)


    button_dy = Button(show_button_frame, text="Animations", command=partial(show_dynamic, list_file), width=13)
    button_dy.grid(row=1, column=1, padx=5, pady=1)
    button_dy.configure(state=DISABLED)

    s_show = Button(show_file_frame, text="Select", command=partial(
                        show_result, list_file, show_map, show_des, show_dis,
                        show_time, show_o, list_st, list_dst, vslr,
                        button_se, button_ptp, button_ss, button_dy
                        ))
    s_show.grid(row=0, column=2, padx=5, pady=1)
    s_show.configure(state=DISABLED)
    
    Button(show_title_frame, text="Show solutions",
        command=partial(select_show, list_file, s_show)).grid(row=1, column=0, padx=5, pady=1)
    Button(root, text="Quit", command=root.quit).pack() #button to close the window

    root.mainloop()
