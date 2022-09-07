# This is a sample Python script.
import glob

from MPA import *
from GA import *
from VisualizeResult import *
import datetime
from tkinter import *
from tkinter.ttk import *
from tkinter import filedialog
from functools import partial


def solve_all(filename, label):
    list_file_solve = glob.glob('TestCase\**\*.txt', recursive=True)
    for file in list_file_solve:
        filename.configure(state=NORMAL)
        filename.delete(0, 100)  # delete
        filename.insert(END, file)  # insert
        filename.configure(state=DISABLED)  # enable
        filename.update()
        solve(filename, label, True)


# solve
def solve(filename, label, isSolveAll=False):
    now = datetime.datetime.now()
    start_time = datetime.datetime.timestamp(now)
    mpa_sol = MPA(filename.get())
    filenamesol = "Solutions\map" + str(mpa_sol.n) + "_" + str(len(mpa_sol.list_dst)) + "_sol.txt"
    if isSolveAll:
        case_map = filename.get()[13:14]
        filenamesol = "Solutions\case" + case_map + "_map" + str(mpa_sol.n) + "_" + str(len(mpa_sol.list_dst)) + "_sol.txt"
    fp = open(filenamesol, "w")
    fp.write(str(mpa_sol.n) + "\n")
    fp.write(str(len(mpa_sol.list_dst)) + "\n")
    for dst in mpa_sol.list_dst:
        fp.write(str(dst[0]) + " " + str(dst[1]) + "\n")
    for line in mpa_sol.environment:
        for i in line:
            fp.write(str(i) + " ")
        fp.write("\n")

    list_des = mpa_sol.list_dst
    map_tsp = [[[0, []] for _ in range(len(list_des))] for _ in range(len(list_des))]
    per = 0
    num_edge = int(len(list_des) * (len(list_des) - 1) / 2)
    for i_s in range(len(list_des)):
        for i_d in range(i_s + 1, len(list_des)):
            label.configure(state=NORMAL)
            label.delete(0, 100)  # delete
            label.insert(END, str(per) + " / " + str(num_edge))  # insert
            label.configure(state=DISABLED)  # enable
            label.update()
            per += 1

            dis_ds, way_sd = mpa_sol.way(list_des[i_s], list_des[i_d])
            map_tsp[i_s][i_d] = [dis_ds, way_sd]
            re_way = list(way_sd)
            re_way.reverse()
            map_tsp[i_d][i_s] = [dis_ds, re_way]
            fp.write(str(round(dis_ds, 2)) + "\n")
            str_sol = ""
            for sol in way_sd:
                str_sol += str(sol[0]) + " " + str(sol[1]) + " "
            fp.write(str_sol + "\n")

    label.configure(state=NORMAL)
    label.delete(0, 100)  # delete
    label.insert(END, "Solve GA")  # insert
    label.configure(state=DISABLED)  # enable
    ga_tsp = GA(map_tsp)
    sol = ga_tsp.solve()
    fp.write(str(round(sol[0], 2)) + "\n")
    print(sol[0])
    list_sol = sol[1]
    for i in list_sol:
        fp.write(str(i) + " ")
    fp.write("\n")

    now = datetime.datetime.now()
    end_time = datetime.datetime.timestamp(now)
    fp.write(str(round(end_time - start_time, 2)) + "\n")
    fp.close()

    label.configure(state=NORMAL)
    label.delete(0, 100)  # delete
    label.insert(END, "Done")  # insert
    label.configure(state=DISABLED)  # enable


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
    select_combobox['values'] = tuple(glob.glob('Solutions\*.txt'))
    select_combobox.current(0)
    select_combobox.configure(state='readonly')


def show_result(select_combobox, s_map, s_des, s_dis, s_time, l_st, l_dst):
    visualizeResult = VisualizeResult(select_combobox.get())
    s_map.configure(state=NORMAL)
    s_map.delete(0, 100)  # delete
    s_map.insert(END, str(visualizeResult.n))  # insert
    s_map.configure(state=DISABLED)  # enable

    s_des.configure(state=NORMAL)
    s_des.delete(0, 100)  # delete
    s_des.insert(END, str(len(visualizeResult.list_des)))  # insert
    s_des.configure(state=DISABLED)  # enable

    s_dis.configure(state=NORMAL)
    s_dis.delete(0, 100)  # delete
    s_dis.insert(END, str(visualizeResult.dis_sol))  # insert
    s_dis.configure(state=DISABLED)  # enable

    s_time.configure(state=NORMAL)
    s_time.delete(0, 100)  # delete
    s_time.insert(END, str(visualizeResult.time_solution))  # insert
    s_time.configure(state=DISABLED)  # enable

    l_st.configure(state=NORMAL)
    l_st['values'] = tuple([str(point) for point in range(len(visualizeResult.list_des))])
    l_st.current(0)
    l_st.configure(state='readonly')

    l_dst.configure(state=NORMAL)
    l_dst['values'] = tuple([str(point) for point in range(len(visualizeResult.list_des))])
    l_dst.current(0)
    l_dst.configure(state='readonly')


def show_environment(select_combobox):
    visualizeResult = VisualizeResult(select_combobox.get())
    visualizeResult.showEnvironment()


def show_ptp(select_combobox, l_st, l_dst):
    visualizeResult = VisualizeResult(select_combobox.get())
    st = int(l_st.get())
    dst = int(l_dst.get())
    if st == dst:
        visualizeResult.showEnvironment()
    else:
        visualizeResult.showPointToPoint(st, dst)


def show_solve(select_combobox):
    visualizeResult = VisualizeResult(select_combobox.get())
    visualizeResult.showSolution()


def show_dynamic(select_combobox):
    visualizeResult = VisualizeResult(select_combobox.get())
    visualizeResult.showSolutionDynamic()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    root = Tk()  # main win
    root.title("MPA")
    root.geometry('350x280')
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
    rs_label = Entry(solve_button_frame, width=20)
    rs_label.grid(row=1, column=2, padx=35, pady=1)
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
    Button(title_frame, text="Solve all solutions in program",
           command=partial(solve_all, file_input, rs_label)).grid(row=0, column=1, padx=5, pady=1)
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

    button_se = Button(show_button_frame, text="Environment",
                       command=partial(show_environment, list_file), width=13)
    button_se.grid(row=0, column=0, padx=5, pady=1)

    button_ptp = Button(show_button_frame, text="Point to Point",
                        command=partial(show_ptp, list_file, list_st, list_dst), width=13)
    button_ptp.grid(row=0, column=1, padx=5, pady=1)

    button_ss = Button(show_button_frame, text="Solutions",
                       command=partial(show_solve, list_file), width=13)
    button_ss.grid(row=1, column=0, padx=5, pady=1)

    button_dy = Button(show_button_frame, text="Animations",
                       command=partial(show_dynamic, list_file), width=13)
    button_dy.grid(row=1, column=1, padx=5, pady=1)

    s_show = Button(show_file_frame, text="Select",
                    command=partial(show_result, list_file, show_map, show_des, show_dis, show_time, list_st, list_dst))
    s_show.grid(row=0, column=2, padx=5, pady=1)

    mainloop()

    #
    # visualizeResult = VisualizeResult("Solutions\map15_4_sol.txt")
    # for i in range(4):
    #     for j in range(i + 1, 4):
    #         visualizeResult.showPointToPoint(i, j)
    # visualizeResult.showSolution()
    # visualizeResult.showSolutionDynamic()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
