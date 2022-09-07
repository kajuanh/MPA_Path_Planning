import matplotlib.pyplot as plt
import numpy as np
import pygame
import math


class VisualizeResult:

    def __init__(self, filename):
        fp = open(filename, "r")
        self.n = int(fp.readline())
        num_des = int(fp.readline())
        self.list_des = []
        self.environment = []
        f_map = []
        for f_i in range(num_des):
            s = fp.readline()
            des = s[:-1].split(" ")
            self.list_des.append([float(des[0]), float(des[1])])
        for f_i in range(self.n):
            s = fp.readline()
            line = s[:-2].split(" ")
            f_map.append(list(map(int, line)))
        for f_i in f_map:
            self.environment.append(list(f_i))
        self.map_tsp = [[[0, []] for _ in range(num_des)] for _ in range(num_des)]
        for i_s in range(num_des):
            for i_d in range(i_s + 1, num_des):
                dis_ds = float(fp.readline())
                line = fp.readline()[:-2].split(" ")
                sol = list(map(float, line))
                way_sd = []
                for i in range(int(len(sol) / 2)):
                    way_sd.append([sol[2 * i], sol[2 * i + 1]])
                self.map_tsp[i_s][i_d] = [dis_ds, way_sd]
                re_way = list(way_sd)
                re_way.reverse()
                self.map_tsp[i_d][i_s] = [dis_ds, re_way]
        self.dis_sol = float(fp.readline())
        self.solution = list(map(int, fp.readline()[:-2].split(" ")))
        self.time_solution = float(fp.readline())
        self.list_move = []
        pre = self.solution[-1]
        for i in self.solution:
            self.list_move += self.map_tsp[pre][i][1]
            pre = i
        fp.close()

    # def __init__(self, f_map, f_list_dst, f_tsp, f_sol):
    #     self.environment = f_map
    #     self.list_des = f_list_dst
    #     self.map_tsp = f_tsp
    #     self.n = len(f_map)
    #     self.solution = f_sol
    #     pre = self.solution[-1]
    #     self.list_move = []
    #     for i in self.solution:
    #         self.list_move += self.map_tsp[pre][i][1]
    #         pre = i

    def showEnvironment(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        ax.set_title("Distance = " + str(self.dis_sol))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.plot([0, self.n, self.n, 0, 0], [0, 0, self.n, self.n, 0], color='red')
        x_plt = np.arange(0, self.n, 0.1)

        for des in self.list_des:
            ax.fill_between(x_plt, self.n - 0.5 - des[0], self.n - des[0] + 0.5,
                            where=(x_plt >= des[1] - 0.5) & (x_plt <= des[1] + 0.5), color='yellow')

        for i_map in range(self.n):
            for j_map in range(self.n):
                if self.environment[i_map][j_map] == 1:
                    ax.fill_between(x_plt, self.n - 1 - i_map, self.n - i_map,
                                    where=(x_plt >= j_map) & (x_plt <= j_map + 1), color='red')
        plt.show()

    def showPointToPoint(self, f_st, f_dst):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        ax.set_title("Distance = " + str(self.map_tsp[f_st][f_dst][0]))
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        xL = []
        yL = []
        for i_sd in self.map_tsp[f_st][f_dst][1]:
            xL.append(i_sd[1])
            yL.append(self.n - i_sd[0])
        ax.plot(xL, yL, color='blue')
        ax.plot([0, self.n, self.n, 0, 0], [0, 0, self.n, self.n, 0], color='red')
        x_plt = np.arange(0, self.n, 0.1)

        for des in self.list_des:
            ax.fill_between(x_plt, self.n - 0.5 - des[0], self.n - des[0] + 0.5,
                            where=(x_plt >= des[1] - 0.5) & (x_plt <= des[1] + 0.5), color='yellow')

        for i_map in range(self.n):
            for j_map in range(self.n):
                if self.environment[i_map][j_map] == 1:
                    ax.fill_between(x_plt, self.n - 1 - i_map, self.n - i_map,
                                    where=(x_plt >= j_map) & (x_plt <= j_map + 1), color='red')
        plt.show()

    def showSolution(self):
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=120)
        ax.set_title("MPA")
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.set_xticks([])
        ax.set_xticklabels([])
        xL = []
        yL = []
        for i_sd in self.list_move:
            xL.append(i_sd[1])
            yL.append(self.n - i_sd[0])
        ax.plot(xL, yL, color='blue')
        ax.plot([0, self.n, self.n, 0, 0], [0, 0, self.n, self.n, 0], color='red')
        x_plt = np.arange(0, self.n, 0.1)

        for des in self.list_des:
            ax.fill_between(x_plt, self.n - 0.5 - des[0], self.n - des[0] + 0.5,
                            where=(x_plt >= des[1] - 0.5) & (x_plt <= des[1] + 0.5), color='yellow')

        for i_map in range(self.n):
            for j_map in range(self.n):
                if self.environment[i_map][j_map] == 1:
                    ax.fill_between(x_plt, self.n - 1 - i_map, self.n - i_map,
                                    where=(x_plt >= j_map) & (x_plt <= j_map + 1), color='red')
        plt.show()

    def showSolutionDynamic(self):
        origin = 0.4

        list_des_pass = []
        list_des_arrive = []
        for i in self.solution:
            list_des_arrive.append([self.list_des[i][0], self.list_des[i][1]])
        start = list_des_arrive[0]
        list_des_arrive.append(start)

        pygame.init()
        scale = 600 / self.n
        win = pygame.display.set_mode((self.n * scale, self.n * scale))

        pygame.display.set_caption("MPA")
        list_move_py = []
        for i in self.list_move:
            list_move_py.append([i[1] * scale, i[0] * scale])
        x = list_move_py[0][0]
        y = list_move_py[0][1]

        width = 2 * origin * scale
        height = 2 * origin * scale

        vel = self.n / 6
        run = True

        while run:
            pygame.time.delay(10)
            if len(list_move_py) >= 2:
                f_st = list_move_py[0]
                f_dst = list_move_py[1]
                if f_st[0] == f_dst[0] and f_st[1] == f_dst[1]:
                    list_move_py.pop(0)
                    f_st = list_move_py[0]
                    f_dst = list_move_py[1]
                f_h = math.sqrt((f_dst[1] - f_st[1]) * (f_dst[1] - f_st[1])
                                + (f_dst[0] - f_st[0]) * (f_dst[0] - f_st[0]))
                pi_x = (f_dst[0] - f_st[0]) / f_h
                pi_y = (f_dst[1] - f_st[1]) / f_h
                vel_x = vel * pi_x
                vel_y = vel * pi_y
                x += vel_x
                y += vel_y
                if (x - f_dst[0]) * (x - f_st[0]) > 0 or (y - f_dst[1]) * (y - f_st[1]) > 0:
                    x = f_dst[0]
                    y = f_dst[1]
                    if list_des_arrive[0][0] == round(y / scale, 2) and list_des_arrive[0][1] == round(x / scale, 2):
                        list_des_pass.append(list_des_arrive[0])
                        list_des_arrive.pop(0)
                    list_move_py.pop(0)
            else:
                x = list_move_py[0][0]
                y = list_move_py[0][1]

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

            win.fill((255, 255, 255))
            for des in list_des_arrive:
                pygame.draw.rect(win, (255, 255, 0), [(des[1] - 0.5) * scale, (des[0] - 0.5) * scale, scale, scale])
            for des in list_des_pass:
                pygame.draw.rect(win, (0, 255, 255), [(des[1] - 0.5) * scale, (des[0] - 0.5) * scale, scale, scale])

            pygame.draw.rect(win, (0, 0, 255), [x - origin * scale, y - origin * scale, width, height])

            for i in range(self.n):
                for j in range(self.n):
                    if self.environment[i][j] == 1:
                        pygame.draw.rect(win, (255, 0, 0), [j * scale, i * scale, scale, scale])

            pygame.display.update()

        pygame.quit()
