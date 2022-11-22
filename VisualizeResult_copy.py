import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pygame
import math
import matplotlib
BLACK = (0, 0, 0,1)
WHITE = (250, 250, 250,1)
YELLOW = (215, 225, 88,1)
GRAY = (150/255, 150/255, 100/255,1)
BLUE = (0,0,230,1)
GREEN = (0,230,0,1)

def split_map(env: np):
    '''return goals, o_env, start, is_have_s (have start)'''
    o_env = env.copy()
    goals = list(map(list, zip(*np.where(env == 2))))
    find_start = list(map(list, zip(*np.where(env == 3))))

    for goal in goals:
        o_env[goal[0], goal[1]] = 0

    if len(find_start) > 1:
        start = find_start[0]
        o_env[start[0], start[1]] = 0
        is_have_s = True
    else:
        is_have_s = False
        start = []

    return goals, o_env, start, is_have_s


def read_file(filepath:str, type:int = 0):
    '''return map, map_size, (map_tsp, dis if type = 1) '''
    try:
        with open(filepath) as f:
            m_s = int(f.readline())
            np_map = np.zeros((m_s, m_s), int)
            for line in range(m_s):
                np_map[line] = (f.readline()).strip().split()
            np_map = np_map.transpose()
            if type == 0:
                return np_map, m_s
            elif type == 1:
                pass
    except Exception as E:
        print(E)
        raise Exception('error read file ') 

# in total we now have 175 colors in the colormap


class DisplayMatplotlib:
    def __init__(self, m_size, env,line=[],  start= [],end=[],goals=[],dis=None):
        self.plt = plt
        self.m_size = m_size
        self.line = line
        self.env = env
        self.start = start
        self.end = end
        self.goals = goals
        self.dis = dis
    def draw_line(self, points: list[list]):#, plt: matplotlib.pyplot):
        xl = []
        yl = []
        for point in points:
            xl.append(point[0]+0.5)
            yl.append(point[1]+0.5)
        plt.plot(xl, yl, '-')
    
    def draw_line_arrow(self, points: list[list],ax):#, plt: matplotlib.pyplot):
        for i in range (len(points)-1):
            x1 =  points[i][0]+0.5
            y1 = points[i][1]+0.5
            x2 =  points[i+1][0]+0.5
            y2 = points[i+1][1]+0.5
            x_mid = (x1+x2)/2
            y_mid = (y1+y2)/2
            dx = (x2-x1)*0.01
            dy = (y2-y1)*0.01
            plt.plot([x1,x2],[y1,y2],'b-')

            plt.arrow(x_mid, y_mid, dx, dy, head_width=0.2, head_length=0.2, length_includes_head=True, color='b')

    def draw(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        ax.set_title("Distance = " + str('????' if self.dis is None else self.dis))
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlim(0, self.m_size)
        ax.set_ylim(0, self.m_size)
        plt.xticks([*range(self.m_size)])
        plt.yticks([*range(self.m_size)])
        # self.draw_line(self.line, self.plt)
        self.draw_line_arrow(self.line,ax)
        if len(self.start) > 0 :
            rect = Rectangle(tuple(self.start), 1, 1, linewidth=1, edgecolor=GRAY, facecolor='g')
            ax.add_patch(rect)
        for goal in self.goals:
            rect = Rectangle(tuple(goal), 1, 1, linewidth=1, edgecolor=GRAY, facecolor='y')
            ax.add_patch(rect)
        plt.imshow(self.env.transpose(), cmap='binary', origin='lower', extent=(0, self.m_size, 0, self.m_size))
        plt.grid()
        ax.invert_yaxis()

        plt.show(block=False)
        plt.pause(50)