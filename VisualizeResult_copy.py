import matplotlib.pyplot as plt
import numpy as np
import pygame
import math
import matplotlib
BLACK = (0, 0, 0,1)
WHITE = (250, 250, 250,1)
YELLOW = (215, 225, 88,1)
GRAY = (150, 150, 100,1)
BLUE = (0,0,230,1)
GREEN = (0,230,0,1)

import matplotlib.colors as mcolors
def split_map(env: np):
    '''return goals , start, end, origin map and is_have_se (have start, end)'''
    o_env = env.copy()
    goals = list(map(list, zip(*np.where(env == 2))))
    start = list(map(list, zip(*np.where(env == 3))))[0]
    end = list(map(list, zip(*np.where(env == 4))))[0]
    for goal in goals:
        o_env[goal[0], goal[1]] = 0
    if len(start) > 1 and len(end) > 1:
        o_env[start[0], start[1]] = 0
        o_env[end[0], end[1]] = 0
        is_have_se = True
    else:
        is_have_se = False

    return goals, start, end, o_env, is_have_se


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
    except:
        raise Exception('error read file ') 

# in total we now have 175 colors in the colormap


colors = np.vstack((WHITE, BLACK, YELLOW, GREEN, BLUE))
print(colors)
cmap=mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)
class DisplayMatplotlib:
    def __init__(self, plt, m_size, line, env, start= [],end=[],goals=[]):
        self.plt = plt
        self.m_size = m_size
        self.line = line
        self.env = env
        self.start = env
        self.end = env
        self.goals = env
        
    def draw_line(self, points: list[list], plt: matplotlib.pyplot):
        xl = []
        yl = []
        for point in points:
            xl.append(point[0]+0.5)
            yl.append(point[1]+0.5)
        plt.plot(xl, yl, '>-')

    def draw(self):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)

        ax.tick_params(top=True, labeltop=True,
                       bottom=False, labelbottom=False)
        ax.set_xlim(0, self.m_size)
        ax.set_ylim(0, self.m_size)
        plt.xticks([*range(self.m_size)])
        plt.yticks([*range(self.m_size)])
        self.draw_line(self.line, self.plt)
        
        plt.imshow(self.env.transpose(), cmap=cmap,origin='lower',
                   extent=(0, self.m_size, 0, self.m_size))
        plt.grid()
        ax.invert_yaxis()

        plt.show(block=False)
        plt.pause(50)