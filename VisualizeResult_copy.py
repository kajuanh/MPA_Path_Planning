import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pygame
import math
import matplotlib
import ast
import os



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
    find_another = list(map(list, zip(*np.where(env >= 4))))
    for goal in goals:
        o_env[goal[0], goal[1]] = 0

    start = []
    is_have_s = False
    
    for st in find_start:
        start = st
        o_env[start[0], start[1]] = 0
        is_have_s = True


    for ano in find_another:
        o_env[ano[0], ano[1]] = 0

    return goals, o_env, start, is_have_s


def read_file(filepath:str, type:int = 0):
    '''return map, map_size, (target, order, dis, result, time, map_tsp, map_way
    if type = 1) '''
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
                target = ast.literal_eval(f.readline())
                order = ast.literal_eval(f.readline())
                dis = float(f.readline())
                result = ast.literal_eval(f.readline())
                time = ast.literal_eval(f.readline())
                map_tsp = ast.literal_eval(f.readline())
                map_way = ast.literal_eval(f.readline())
                return np_map, m_s, target, order, dis, result, time, map_tsp, map_way
    except Exception as E:
        print(E)
        raise Exception('error read file ') 

# in total we now have 175 colors in the colormap


class DisplayMatplotlib:
    def __init__(self, m_size, env,line=[],  start= [],goals=[],dis=None):
        self.plt = plt
        self.m_size = m_size
        self.line = line
        self.env = env
        self.start = start
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

    def draw(self, arrow: bool):
        fig, ax = plt.subplots()
        fig.set_size_inches(10, 10)
        if not self.dis is None:
            ax.set_title("Distance = " + str(self.dis))
        else:
            if len(self.start)>1:
                hs = 'start at' + str(self.start)
            else:
                hs = 'no start'
            ax.set_title("Map size %d -- %d goals -- "%(self.m_size,len(self.goals)) + hs)
        ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
        ax.set_xlim(0, self.m_size)
        ax.set_ylim(0, self.m_size)
        plt.xticks([*range(self.m_size)])
        plt.yticks([*range(self.m_size)])
        if arrow:
            self.draw_line_arrow(self.line,ax)
        else:
            self.draw_line(self.line)

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

class VisualizeResult:
    m_size = None
    env = None
    o_env=None,
    goals=[]
    start=[]
    target=[]
    order=[]
    dis=None
    result=[]
    is_have_s = False
    time=None
    
from tkinter import messagebox
from tkinter import filedialog


class DisplayPygame:
    def __init__(self,map_size,env,MARGIN=1,BLOCKSIZE = 40) -> None:
        self.map_size =map_size
        self.env =env
        self.MARGIN = MARGIN
        self.BLOCKSIZE = BLOCKSIZE-MARGIN
        WINDOW_WIDTH = WINDOW_HEIGHT = map_size*(BLOCKSIZE)
        global SCREEN, CLOCK
        SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        CLOCK = pygame.time.Clock()
        self.work = {3: 'start', 1: 'obstacle', 2: 'goal'}
        self.key_math = 1

    def drawGrid(self):
        for x in range(self.map_size):
            for y in range(self.map_size):
                rect = pygame.Rect(
                    x*(self.BLOCKSIZE+self.MARGIN) ,
                    y*(self.BLOCKSIZE+self.MARGIN),
                    self.BLOCKSIZE, self.BLOCKSIZE
                    )
                if self.env[x][y] == 1:
                    pygame.draw.rect(SCREEN, BLACK, rect)
                elif self.env[x][y] == 2:
                    pygame.draw.rect(SCREEN, YELLOW, rect)
                elif self.env[x][y] == 3:
                    pygame.draw.rect(SCREEN, GREEN, rect)
                else:
                    pygame.draw.rect(SCREEN, BLACK, rect, self.MARGIN)
    def save(self):
        files = [('All Files', '*.*'), 
                    ('Python Files', '*.py'),
                    ('Text Document', '*.txt')]
        file =  filedialog.asksaveasfile(filetypes = files, defaultextension = files)
        if file is not None:
            with open(file.name,'w') as f:
                f.write(str(self.map_size)+'\n')
                save_env = self.env.transpose()
                print(save_env)
                for line in save_env:
                    f.write(' '.join(list(map(str,(line.tolist()))))+'\n')
            
            return os.path.relpath(file.name)
        else:
            return None
    def event_create_map(self,done): 
        file_path = None       
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                save = messagebox.askokcancel('Done','Do you want to save')
                if save:
                    file_path = self.save()
                done = True
                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                pos =pygame.mouse.get_pos()
                x = pos[0]//(self.BLOCKSIZE+self.MARGIN)
                y = pos[1]//(self.BLOCKSIZE+self.MARGIN)
                print('Click ', pos, 'Grid coordinates: (', x, ',', y, ')')

                if self.key_math == 1:
                    if self.env[x, y] == 1:
                        self.env[x, y] = 0
                    elif self.env[x, y] == 0:
                        self.env[x, y] = 1
                elif self.key_math == 2:
                    if self.env[x, y] == 2:
                        self.env[x, y] = 0
                    elif self.env[x, y] == 0:
                        self.env[x, y] = 2
                elif self.key_math == 3:
                    if self.env[x, y] == 0:
                        os_x, os_y = np.where(self.env==3)
                        if len(os_x) >0 and len(os_y)>0:
                            self.env[int(os_x)][int(os_y)] = 0
                        self.env[x][y] = 3
                    elif self.env[x, y] == 3:
                        self.env[x][y] = 0
            elif event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                if event.key == pygame.K_s:
                    self.key_math = 3
                elif event.key == pygame.K_o:
                    self.key_math = 1
                elif event.key == pygame.K_g:
                    self.key_math = 2
        return done, file_path
    def create_map(self):
        done = False
        while not done:
            SCREEN.fill(WHITE)
            pygame.display.set_caption('Update %s (key-> o: obstacle; s: start; g: goal)'%self.work[self.key_math])

            done, file_path = self.event_create_map(done)
            self.drawGrid()
            pygame.display.flip()
        pygame.quit()
        return file_path


# env, ms = read_file('Test/map_.txt')
# d = DisplayPygame(ms,env,1,60-ms)
# e = d.create_map()