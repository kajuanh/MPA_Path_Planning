import pygame
import math
import numpy as np
import time
# # # # #
# # # # from tkinter import *
# # # # def motion(event):
# # # #   print("Mouse position: (%s %s)" % (event.x, event.y))
# # # #   return

# # # # master = Tk()
# # # # whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
# # # # msg = Message(master, text = whatever_you_do)
# # # # msg.config(bg='lightgreen', font=('times', 24, 'italic'))
# # # # msg.bind('<Motion>',motion)
# # # # msg.pack()
# # # # master.mainloop()

# # # # with open('Test/map15_3.txt', "r") as file:
# # # #   #area = map_size*map_size (map_size located on line 1 of the file)
# # # #   map_size = int(file.readline())
# # # #   #number of goal (num_goal located on line 2 of the file)
# # # #   num_goal = int(file.readline())
# # # #   #get coordinates of goal on (num_goal) line next
# # # #   list_goal = []
# # # #   for i in range(num_goal):
# # # #     goal = file.readline().strip()
# # # #     goal_coordinates = tuple(map(to_center, goal.split()))
# # # #     list_goal.append(goal_coordinates)

# # # #   #get map from file
# # # #   np_map = np.zeros((map_size, map_size))
# # # #   for line in range(map_size):
# # # #     np_map[line] = (file.readline()).strip().split()
# # # #   l_dst = np.where(np_map == 1)
# # # #   l_dst = list(zip(l_dst[0], l_dst[1]))
# # # #   l_goal = np.where(np_map == 1)
# # # #   l_goal = list(zip(l_goal[0], l_goal[1]))


def to_center(var: str):
  return int(var)+0.5


class MPA:
  environment = None
  map_size = None
  goals = []
  obstacles = []
  x_min = 1
  x_max = 5
  d_min = 2
  origin = 0.4

  def __init__(self, filepath):
    with open(filepath, "r") as file:
      #area = map_size*map_size (map_size located on line 1 of the file)
      map_size = int(file.readline())

      #number of goal (num_goal located on line 2 of the file)
      num_goal = int(file.readline())

      #get coordinates of goal on (num_goal) line next
      list_goal = []
      for i in range(num_goal):
        goal = file.readline().strip()
        goal_coordinates = tuple(map(to_center, goal.split()))
        list_goal.append((goal_coordinates[1],goal_coordinates[0]))

      #get map from file
      np_map = np.zeros((map_size, map_size), int)
      for line in range(map_size):
        np_map[line] = (file.readline()).strip().split()

      #get list obstacle from map
      list_obstacle = np.where(np_map == 1)
      list_obstacle = list(zip(list_obstacle[1], list_obstacle[0]))
      #change node goal to empty
      for goal in list_goal:
        np_map[int(goal[0])][int(goal[1])] = 0
      #save data about map
      self.obstacles = list_obstacle
      self.goals = list_goal
      self.environment = np_map.transpose()
      self.map_size = map_size

  def check_collision(self, f_X1, f_X2):
    x1 = int(f_X1[0])
    y1 = int(f_X1[1])
    x2 = int(f_X2[0])
    y2 = int(f_X2[1])
    a = x2 - x1
    b = y2 - y1
    t = 0
    toc_do = math.sqrt(abs(0.01/(a*b)))if a != 0 and b != 0 else 0.01

    if self.environment[x1][y1] == 1 or self.environment[x2][y2] == 1:
      return True
    while t < 1-toc_do:
      t += toc_do
      x = int(x1+a*t)
      y = int(y1+b*t)
      x_top_left = math.floor(x1-0.5+a*t)
      y_top_left = math.floor(y1-0.5+b*t)
      x_bottom_right = math.floor(x1+0.5+a*t)
      y_bottom_right = math.floor(y1+0.5+b*t)
      for obs in self.obstacles:
        if (x, y) == obs or \
            (x_top_left, y_top_left) == obs or \
                (x_bottom_right, y_bottom_right) == obs:
            return True
    return False

    # for obstacle in self.obstacles:
    #   obs_x = obstacle[0]
    #   obs_Y = obstacle[1]
    #   # if ((obs_x-x1)/(x2-x1) == (obs_Y-y1)/(y2-y1)):
    #   #   return True

    return False

# # # class MPA:
# # #     environment = []
# # #     list_dst = []
# # #     n = 0
# # #     x_min = 1
# # #     x_max = 5
# # #     d_min = 2
# # #     origin = 0.4

# # #     def __init__(self, filename):
# # #         fp = open(filename, "r")
# # #         f_n = int(fp.readline())
# # #         f_l = int(fp.readline())
# # #         l_dst = []
# # #         f_map = []
# # #         for f_i in range(f_l):
# # #             s = fp.readline()
# # #             s_l = s[:-1].split(" ")
# # #             l_dst.append([int(s_l[0]) + 0.5, int(s_l[1]) + 0.5])
# # #         print(l_dst)
# # #         for f_i in range(f_n):
# # #             s = fp.readline()
# # #             s_l = s[0:-2].split(" ")
# # #             f_map.append(list(map(int, s_l)))
# # #         for f_i in l_dst:
# # #             f_map[int(f_i[0])][int(f_i[1])] = 0
# # #         self.n = f_n
# # #         self.list_dst = list(l_dst)
# # #         self.environment = []
# # #         print(f_map)
# # #         print(self.list_dst)
# # #         for f_i in f_map:
# # #             self.environment.append(list(f_i))
# # #         fp.close()
# # run = True
# # while run:
# #     screen.fill('black')

# #     rect = pygame.Rect(50, 200, 100, 50)
# #     pygame.draw.rect(screen, 'red', rect)

# #     for i in range(DENSITY):
# #         mouse_pos = pygame.mouse.get_pos()
# #         pos_fin = (RADIUS * math.cos(2*math.pi / DENSITY * i) + mouse_pos[0], RADIUS * math.sin(2*math.pi / DENSITY * i) + mouse_pos[1])
# #         if rect.collidepoint(mouse_pos) == False:
# #             for extrem_1, extrem_2 in [(rect.bottomright, rect.topright), (rect.topright, rect.topleft), (rect.topleft, rect.bottomleft), (rect.bottomleft, rect.bottomright)]:
# #                 deno = (mouse_pos[0] - pos_fin[0]) * (extrem_1[1] - extrem_2[1]) - (mouse_pos[1] - pos_fin[1]) * (extrem_1[0] - extrem_2[0])
# #                 if deno != 0:
# #                     param_1 = ((extrem_2[0] - mouse_pos[0]) * (mouse_pos[1] - pos_fin[1]) - (extrem_2[1] - mouse_pos[1]) * (mouse_pos[0] - pos_fin[0]))/deno
# #                     param_2 = ((extrem_2[0] - mouse_pos[0]) * (extrem_2[1] - extrem_1[1]) - (extrem_2[1] - mouse_pos[1]) * (extrem_2[0] - extrem_1[0]))/deno
# #                     if 0 <= param_1 <= 1 and 0 <= param_2 <= 1:
# #                         p_x = mouse_pos[0] + param_2 * (pos_fin[0] - mouse_pos[0])
# #                         p_y = mouse_pos[1] + param_2 * (pos_fin[1] - mouse_pos[1])
# #                         pos_fin = (p_x, p_y)
# #             pygame.draw.aaline(screen, 'white', mouse_pos, pos_fin)

# #     for event in pygame.event.get():
# #         if event.type == pygame.QUIT:
# #             run = False
# #
# # # pygame.mouse.set_visible(False)
# # DENSITY = 500
# # RADIUS = 1000


mpa_obj = MPA('Test/map15_3.txt')
print(map_size := mpa_obj.map_size)
print(goals := mpa_obj.goals)
mpa_obj.obstacles.sort()
print(obstacles := mpa_obj.obstacles)
print(environment := mpa_obj.environment.transpose())
start_time = time.time()
for i in range(10000000):
  mpa_obj.check_collision(goals[0], goals[1])

# import pygame, math, sys
list_point = []
start = mpa_obj.goals[0]
end = mpa_obj.goals[1]
# start = (1,6)
# end = (2,6
a = end[0]-start[0]
b = end[1]-start[1]
t = 0

toc_do = math.sqrt(abs(0.01/(a*b)))if a != 0 and b != 0 else 0.01
x0 = start[0]
y0 = start[1]
while t < 1-toc_do:
  t += toc_do
  x = int(x0+a*t)
  y = int(y0+b*t)
  x1 = math.floor(x0-0.5+a*t)
  y1 = math.floor(y0-0.5+b*t)
  x2 = math.floor(x0+0.5+a*t)
  y2 = math.floor(y0+0.5+b*t)
  list_point.append((x, y))
  list_point.append((x1, y1))
  list_point.append((x2, y2))
list_point = list(set(list_point))
list_point.sort()
list_point = [point for point in list_point if point in obstacles]
print(list_point)
end_time = time.time()
print(end_time - start_time)
pygame.init()
clock = pygame.time.Clock()
screen_width = 800
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption('Rays')
obstacles_Rect = []
# for obs in obstacles:
# #   # obs_dict = {}
# #   # obs_dict['coordinates']=obs
# #   # obs.__dict__
# #   # obs_dict['Rect']=pygame.Rect(100,100,100,100)
#   rect= pygame.Rect(50,50,50,50)
#   rect.x,rect.y=obs[0],obs[1]
#   obstacles_Rect.append(rect)
# x_speed = (int(x0)+a*t)*puvt
# y_speed = (int(y0)+b*t)*puvt

run = True
pad = 800/30
puvt = 800/15  # length 1 unit vector
width = height = 800/15-0.4
t = 0


def straight_line(x_speed, y_speed):
  moving_rect.x += x_speed
  moving_rect.y += y_speed
  for obs in obstacles_Rect:
    if moving_rect.colliderect(obs):
      pygame.draw.rect(screen, (0, 0, 255), obs)
  pygame.draw.rect(screen, (0, 255, 0), moving_rect)
  pygame.display.update()
  time.sleep(0.1)

x_speed = a
y_speed = b

t = 0
obstacles_collision = []
for point in list_point:
  rect = pygame.Rect(point[0]*puvt, point[1]*puvt, width, height)
  obstacles_collision.append(rect)
moving_rect = pygame.Rect(int(x0)*puvt, int(y0)*puvt, width, height)
line_width = math.floor(math.sqrt(width*height))

while run:
  screen.fill((30, 30, 30))
  for obs in obstacles:
    rect = pygame.Rect(int(obs[0]*puvt), int(obs[1]*puvt), width, height)
    obstacles_Rect.append(rect)
    pygame.draw.rect(screen, (255, 0, 0), rect)
  x1 = int(end[0])*puvt
  y1 = int(end[1])*puvt
  rect = pygame.Rect(int(x0)*puvt, int(y0)*puvt, width, height)
  pygame.draw.rect(screen, (0, 126, 126), rect)
  rect = pygame.Rect(int(end[0])*puvt, int(end[1])*puvt, width, height)
  pygame.draw.rect(screen, (0, 126, 126), rect)
  pygame.draw.rect(screen, (0, 255, 0), moving_rect)
  # while t < 1:
  if t<=1:
    t += 1/(abs(a*b))
    straight_line(x_speed, y_speed)
    # pygame.display.update()
  else:
    for rect in obstacles_collision:
      pygame.draw.rect(screen, (0, 0, 254), rect)

    pygame.draw.line(screen, (0, 255, 0), (x0*puvt, int(y0)*puvt),
                    (end[0]*puvt, int(end[1])*puvt), int(width))
    pygame.draw.line(screen, (0, 255, 0), (x0*puvt, int(y0)*puvt+puvt),
                    (end[0]*puvt, int(end[1])*puvt+puvt), int(width))


  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      run = False
    pygame.display.update()
pygame.quit()
