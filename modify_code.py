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
        np_map[int(goal[1])][int(goal[0])] = 0
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

    if self.environment[x1][y1] == 1 or self.environment[x2][y2] == 1:
      return True
    for obstacle in self.obstacles:
      obs_x = obstacle[0]
      obs_Y = obstacle[1]
      if ((obs_x-x1)/(x2-x1)==(obs_Y-y1)/(y2-y1)):
        return True

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
print(obstacles := mpa_obj.obstacles)
environment = mpa_obj.environment
print(environment.T)
# start = time.time()
# for i in range(10000):
#   mpa_obj.check_collision((1,1),(0,5))
# end = time.time()
# print(end - start)
import pygame, math, sys
def point_to_square (point:tuple):
  x = point[0]
  y = point[1]
  return (x+0.5,y+0.5), (x-0.5,y-0.5), (x-0.5,y+0.5), (x+0.5,y-0.5)
list_point = []
start = mpa_obj.goals[0]
end = mpa_obj.goals[2]
# start = (1,7)
# end = (0,5)
print(start,end)
a = end[0]-start[0]
b = end[1]-start[1]
print(a,b)
t=0
toc_do = math.sqrt(abs(0.01/(a*b))) if a!=0 and b!=0 else 0.01
print(toc_do)
x0=start[0]
y0=start[1]
start_time = time.time()
while t < 1:
  t += toc_do
  x_top_left = math.floor(x0-0.5+a*t)
  y_top_left = math.floor(y0-0.5+b*t)
  x_bottom_right = math.floor(x0+0.5+a*t)
  y_bottom_right = math.floor(y0+0.5+b*t)
  x_top_right = math.floor(x0+0.5+a*t)
  y_top_right = math.floor(y0-0.5+b*t)
  x_bottom_left = math.floor(x0-0.5+a*t)
  y_bottom_left = math.floor(y0+0.5+b*t)
  if environment[x_top_left][y_top_left] == 1:
    list_point.append((x_top_left, y_top_left))
  if environment[x_bottom_right][y_bottom_right] == 1:
    list_point.append((x_bottom_right, y_bottom_right))
  if environment[x_top_right][y_top_right] == 1:
    list_point.append((x_top_right, y_top_right))
  if environment[x_bottom_left][y_bottom_left] == 1:
    list_point.append((x_bottom_left, y_bottom_left))

list_point = list(set(list_point))
list_point.sort()
end_time = time.time()
print(list_point)
import pygame
print(start_time-end_time)
pygame.init()
clock = pygame.time.Clock()
screen_width = 800
screen_height = 800
screen = pygame.display.set_mode((screen_width, screen_height))
# pygame.display.set_caption('Rays')
# moving_rect = pygame.rect(100,100,100,100)

# x_speed, y_speed = 5, 4
start_square = list(point_to_square(start))
end_square = list(point_to_square(end))
obstacles_Rect = []
obstacles_colliderect = []
run = True
puvt = 800/15 # length 1 unit vector
width = height = 800/15-0.4
t=0
for point in list_point:
  rect= pygame.Rect(point[0]*puvt,point[1]*puvt,width,height)
  obstacles_colliderect.append(rect)
def straight_line(x_speed,y_speed):
  moving_rect.x+= x_speed
  moving_rect.y+= y_speed
  for obs in obstacles_colliderect:
    if moving_rect.colliderect(obs):
      pygame.draw.rect(screen,(0,0,255),obs)
  pygame.draw.rect(screen, (0,255,0), moving_rect)
  time.sleep(0.1)
  pygame.display.update()
for obs in obstacles:
  rect= pygame.Rect(obs[0]*puvt,obs[1]*puvt,width,height)
  obstacles_Rect.append(rect)

t=0
x_speed = a
y_speed = b

# start = (1,6)
# end = (12,2)
x1 = end[0]
y1 = end[1]
moving_rect = pygame.Rect(int(x0)*puvt,int(y0)*puvt,width,height)
start_rect = pygame.Rect(int(x0)*puvt,int(y0)*puvt,width,height)
end_rect = pygame.Rect(int(x1)*puvt,int(y1)*puvt,width,height)
while run:
  screen.fill((30,30,30))
  for rect in obstacles_Rect:
    pygame.draw.rect(screen, (255,0,0), rect)
  pygame.draw.rect(screen, (0,126,126), start_rect,2)
  pygame.draw.rect(screen, (0,126,126), end_rect,2)


  if moving_rect.x<int(x1)*puvt and moving_rect.y<=int(y1)*puvt:
    straight_line(x_speed,y_speed)
  else:
    for rect in obstacles_colliderect:
      pygame.draw.rect(screen, (0, 0, 254), rect)
    for i in range(4):
      pygame.draw.line(screen, (0, 255, 0), 
                       (start_square[i][0]*puvt, start_square[i][1]*puvt),
                       (end_square[i][0]*puvt, end_square[i][1]*puvt),
                       )
    # pygame.draw.line(screen, (0, 255, 0), (x0*puvt, y0*puvt),
    #                 (x1*puvt, y1*puvt), 1)
    # pygame.draw.line(screen, (0, 255, 0), (x0*puvt, y0*puvt),
    #                 (x1*puvt, y1*puvt), 1)

    # pygame.draw.line(screen, (0, 255, 0), (x0*puvt, int(y0)*puvt),
    #                 (end[0]*puvt, int(end[1])*puvt), int(width))
    # pygame.draw.line(screen, (0, 255, 0), (x0*puvt, int(y0)*puvt+puvt),
    #                 (end[0]*puvt, int(end[1])*puvt+puvt), int(width))
    pygame.display.flip()

  for event in pygame.event.get():
    if event.type == pygame.QUIT:
      run = False
    pygame.display.update()
pygame.quit()
