import pygame
import math
from typing import List, Union
import numpy as np
import random as rd
import time
from typing import Callable, TypeVar
from scipy.stats import levy

def to_center(var: str):
  return int(var)+0.5


def distance(f_X1, f_X2):
  return math.sqrt((f_X1[0] - f_X2[0])**2 + (f_X1[1] - f_X2[1]) ** 2)


class Node:
  def __init__(self, x: int, y: int) -> None:
    '''a node have coordinates (x,y)
    is a square as :
      (x,y)___________(x_r,y_r)
        |                 |
        |    x_c,y_c      |
    (x_b,y_b)________(x_br,y_br)
    '''

    self.x = x
    self.y = y
    self.three_corners = [(x+1, y), (x, y+1), (x+1, y+1)]
    self.center = (x+0.5, y+0.5)
    self.a = None
    self.b = None
    # self.delta_t = None

  def linear_equations_to(self, x: int, y: int) -> None:
    '''linear_equations AB:
      |x = xA+a*t (a = xB - xA)
      |y = yA+b*t (b = yB - yA)

    if t = 1 : x = xB, y = yB => AB ~ (0<delta_t<1)  
    '''     
    self.a = x-self.x
    self.b = y-self.y
    # self.delta_t = 1/(self.a*self.b) if self.a !=0 and self.b != 0 else 0.1

  def check_line_collision(self, environment) -> bool:
    '''check '''
    check = False
    t = 0
    while t < 1-0.05:
      t += 0.05
      if self.a != 0 and self.b != 0:
        location_x = math.floor(self.x + self.a*t)
        location_y = math.floor(self.y + self.b*t)
        if environment[location_x][location_y] == 1:
          check = True
          break
        for x, y in self.three_corners:
          x = math.floor(x + self.a*t)
          y = math.floor(y + self.b*t)
          if environment[x][y] == 1:
            check = True
            break
      else:
        location_x = math.floor(self.x + self.a*t)
        location_y = math.floor(self.y + self.b*t)
        if environment[location_x][location_y] == 1:
          check = True
          break

      if check == True:
        break
    return check

  def collision_coordinates(self, environment: np, end) -> list[tuple]:
    collision = []
    t = 0
    while t<1-0.05:
      t += 0.05
      if self.a!=0 and self.b!=0:
        location_x =  math.floor(self.x + self.a*t)
        location_y =  math.floor(self.y + self.b*t)

        if environment[location_x][location_y] == 1:
          collision.append((location_x,location_y))
        for x, y in self.three_corners:
            x = math.floor(x + self.a*t)
            y = math.floor(y + self.b*t)
            if environment[x][y] == 1:
              collision.append((x,y))
      else:
        location_x =  math.floor(self.x + self.a*t)
        location_y =  math.floor(self.y + self.b*t)
        if environment[location_x][location_y] == 1:
          collision.append((location_x,location_y))

    collision = list(set(collision))
    collision.sort()
    return collision

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

      #get coordinates of goal on (num_goal) line next and reverse
      list_goal = []
      for i in range(num_goal):
        goal = file.readline().strip()
<<<<<<< HEAD
        goal_coordinates = tuple(map(to_center, goal.split()))
        list_goal.append((goal_coordinates[1],goal_coordinates[0]))
=======
        goal_coordinates = list(map(int, goal.split()))
        list_goal.append((goal_coordinates[1], goal_coordinates[0]))
>>>>>>> outsider

      #get map from file
      np_map = np.zeros((map_size, map_size), int)
      for line in range(map_size):
        np_map[line] = (file.readline()).strip().split()

<<<<<<< HEAD
      #get list obstacle from map
      list_obstacle = np.where(np_map == 1)
      list_obstacle = list(zip(list_obstacle[1], list_obstacle[0]))
      #change node goal to empty
      for goal in list_goal:
        np_map[int(goal[1])][int(goal[0])] = 0
=======
      #convert to window coordinate system
      np_map = np_map.transpose()

      #get list obstacle from map
      list_obstacle = []
      list_empty = []
      for i in range(map_size):
        for j in range(map_size):
          if np_map[i][j] == 1:
            list_obstacle.append((i, j))
          else:
            list_empty.append((i, j))
                   
      #change node goal to empty
      for goal in list_goal:
        np_map[int(goal[0])][int(goal[1])] = 0

>>>>>>> outsider
      #save data about map
      self.obstacles = list_obstacle
      self.empty = list_empty
      self.goals = list_goal
      self.environment = np_map.transpose()
      self.map_size = map_size
    
  # def collision(self, x1, y1, x2, y2):

<<<<<<< HEAD
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
=======
  def check_collision(self, st_point: Union[tuple[int, int], list[int, int]], en_point: Union[tuple[int, int], list[int, int]]) -> bool:
    x1 = st_point[0]
    y1 = st_point[1]
    x2 = en_point[0]
    y2 = en_point[1]
    ''''''
    if x1 >= self.map_size or y1 >= self.map_size or x2 >= self.map_size or y2 >= self.map_size:
      return True
    if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0:
      return True
    if self.environment[x1][y1] == 1 or self.environment[x2][y2] == 1:
      return True
    
    #convert to Node
    node = Node(x1, y1)
    node.linear_equations_to(x2, y2)
    if node.check_line_collision(self.environment):
      return True
>>>>>>> outsider

    return False

  def point_collisions(self, x1: int, y1: int, x2: int, y2: int):
    #convert to Node
    node = Node(x1, y1)
    node.linear_equations_to(x2, y2)
    return node.collision_coordinates(self.environment, (x2, y2))
    
  def check(self, f_X1, f_X2):
    f_x = f_X1[0]
    f_y = f_X1[1]
    pre_x = f_X2[0]
    pre_y = f_X2[1]
    a_y = (f_y - pre_y) / (f_x - pre_x) if (f_x - pre_x) != 0 else 0
    a_x = (f_x - pre_x) / (f_y - pre_y) if (f_y - pre_y) != 0 else 0

    if f_x < self.origin:
        f_a = a_y
        f_b = f_y - f_a * f_x
        f_x = self.origin
        f_y = f_a * f_x + f_b
    elif f_x >= self.map_size:
        f_a = a_y
        f_b = f_y - f_a * f_x
        f_x = self.map_size - self.origin
        f_y = f_a * f_x + f_b
    elif f_y < self.origin:
        f_a = a_x
        f_b = f_x - f_a * f_y
        f_y = self.origin
        f_x = f_a * f_y + f_b
    elif f_y >= self.map_size:
        f_a = a_x
        f_b = f_x - f_a * f_y
        f_y = self.map_size - self.origin
        f_x = f_a * f_y + f_b
    if f_x >= self.map_size - self.origin:
        f_x = self.map_size - self.origin
    if f_y >= self.map_size - self.origin:
        f_y = self.map_size - self.origin
    if f_x <= self.origin:
        f_x = self.origin
    if f_y <= self.origin:
        f_y = self.origin

    return round(f_x, 2), round(f_y, 2)

  def calculator(self, f_sol, f_st, f_dst):
      s = 0
      is_dst = False
      pre_x = f_st
      for f_x in f_sol:
          if self.check_collision(pre_x, f_x):
              return 1, s
          s += distance(pre_x, f_x)
          if not self.check_collision(f_x, f_dst):
              s += distance(f_x, f_dst)
              is_dst = True
              break
          pre_x = f_x
      if not is_dst:
          if self.check_collision(f_sol[-1], f_dst):
              return 1, s
          s += distance(f_sol[-1], f_dst)
      return 0, s

<<<<<<< HEAD
# #     for event in pygame.event.get():
# #         if event.type == pygame.QUIT:
# #             run = False
# #
# # # pygame.mouse.set_visible(False)
# # DENSITY = 500
# # RADIUS = 1000
=======
  def normal_search(self, f_sol):
      f_ns_sol = []
      for f_i in f_sol:
          f_x = rd.randint(-2, 2)
          f_y = rd.randint(-2, 2)
          # print(f_x,f_y)
          f_ns_sol.append([f_i[0] + f_x, f_i[1] + f_y])
      return f_ns_sol
>>>>>>> outsider

  def evolution(self, f_father, f_mother, f_st, f_dst):
      f_child = [f_st]
      for f_i in range(len(f_father)):
          if self.check_collision(f_child[-1], f_father[f_i]):
              f_child.append(f_mother[f_i])
          elif self.check_collision(f_child[-1], f_mother[f_i]):
              f_child.append(f_father[f_i])
          else:
              if distance(f_child[-1], f_father[f_i]) < distance(f_child[-1], f_mother[f_i]):
                  f_child.append(f_father[f_i])
              else:
                  f_child.append(f_mother[f_i])
          if not self.check_collision(f_child[-1], f_dst):
              break
      f_child.remove(f_st)
      for f_i in range(len(f_father) - len(f_child)):
          f_child.append(f_dst)
      return f_child

<<<<<<< HEAD
mpa_obj = MPA('Test/map15_3.txt')
print(map_size := mpa_obj.map_size)
print(goals := mpa_obj.goals)
mpa_obj.obstacles.sort()
print(obstacles := mpa_obj.obstacles)
print(environment := mpa_obj.environment.transpose())
start_time = time.time()
# for i in range(10000000):
#   mpa_obj.check_collision(goals[0], goals[1])

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

=======
  def way(self, st, dst):
    if not self.check_collision(st, dst):
      return distance(st, dst), [st, dst]
    n_child = self.map_size
    min_s = math.inf
    prey = []
    best_prey = []
    old_s = []
    max_d = 0

    # for index in range(10):
    i = 5
    limit_rd = (i+1)**2-1
    for index in range(n_child):
      origin_sol = [st]
      temp_map = self.environment.copy()
      temp_map[st[0]][st[1]] = 3
      limit = 0
      limit_loop = 0

      while self.check_collision(origin_sol[-1], dst):
        # print(origin_sol[-1])
        limit_loop += 1
        while True:
              limit += 1
              x = origin_sol[-1][0] + rd.randint(-i, i)
              y = origin_sol[-1][1] + rd.randint(-i, i)
              if limit > limit_rd:
                temp_map[origin_sol[-1][0]][origin_sol[-1][1]] = 0
                origin_sol.pop()
                limit = 0
                break
              if self.map_size <= x or x < 0 or self.map_size <= y or y < 0:
                continue
              if temp_map[x][y] == 3 or temp_map[x][y] == 1:
                  continue
              if not self.check_collision([x, y], origin_sol[-1]):
                origin_sol.append([x, y])
                temp_map[x][y] = 3
                limit = 0
                break
        if len(origin_sol) == 0 or limit_loop > 50:
          origin_sol = [st]
          temp_map = self.environment.copy()
          temp_map[origin_sol[-1][0]][origin_sol[-1][0]] = 3
          limit = 0
          limit_loop = 0
      reduce_sol = [st]
      while self.check_collision(reduce_sol[-1], dst):
          i = len(origin_sol) - 1
          while self.check_collision(reduce_sol[-1], origin_sol[i]):
              i -= 1
          reduce_sol.append(origin_sol[i])
      pre_sol = st
      reduce_sol.pop(0)
      reduce_sol.append(dst)
      iPrey = []
      for sol in reduce_sol:
          loop = int(distance(pre_sol, sol) / self.d_min)
          for i in range(1, loop):
              x = round(pre_sol[0] + (sol[0] - pre_sol[0]) * i / loop)
              y = round(pre_sol[1] + (sol[1] - pre_sol[1]) * i / loop)
              if self.check_collision(pre_sol, [x, y]) or self.check_collision([x, y], sol):
                  continue
              iPrey.append([x, y])
          iPrey.append(sol)
          pre_sol = sol
      iPrey.pop()
      sol_d = len(iPrey)
      if max_d < sol_d:
          max_d = sol_d
      v, dis_prey = self.calculator(iPrey, st, dst)
      old_s.append(dis_prey)
      prey.append(iPrey)
      if min_s > dis_prey:
          min_s = dis_prey
          best_prey = list(iPrey)
    for i in prey:
      i.extend([dst for _ in range(max_d-len(i))])
    a_prey = np.array(prey)
    
    best_prey.extend([dst for _ in range(max_d-len(best_prey))])
    # for i in a_prey:
    #   print(i.T)
    # return a_prey
    d = max_d
    X_min = np.array([[self.x_min, self.x_min] for _ in range(d)])
    X_max = np.array([[self.x_max, self.x_max] for _ in range(d)])
    prey = np.array(prey)
    old_prey = np.array(prey)
    loop = 100
    levy.a = 1.5
    levy.b = 1
    P = np.array([[0.5, 0.5] for _ in range(d)])

    for index in range(loop):
        for i in range(n_child):
            new_v, new_dis = self.calculator(prey[i], st, dst)
            if new_v == 0:
                if new_dis < old_s[i]:
                    old_s[i] = new_dis
                else:
                    prey[i] = np.array(old_prey[i])
                if new_dis < min_s:
                    min_s = new_dis
                    print(min_s)
                    best_prey = np.array(prey[i])
            else:
                prey[i] = np.array(old_prey[i])
        Elite = np.array([list(best_prey) for _ in range(n_child)])
        old_prey = np.array(prey)

        cf = math.pow(1 - index / loop, 2 * index / loop)
        CF = np.array([[cf, cf] for _ in range(d)])
        for i in range(n_child):
            pre_prey = []
            rBx = np.random.normal(0, 1, d)
            rBy = np.random.normal(0, 1, d)
            Rb = np.array([[rBx[j], rBy[j]] for j in range(d)])
            rLx = levy.rvs(0, 1, d)
            rLy = levy.rvs(0, 1, d)
            Rl = np.array([[rLx[j], rLy[j]] for j in range(d)])
            rx = np.random.uniform(0, 1, d)
            ry = np.random.uniform(0, 1, d)
            R = np.array([[rx[j], ry[j]] for j in range(d)])
            if index < loop / 3:
                step = Rb * (Elite[i] - Rb * prey[i])
                pre_prey = prey[i] + P * R * step
            elif loop / 3 <= index < 2 * loop / 3:
                if i < self.map_size / 2:
                    step = Rl * (Elite[i] - Rl * prey[i])
                    pre_prey = prey[i] + P * R * step
                else:
                    step = Rb * (Rb * Elite[i] - prey[i])
                    pre_prey = Elite[i] + P * CF * step
            elif index >= 2 * loop / 3:
                step = Rl * (Rl * Elite[i] - prey[i])
                pre_prey = Elite[i] + P * CF * step
            for j in range(d):
                pre_prey[j] = self.check(pre_prey[j], prey[i][j])
            prey[i] = np.array(pre_prey)

        for i in range(n_child):
            new_v, new_dis = self.calculator(prey[i], st, dst)
            if new_v == 0:
                if new_dis < old_s[i]:
                    old_s[i] = new_dis
                else:
                    prey[i] = np.array(old_prey[i])
                if new_dis < min_s:
                    min_s = new_dis
                    best_prey = np.array(prey[i])
                    print(min_s)
            else:
                prey[i] = np.array(old_prey[i])
>>>>>>> outsider

            child = self.normal_search(prey[i])
            new_v, new_dis = self.calculator(child, st, dst)
            if new_v == 0:
                if new_dis < old_s[i]:
                    old_s[i] = new_dis
                    prey[i] = np.array(child)
                if new_dis < min_s:
                    min_s = new_dis
                    best_prey = np.array(child)
                    print(min_s)

            ga_child = self.evolution(child, prey[i], st, dst)
            new_v, new_dis = self.calculator(ga_child, st, dst)
            if new_v == 0:
                if new_dis < old_s[i]:
                    old_s[i] = new_dis
                    prey[i] = np.array(ga_child)
                if new_dis < min_s:
                    min_s = new_dis
                    best_prey = np.array(ga_child)
                    print(min_s)

        old_prey = np.array(prey)

        r_x = rd.random()
        fad = 0.2 * (1 - r_x) + r_x
        Fad = np.array([[fad, fad] for _ in range(d)])
        ux = np.random.randint(0, 1, d)
        uy = np.random.randint(0, 1, d)
        U = np.array([[ux[j], uy[j]] for j in range(d)])

        for i in range(n_child):
            rx = np.random.uniform(0, 1, d)
            ry = np.random.uniform(0, 1, d)
            R = np.array([[rx[j], ry[j]] for j in range(d)])
            pre_prey = prey[i]
            if r_x < 0.2:
                pre_prey = pre_prey + CF * (X_min + R * (X_max - X_min)) * U
            else:
                pre_prey = pre_prey + Fad * (prey[rd.randint(0, n_child - 1)] - prey[rd.randint(0, n_child - 1)])
            for j in range(d):
                pre_prey[j] = self.check(pre_prey[j], prey[i][j])
            prey[i] = pre_prey

    best_reduce = []
    for i in best_prey:
        best_reduce.append(i)
        if not self.check_collision(i, dst):
            break

    final_sol = list([list(st)])
    while self.check_collision(final_sol[-1], dst):
        i = len(best_reduce) - 1
        while self.check_collision(final_sol[-1], best_reduce[i]):
            i -= 1
        final_sol.append(list(best_reduce[i]))
    final_sol.append(list(dst))
    v_sol, dis_sol = self.calculator(final_sol, st, dst)
    return dis_sol, final_sol, a_prey
        # if a_prey[i][j] !=prey[i][j]:
        #   print('another')
    # for i in range(10):
    #   a_prey[i]=prey[i]
    #   a_prey[i] = prey[i]
    # print(a_prey)
    # prey = np.array(prey)
    # print(prey)
# print(distance((1,1),(2,2)))
# mpa_obj = MPA('Test/map15_3.txt')
# print(map_size := mpa_obj.map_size)
# print(goals:= mpa_obj.goals)
# print(obstacles := mpa_obj.obstacles)
# environment = mpa_obj.environment
# print(environment.T)

# def solve(filename, label, isSolveAll=False):
#     now = datetime.datetime.now()
#     start_time = datetime.datetime.timestamp(now)
#     mpa_sol = MPA(filename)
#     file_name_sol = "Solutions/map" + str(mpa_sol.map_size) + "_" + str(len(mpa_sol.goals)) + "_sol.txt"
#     os.makedirs(os.path.dirname(file_name_sol), exist_ok=True)
#     shutil.copyfile(filename, file_name_sol)
#     # if isSolveAll:
#     #     case_map = filename.get()[13:14]
#     #     file_name_sol = "Solutions\case" + case_map + "_map" + str(mpa_sol.n) + "_" + str(len(mpa_sol.list_dst)) + "_sol.txt"
#     now = datetime.datetime.now()
#     end_time = datetime.datetime.timestamp(now)
#     print(end_time - start_time)

# solve('Test/map15_3.txt','')
