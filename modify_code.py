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
        goal_coordinates = list(map(int, goal.split()))
        list_goal.append((goal_coordinates[1], goal_coordinates[0]))

      #get map from file
      np_map = np.zeros((map_size, map_size), int)
      for line in range(map_size):
        np_map[line] = (file.readline()).strip().split()

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

      #save data about map
      self.obstacles = list_obstacle
      self.empty = list_empty
      self.goals = list_goal
      self.environment = np_map
      self.map_size = map_size
    
  # def collision(self, x1, y1, x2, y2):

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

    return False

  def random_space(self, s: int, center: list, temp_map: np):
    random_space = []
    for i in range(-s, s+1):
        for j in range(-s, s+1):
            if i == j == 0:
                continue
            x = center[0]+i
            y = center[1]+j
            if self.map_size <= x or x < 0 or self.map_size <= y or y < 0:
              continue
            if temp_map[x][y] != 0:
              continue
            random_space.append([x, y])
    return random_space


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

  def normal_search(self, f_sol):
      f_ns_sol = []
      for f_i in f_sol:
          f_x = rd.randint(-1, 1)
          f_y = rd.randint(-1, 1)
          f_ns_sol.append([f_i[0] + f_x, f_i[1] + f_y])
      return f_ns_sol

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

  def way(self, st, dst):
    if not self.check_collision(st, dst):
      return distance(st, dst), [st, dst]
    n_child = self.map_size
    min_s = math.inf
    prey = []
    best_prey = []
    old_s = []
    max_d = 0

    for index in range(n_child):
      origin_sol = [st]
      temp_map = self.environment.copy()
      temp_map[st[0]][st[1]] = 3

      while self.check_collision(origin_sol[-1], dst):
        rd_values = self.random_space(3,origin_sol[-1],temp_map)
        while True:
          if len(rd_values) == 0:
                origin_sol.pop()
                break
          selected = rd.choice(rd_values)
          rd_values.remove(selected)

          if not self.check_collision(selected, origin_sol[-1]):
                origin_sol.append(selected)
                temp_map[selected[0]][selected[1]] = 3
                break
        if len(origin_sol) == 0:
          origin_sol = [st]
          temp_map = self.environment.copy()
          temp_map[origin_sol[-1][0]][origin_sol[-1][0]] = 3
      
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
    d = max_d
    X_min = np.array([[self.x_min, self.x_min] for _ in range(d)])
    X_max = np.array([[self.x_max, self.x_max] for _ in range(d)])
    prey = np.array(prey)
    old_prey = np.array(prey)
    loop = 50
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
