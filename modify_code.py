import math
import numpy as np
import time

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

  def linear_equations_to(self, x:int,y:int) -> None:
    '''linear_equations AB:
      |x = xA+a*t (a = xB - xA)
      |y = yA+b*t (b = yB - yA)

    if t = 1 : x = xB, y = yB => AB ~ (0<delta_t<1)  
    '''     
    self.a = x-self.x
    self.b = y-self.y
    print(self.a,self.b)
    # self.delta_t = 1/(self.a*self.b) if self.a !=0 and self.b != 0 else 0.1

  def check_line_collision(self, environment) -> bool:
    '''check '''
    check = False
    t = 0
    print(self.a,self.b)
    while t < 1-0.05:
      t += 0.1
      for x, y in self.three_corners:
          x = math.floor(x + self.a*t)
          y = math.floor(y + self.b*t)
          if environment[x][y] == 1:
            check = True
            print('found(',x,y,')')
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
      list_obstacle = np.where(np_map == 1)
      list_obstacle = list(zip(list_obstacle[0], list_obstacle[1]))

      #change node goal to empty
      for goal in list_goal:
        np_map[int(goal[0])][int(goal[1])] = 0

      #save data about map
      self.obstacles = list_obstacle
      self.goals = list_goal
      self.environment = np_map
      self.map_size = map_size
    
  # def collision(self, x1, y1, x2, y2):

  def check_collision(self, x1:int, y1:int, x2:int, y2:int):
    ''''''
    if self.environment[x1][y1] == 1 or self.environment[x2][y2] == 1:
      return True
    
    #convert to Node
    node = Node(x1, y1)
    node.linear_equations_to(x2, y2)
    # print(node.check_line_collision(self.environment))
    if node.check_line_collision(self.environment):
      return True

    return False

  def point_collisions(self, x1:int, y1:int, x2:int, y2:int):
    #convert to Node
    node = Node(x1, y1)
    node.linear_equations_to(x2, y2)
    # print(node.check_line_collision(self.environment))
    node.collision_coordinates(self.environment,(x2,y2))
    
  


# print(distance((1,1),(2,2)))
# mpa_obj = MPA('Test/map15_3.txt')
# print(map_size := mpa_obj.map_size)
# print(goals:= mpa_obj.goals)
# print(obstacles := mpa_obj.obstacles)
# environment = mpa_obj.environment
# print(environment.T)

# print(mpa_obj.check_collision(0,0,12,0))
# while t < 1-2*toc_do:
#     t += toc_do
#     if a != 0 and b != 0:
#     #     if a!=1 and b!=1: 
#       x_top_left = math.floor(x0-0.5+a*t)
#       y_top_left = math.floor(y0-0.5+b*t)
#       x_bottom_right = math.floor(x0+0.5+a*t)
#       y_bottom_right = math.floor(y0+0.5+b*t)
#       x_top_right = math.floor(x0+0.5+a*t)
#       y_top_right = math.floor(y0-0.5+b*t)
#       x_bottom_left = math.floor(x0-0.5+a*t)
#       y_bottom_left = math.floor(y0+0.5+b*t)
#       if environment[x_top_left][y_top_left] == 1:
#           list_point.append((x_top_left, y_top_left))
#       if environment[x_bottom_right][y_bottom_right] == 1:
#           list_point.append((x_bottom_right, y_bottom_right))
#       if environment[x_top_right][y_top_right] == 1:
#           list_point.append((x_top_right, y_top_right))
#       if environment[x_bottom_left][y_bottom_left] == 1:
#           list_point.append((x_bottom_left, y_bottom_left))
# import datetime
# import shutil
# import sys 
# import os
# sys.path.append(os.path.abspath("/home/miichi/tmpcode/DA/MPA_Path_Planning"))
# from modify_code import MPA
# def solve(filename, label, isSolveAll=False):
#     now = datetime.datetime.now()
#     start_time = datetime.datetime.timestamp(now)
#     mpa_sol = MPA(filename)
#     filenamesol = "Solutions/map" + str(mpa_sol.map_size) + "_" + str(len(mpa_sol.goals)) + "_sol.txt"
#     os.makedirs(os.path.dirname(filenamesol), exist_ok=True)
#     shutil.copyfile(filename, filenamesol)
#     # if isSolveAll:
#     #     case_map = filename.get()[13:14]
#     #     filenamesol = "Solutions\case" + case_map + "_map" + str(mpa_sol.n) + "_" + str(len(mpa_sol.list_dst)) + "_sol.txt"
#     now = datetime.datetime.now()
#     end_time = datetime.datetime.timestamp(now)
#     print(end_time - start_time)

# solve('Test/map15_3.txt','')