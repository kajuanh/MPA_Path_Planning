import numpy as np
# # #
# # from tkinter import *
# # def motion(event):
# #   print("Mouse position: (%s %s)" % (event.x, event.y))
# #   return

# # master = Tk()
# # whatever_you_do = "Whatever you do will be insignificant, but it is very important that you do it.\n(Mahatma Gandhi)"
# # msg = Message(master, text = whatever_you_do)
# # msg.config(bg='lightgreen', font=('times', 24, 'italic'))
# # msg.bind('<Motion>',motion)
# # msg.pack()
# # master.mainloop()

# # with open('Test/map15_3.txt', "r") as file:
# #   #area = map_size*map_size (map_size located on line 1 of the file)
# #   map_size = int(file.readline())
# #   #number of goal (num_goal located on line 2 of the file)
# #   num_goal = int(file.readline())
# #   #get coordinates of goal on (num_goal) line next
# #   list_goal = []
# #   for i in range(num_goal):
# #     goal = file.readline().strip()
# #     goal_coordinates = tuple(map(to_center, goal.split()))
# #     list_goal.append(goal_coordinates)

# #   #get map from file
# #   np_map = np.zeros((map_size, map_size))
# #   for line in range(map_size):
# #     np_map[line] = (file.readline()).strip().split()
# #   l_dst = np.where(np_map == 1)
# #   l_dst = list(zip(l_dst[0], l_dst[1]))
# #   l_goal = np.where(np_map == 1)
# #   l_goal = list(zip(l_goal[0], l_goal[1]))


def to_center(var: str):
  return int(var)+0.5


class MPA:
  environment = None
  map_size = None
  list_goal = []
  list_obstacle = []
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
        list_goal.append(goal_coordinates)

      #get map from file
      np_map = np.zeros((map_size, map_size), int)
      for line in range(map_size):
        np_map[line] = (file.readline()).strip().split()

      #get list obstacle from map
      list_obstacle = np.where(np_map == 1)
      list_obstacle = list(zip(list_obstacle[0], list_obstacle[1]))

      #change node goal to empty
      for goal in list_goal:
        np_map[int(goal[0])][int(goal[1])] = 0
      #save data about map
      self.list_obstacle = list_obstacle
      self.list_goal = list_goal
      self.environment = np_map
      self.map_size = map_size


# class MPA:
#     environment = []
#     list_dst = []
#     n = 0
#     x_min = 1
#     x_max = 5
#     d_min = 2
#     origin = 0.4

#     def __init__(self, filename):
#         fp = open(filename, "r")
#         f_n = int(fp.readline())
#         f_l = int(fp.readline())
#         l_dst = []
#         f_map = []
#         for f_i in range(f_l):
#             s = fp.readline()
#             s_l = s[:-1].split(" ")
#             l_dst.append([int(s_l[0]) + 0.5, int(s_l[1]) + 0.5])
#         print(l_dst)
#         for f_i in range(f_n):
#             s = fp.readline()
#             s_l = s[0:-2].split(" ")
#             f_map.append(list(map(int, s_l)))
#         for f_i in l_dst:
#             f_map[int(f_i[0])][int(f_i[1])] = 0
#         self.n = f_n
#         self.list_dst = list(l_dst)
#         self.environment = []
#         print(f_map)
#         print(self.list_dst)
#         for f_i in f_map:
#             self.environment.append(list(f_i))
#         fp.close()


mpa_obj = MPA('Test/map15_3.txt')
print(mpa_obj.map_size)
print(mpa_obj.list_goal)
print(mpa_obj.list_obstacle)
print(mpa_obj.environment)
