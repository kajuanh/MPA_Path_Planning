import sys, os, math
sys.path.append(os.path.abspath(""))
from modify_code import distance, MPA
from Class import MPAs

environment = MPA('Test/map30_7.txt')
mpa_obj = MPAs(environment.map_size,environment.environment)

# path = Top_predator_pos.tolist()
# path = mpa_obj.shorten(path, [1, 2], [11, 11])
# print(path)
# path = [[1, 5],  [6,  5],  [6, 12], [8, 14],  [
#     11, 13], [11, 11], [9, 9], [6, 6], [7, 7]]
# path = [0, 4], [6,  5], [4, 10], [7,  13], [9,  11],[8,  8],[1,  1],[11, 11],[5,  5]
# clr# print(mpa_obj.calculator(path,[1,1],[11,11]))
# print(path_short := mpa_obj.shorten(path,start,end))
# print(mpa_obj.calculator(path_short,[1,1],[11,11]))

# def shorten(list,st,en):
# len_list = len(path)
# print(len_list)

# print(len_list)
# remove_index = []
# for i in range(1,len_list-1):
#     print(i)
#     if mpa_obj.check_collision(path[i],path[i-1]) and mpa_obj.check_collision(path[i],path[i+1]):
#         remove_index.append(i)

# for i in range(len(remove_index)):
#     print(path[remove_index[i-i]])
#     path.pop(remove_index[i-i])
# if mpa_obj.check_collision(path[0],start) and mpa_obj.check_collision(path[0],path[1]):
    
# for x in remove_index:
#     path.pop(x)
# print(path)
def remove_no_connection(self,path :list[list],end:list[int]):
    len_list = len(path)
    remove_index = []
    for i in range(1,len_list-1):
        if self.check_collision(path[i],path[i-1]) and mpa_obj.check_collision(path[i],path[i+1]):
            remove_index.append(i)
    if mpa_obj.check_collision(path[-1],end) and mpa_obj.check_collision(path[-1],path[-2]):
            remove_index.append(len_list-1)
    remove_index.reverse()
    for i in remove_index:
        path.pop(i)
        
    # for x in remove_index:
    #     path.pop(x)
    return path

def remove_duplicate(path :list[list]):
    duplicate = True
    while duplicate:
        length = len(path)
        index_l = index_r = None
        for cdn in path:
            if path.count(cdn)>1:
                index_l = path.index(cdn)
                index_r = length - 1 - path[::-1].index(cdn)
                break
        if index_l is not None and index_r is not None:
            temp = path[:index_l]
            temp.extend(path[index_r:])
            path = temp
        else:
            duplicate = False
    return path

def shorten(self, path: list[list[int,int]],start: list[int,int],end: list[int,int]):
    len_list = len(path)
    index_l = 0
    index_r = len_list-1
    print(index_r)
    for i in range(len_list):
        if path[i][0]==start[0]and path[i][1]==start[1]:
            index_l = i+1
    for j in range(index_r,-1,-1):
        if path[j][0]==end[0] and path[j][1]==end[1]:
            index_r = j-1
    path_trip = path[index_l:index_r+1]
    path_trip = remove_no_connection(self,path_trip,end)
    print(path_trip)
    path_trip = remove_duplicate(path_trip)
    len_list = len(path_trip)
    index_l = tem_id = 0
    index_r = len_list-1
    left = [start]
    right = [end]
    t=0
    while t<10 and self.check_collision(left[-1],right[-1]):
        far_l = near_r = None
        for i in range(index_l, index_r+1):
            # print(path_trip[i])
            if not self.check_collision(left[-1], path_trip[i]) and not (left[-1][0] == path_trip[i][0] and left[-1][1] == path_trip[i][1]):
                far_l = path_trip[i]
                tem_id = i+1
        if far_l is not None:
            left.append(far_l)
            index_l = tem_id

        if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1] or not self.check_collision(left[-1],right[-1]):
            break

        for j in range(index_r,index_l-1,-1):
            print(path_trip[j])
            if not self.check_collision(right[-1], path_trip[j]) and not (right[-1][0] == path_trip[j][0] and right[-1][1] == path_trip[j][1]) :
                near_r = path_trip[j]
                tem_id = j-1
        if near_r is not None:
            right.append(near_r)
            index_r = tem_id
        
        if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1]or not self.check_collision(left[-1],right[-1]):
            break
        t+=1
        # print(index_l,index_r)
        print('left',left)
        print('right',right)
    if t==10:
        print(path)
        raise('loi')
    if left[-1][0]==right[-1][0] and left[-1][1]==right[-1][1]:
        right.pop()
    left.pop(0)
    right.reverse()
    left.extend(right)
    left = remove_duplicate(left)
    return left

# display(mpa_obj.data,right,plt)
# print(mpa_obj.check_collision([7,13],[11,11]))
start = [12,1]
end = [23,6]
path_2 = [(12, 1), [13, 1], [14, 1], [16, 1], [18, 0], [20, 2], [20, 3], [19, 4], [18, 4], [17, 3], [16, 3], [16, 5], [16, 4], [15, 4], [17, 4], [17, 2], [19, 1], [20, 0], [22, 0], [21, 1], [20, 1], [22, 2], [20, 4], [19, 3], [21, 5], [23, 5], [24, 4], [
    24, 2], [23, 3], [23, 4], [24, 5], [22, 7], [23, 7], [21, 6], [22, 6], [20, 5], [20, 7], [21, 8], [20, 6], [18, 5], [19, 6], [19, 8], [19, 7], [18, 6], [20, 8], [22, 8], [24, 7], [24, 8], [23, 6], [23, 8], [23, 10], [22, 9], [23, 9], [21, 11]]
print(environment.goals)
path_3=[(12, 1), [14, 1], [15, 1], [17, 1]]
path_4 = [(12, 1), [11, 1], [13, 1], [14, 1], [15, 1], 
        [16, 1], [18, 0], [18, 2], [17, 4], [16, 3], 
        [18, 4], [16, 4], [16, 2], [18, 1], [18, 3],
        [20, 5], [19, 7], [18, 6], [18, 7], [19, 6], 
        [21, 4], [19, 3], [20, 4], [22, 5], [22, 6], 
        [24, 4], [23, 5], [24, 5], [22, 4], [20, 2], 
        [22, 1], [20, 0], [22, 2], [23, 3], [21, 1], 
        [22, 3], [20, 1], [21, 3], [21, 5], [23, 6], 
        [21, 6], [23, 4], [21, 2], [21, 0], [19, 1], 
        [17, 1], [19, 0], [17, 2], [19, 2], [17, 3], 
        [18, 5], [20, 6], [22, 8], [20, 7], [22, 7], 
        [20, 8], [21, 8], [21, 7], [21, 9], [21, 10], 
        [23, 8], [24, 7], [24, 6], [23, 7], [22, 9], 
        [23, 9], [23, 10], [23, 12], [22, 10], [24, 12], 
        [24, 13], [23, 15], [25, 15], [23, 13], [25, 14], 
        [24, 14], [24, 15], [22, 14], [20, 13], [20, 11]]
path_5  = [(12, 1), [14, 1], [13, 1], [15, 1], [16, 1], [18, 0], 
        [20, 1], [21, 1], [20, 2], [18, 2], [16, 0], [17, 0], 
        [18, 1], [19, 3], [19, 2], [17, 4], [16, 3], [18, 3], 
        [16, 4], [18, 6], [17, 5], [18, 7], [19, 6], [18, 4], 
        [19, 5], [17, 3], [16, 5], [16, 6], [16, 7], [17, 9], 
        [15, 8], [17, 8], [19, 8], [20, 7], [21, 7], [23, 5], 
        [23, 6], [22, 8], [22, 6], [24, 7], [24, 5], [22, 3], 
        [23, 2], [21, 2], [22, 4], [23, 3], [24, 2], [24, 3], 
        [22, 2], [20, 4], [21, 5], [22, 5], [20, 6], [20, 8], 
        [18, 8], [18, 10], [16, 9], [15, 7], [17, 7], [18, 5], 
        [20, 5], [22, 7], [21, 9], [23, 9], [22, 11], [23, 11], 
        [25, 13], [24, 14], [22, 12], [22, 13], [24, 15], [22, 15], 
        [23, 14], [22, 16], [23, 15], [21, 14], [23, 13], [21, 11], 
        [21, 13], [21, 15], [21, 16], [22, 14], [20, 16], [20, 14], 
        [21, 12], [23, 12], [22, 10], [22, 9], [21, 10], [23, 8], 
        [21, 6], [19, 4], [20, 3], [21, 4], [21, 3], [23, 4], [24, 6], 
        [24, 8], [23, 7], [21, 8], [19, 7], [17, 6], [16, 8], [14, 8], 
        [12, 7], [11, 7], [13, 7], [14, 9], [14, 10], [13, 8], [15, 10], 
        [13, 9], [12, 11], [10, 13], [8, 11], [6, 9], [8, 9], [7, 7], 
        [8, 8], [6, 8], [6, 7], [8, 7], [7, 8], [5, 8], [7, 10], [7, 12], 
        [7, 11], [9, 13], [9, 12], [8, 12], [7, 13], [5, 13], [5, 15], 
        [3, 15], [4, 13], [5, 12], [5, 14], [6, 16], [8, 14], [7, 16], 
        [6, 14], [6, 15], [4, 14], [2, 16], [2, 15], [3, 17], [3, 16], 
        [4, 15], [5, 17], [6, 17], [5, 16], [7, 17], [9, 17], [8, 18], 
        [9, 20], [7, 20], [9, 19], [8, 21], [8, 23], [6, 22], [8, 22], 
        [6, 20], [7, 19], [7, 21], [6, 21], [7, 22], [7, 23], [9, 21], 
        [9, 22], [8, 20], [7, 18], [8, 17], [8, 15], [8, 13], [9, 11], 
        [10, 12], [10, 11], [12, 12], [12, 14], [13, 12], [13, 11], 
        [12, 13], [11, 12], [11, 14], [11, 15], [12, 15], [13, 15], 
        [14, 15], [15, 14], [14, 16], [16, 15], [15, 15], [13, 17], 
        [12, 16], [13, 14], [15, 16], [17, 15], [16, 14], [16, 16], 
        [16, 18], [14, 18], [14, 17], [16, 17], [17, 19], [16, 20], 
        [16, 22], [18, 23], [17, 22], [18, 21], [16, 19], [18, 20], 
        [17, 20], [15, 18], [13, 16], [14, 14], [14, 12], [16, 13], 
        [17, 11], [16, 10], [17, 10], [18, 11], [18, 12], [18, 14], 
        [20, 13], [19, 12], [18, 13], [17, 13], [16, 12], [17, 12], 
        [19, 13], [20, 12], [19, 14], [18, 16], [18, 17], [18, 15], 
        [17, 14], [19, 16], [19, 15], [17, 16], [19, 17], [17, 17], 
        [17, 18], [15, 17], [13, 18], [15, 19], [15, 20], [17, 21], 
        [15, 22], [14, 23], [13, 22], [12, 22], [14, 20], [16, 21], 
        [18, 22], [19, 20], [21, 19], [23, 20]]
path_6 = [[1, 5], [6, 5], [6, 12], [7, 13], [11, 13], [8, 8], [9, 9], [1, 1], [10, 10]]
path_8 = [
    (1, 14), [0, 16], [2, 16], [1, 16], [2, 15], [4, 15], [5, 13], [5, 12], [6, 11],
    [6, 13], [8, 15], [6, 17], [7, 16], [5, 17], [6, 16], [8, 17], [8, 19], [10, 19],
    [8, 21], [8, 22], [9, 22], [7, 21], [7, 20], [6, 20], [8, 20], [9, 21], [9, 20], 
    [11, 20], [12, 19], [13, 21], [15, 23], [16, 23], [14, 21], [14, 19], [13, 20], 
    [13, 22], [12, 20], [12, 21], [13, 23], [13, 24], [11, 23], [12, 25], [11, 25], 
    [11, 24], [13, 25], [14, 26], [13, 28], [13, 29], [12, 27], [13, 27], [13, 26], 
    [12, 24], [12, 26], [14, 28], [15, 27], [17, 29], [16, 27], [16, 29], [17, 28], 
    [19, 27], [18, 27], [17, 27], [18, 28], [16, 26], [17, 26], [15, 28], [15, 26], 
    [17, 24], [16, 24], [16, 22], [15, 22], [17, 21], [17, 20], [15, 21], [16, 21], 
    [17, 22], [18, 23], [18, 22], [19, 23], [19, 22], [19, 20], [21, 20], [21, 21], 
    [23, 19], [22, 21], [23, 22], [23, 23], [25, 21], [27, 22], [29, 23], [27, 23], 
    [28, 21], [28, 19], [27, 21], [26, 22], [26, 24], [27, 26], [26, 26], [24, 26], 
    [24, 24], [24, 25], [23, 25], [25, 27], [25, 26], [27, 28], [27, 29], [28, 29], 
    [27, 27], [28, 28], [28, 27], [26, 29], [26, 27], [24, 27], [24, 29], [24, 28], 
    [23, 26], [23, 24], [24, 23], [25, 22], [25, 23], [24, 21], [22, 19], [23, 21], 
    [25, 19], [26, 19], [24, 20], [22, 22], [24, 22], [26, 20], [26, 21], [28, 20], 
    [27, 19], [25, 20], [24, 19], [22, 20], [21, 22], [23, 20], [21, 19], [19, 19], 
    [17, 19], [16, 17]]
# print (shorten(mpa_obj,path_2, start,end))
# print (shorten(mpa_obj,path_1, start,end))
# print (shorten(mpa_obj,path_5, [12,1],[23,23]))
# print(path_3)
# print(path_2[1:2])
# print(mpa_obj.check_collision([6,12],[7,13]))
list = [[1,2],[4,5],[2,5],[1,3],[2,5],[7,8]]
path_7=[[1, 5], [6, 5], [6, 12], [8, 14], [11, 13], [10, 10], [10, 10], [7, 7], [6, 6]]
# print (shorten(mpa_obj,path_5, [12,1],[23,23]))
print (shorten(mpa_obj,path_8, [1,14],[18,17]))
# print(list[-2])
# print(mpa_obj.shorten(path_8, [1,2],[11,11]))
# print (remove_no_connection(mpa_obj,path_6,[11,11]))
# path_trip = [[6, 12]]
# left = [[6, 5]]
# print(not(left[-1][0] == path_trip[0][0] and left[-1][1] == path_trip[0][1]))