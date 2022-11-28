'''map_tsp = [
    [ 0.0 , 12.23,  6.65, 15.99, 22.5 ],
    [12.23,  0.0 , 18.48, 18.99, 13.21],
    [ 6.65, 18.48,  0.0 , 10.58, 22.46],
    [15.99, 18.99, 10.58,  0.0 , 10.58],
    [22.5 , 13.21, 22.46, 10.58,  0.0 ],
]

>>>>>best_cost
 53.242
[[None list([[0, 1], [1, 4], [2, 6], [2, 10], [0, 12]])
  list([[0, 1], [2, 0], [5, 0], [6, 1]])
  list([[0, 1], [0, 3], [2, 5], [10, 5], [13, 4]])
  list([[0, 1], [1, 4], [2, 5], [6, 5], [6, 11], [8, 13], [13, 14]])]
 [list([[0, 12], [2, 10], [2, 6], [1, 4], [0, 1]]) None
  list([[0, 12], [2, 10], [2, 6], [1, 4], [1, 0], [5, 0], [6, 1]])
  list([[0, 12], [2, 10], [2, 5], [10, 5], [13, 4]])
  list([[0, 12], [3, 13], [13, 14]])]
 [list([[6, 1], [5, 0], [2, 0], [0, 1]])
  list([[6, 1], [5, 0], [1, 0], [1, 4], [2, 6], [2, 10], [0, 12]]) None
  list([[6, 1], [7, 0], [10, 0], [13, 1], [13, 4]])
  list([[6, 1], [6, 0], [10, 0], [13, 1], [14, 3], [12, 5], [12, 12], [13, 14]])]
 [list([[13, 4], [10, 5], [2, 5], [0, 3], [0, 1]])
  list([[13, 4], [10, 5], [2, 5], [2, 10], [0, 12]])
  list([[13, 4], [13, 1], [10, 0], [7, 0], [6, 1]]) None
  list([[13, 4], [12, 5], [12, 11], [13, 14]])]
 [list([[13, 14], [8, 13], [6, 11], [6, 5], [2, 5], [1, 4], [0, 1]])
  list([[13, 14], [3, 13], [0, 12]])
  list([[13, 14], [12, 12], [12, 5], [14, 3], [13, 1], [10, 0], [6, 0], [6, 1]])
  list([[13, 14], [12, 11], [12, 5], [13, 4]]) None]]
'''
from VisualizeResult_copy import read_file,split_map,  DisplayMatplotlib
map, map_size, target, order, dis, result, time, map_tsp, map_way  = read_file('Solutions/map15_11_s_sol.txt',1)
goals, o_env, start, is_have_s = split_map(map)
# Python3 program to solve
# Traveling Salesman Problem using
# Branch and Bound.
import math
maxsize = float('inf')
 
# Function to copy temporary solution
# to the final solution
def copyToFinal(curr_path,N):
    final_path[:N + 1] = curr_path[:]
    final_path[N] = curr_path[0]
 
# Function to find the minimum edge cost
# having an end at the vertex i
def firstMin(adj, i,N):
    min = maxsize
    for k in range(N):
        if adj[i][k] < min and i != k:
            min = adj[i][k]
 
    return min
 
# function to find the second minimum edge
# cost having an end at the vertex i
def secondMin(adj, i,N):
    first, second = maxsize, maxsize
    for j in range(N):
        if i == j:
            continue
        if adj[i][j] <= first:
            second = first
            first = adj[i][j]
 
        elif(adj[i][j] <= second and
             adj[i][j] != first):
            second = adj[i][j]
 
    return second
 
# function that takes as arguments:
# curr_bound -> lower bound of the root node
# curr_weight-> stores the weight of the path so far
# level-> current level while moving
# in the search space tree
# curr_path[] -> where the solution is being stored
# which would later be copied to final_path[]
def TSPRec(adj, curr_bound, curr_weight,
              level, curr_path, visited,N):
    global final_res
     
    # base case is when we have reached level N
    # which means we have covered all the nodes once
    if level == N:
         
        # check if there is an edge from
        # last vertex in path back to the first vertex
        if adj[curr_path[level - 1]][curr_path[0]] != 0:
             
            # curr_res has the total weight
            # of the solution we got
            curr_res = curr_weight + adj[curr_path[level - 1]]\
                                        [curr_path[0]]
            if curr_res < final_res:
                copyToFinal(curr_path, N)
                final_res = curr_res
        return
 
    # for any other level iterate for all vertices
    # to build the search space tree recursively
    for i in range(N):
         
        # Consider next vertex if it is not same
        # (diagonal entry in adjacency matrix and
        #  not visited already)
        if (adj[curr_path[level-1]][i] != 0 and
                            visited[i] == False):
            temp = curr_bound
            curr_weight += adj[curr_path[level - 1]][i]
 
            # different computation of curr_bound
            # for level 2 from the other levels
            if level == 1:
                curr_bound -= ((firstMin(adj, curr_path[level - 1], N) +
                                firstMin(adj, i, N)) / 2)
            else:
                curr_bound -= ((secondMin(adj, curr_path[level - 1], N) +
                                 firstMin(adj, i, N)) / 2)
 
            # curr_bound + curr_weight is the actual lower bound
            # for the node that we have arrived on.
            # If current lower bound < final_res,
            # we need to explore the node further
            if curr_bound + curr_weight < final_res:
                curr_path[level] = i
                visited[i] = True
                 
                # call TSPRec for the next level
                TSPRec(adj, curr_bound, curr_weight,
                       level + 1, curr_path, visited, N)
 
            # Else we have to prune the node by resetting
            # all changes to curr_weight and curr_bound
            curr_weight -= adj[curr_path[level - 1]][i]
            curr_bound = temp
 
            # Also reset the visited array
            visited = [False] * len(visited)
            for j in range(level):
                if curr_path[j] != -1:
                    visited[curr_path[j]] = True
 
# This function sets up final_path
def TSP(adj,N):
     
    # Calculate initial lower bound for the root node
    # using the formula 1/2 * (sum of first min +
    # second min) for all edges. Also initialize the
    # curr_path and visited array
    curr_bound = 0
    curr_path = [-1] * (N + 1)
    visited = [False] * N
 
    # Compute initial bound
    for i in range(N):
        curr_bound += (firstMin(adj, i,N) +
                       secondMin(adj, i,N))
 
    # Rounding off the lower bound to an integer
    curr_bound = math.ceil(curr_bound / 2)
 
    # We start at vertex 1 so the first vertex
    # in curr_path[] is 0
    visited[0] = True
    curr_path[0] = 0
 
    # Call to TSPRec for curr_weight
    # equal to 0 and level 1
    TSPRec(adj, curr_bound, 0, 1, curr_path, visited, N)
 
# Driver code
 
# Adjacency matrix for the given graph
# adj = [[0, 10, 15, 20],
#        [10, 0, 35, 25],
#        [15, 35, 0, 30],
#        [20, 25, 30, 0]]
# N = 4
adj = map_tsp
N = len(map_tsp)
# final_path[] stores the final solution
# i.e. the // path of the salesman.
final_path = [None] * (N + 1)
 
# visited[] keeps track of the already
# visited nodes in a particular path
visited = [False] * N
 
# Stores the final minimum weight
# of shortest tour.
import time
final_res = maxsize
st = time.time()
TSP(adj,N)
et = time.time()
print("Minimum cost :", final_res)
print("Path Taken : ", end = ' ')
for i in range(N + 1):
    print(final_path[i], end = ' ')
print('\nt1 %.4f'%(et-st))


from new_Ga import GA_TSP
st2 = time.time()
ga_tsp = GA_TSP(map_tsp,N*(N-1),15,0.3,index_s=0)
best_cost, best = ga_tsp.solve()
et2 = time.time()

print('>>>>>>>',best_cost)
best.insert(0,0)
best.append(0)
print('>>>>>>>',best)
print('t2 %.4f'%(et2-st2))
print(str(final_path))
l1 = []
l2 = []
for i in range(len(final_path)-1):
        l1.extend(map_way[final_path[i]][final_path[i+1]])
for j in range(len(best)-1):
        l2.extend(map_way[best[j]][best[j+1]])
dis1 = DisplayMatplotlib(map_size, o_env, l1, target[0], goals,final_res)
dis2 = DisplayMatplotlib(map_size, o_env, l2, target[0], goals,best_cost)
dis1.draw(arrow=True)
dis2.draw(arrow=True)