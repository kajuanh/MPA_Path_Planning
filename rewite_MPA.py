import sys, os, math
sys.path.append(os.path.abspath(""))
from mytempcode_MPA.initialization import initialization
from mytempcode_MPA.levy import levy as levys
from modify_code import distance, MPA
from scipy.stats import levy
import random as rd
import numpy as np
from Class import GridMap
import matplotlib.pyplot as plt
import matplotlib
from Class import MPAs
environment = MPA('Test/map15_3.txt')
from decimal import Decimal

mpa_obj = MPAs(environment.map_size,environment.environment)
# tl = len(mpa_obj.empty)/len(mpa_obj.obstacles)
data = [
[[1, 2], [1, 0], [14, 0], [14, 3], [12, 5], [8, 5], [6, 7], [6, 13], [10, 14], [12, 12], [11, 11]],
[[1, 2], [0, 2], [2, 0], [13, 0], [14, 4], [7, 6], [7, 7], [5, 9], [7, 14], [12, 14], [11, 11]],
[[1, 2], [0, 6], [2, 5], [6, 5], [6, 12], [10, 14], [11, 13], [11, 11]],
[[1, 2], [1, 0], [14, 0], [14, 5], [9, 5], [6, 8], [6, 14], [11, 14], [11, 11]],
[[1, 2], [1, 5], [9, 5], [6, 8], [6, 14], [12, 14], [11, 11]],
[[1, 2], [0, 6], [2, 5], [8, 5], [7, 7], [5, 9], [7, 14], [11, 13], [11, 11]],
[[1, 2], [1, 0], [12, 0], [13, 1], [13, 5], [10, 5], [7, 6], [5, 10], [7, 13], [11, 14], [11, 11]],
[[1, 2], [0, 6], [2, 5], [6, 5], [6, 11], [9, 14], [11, 14], [11, 11]],
[[1, 2], [0, 2], [2, 0], [13, 0], [13, 4], [8, 6], [6, 8], [6, 13], [11, 14], [11, 11]],
[[1, 2], [0, 1], [2, 0], [14, 0], [13, 4], [8, 6], [5, 9], [7, 13], [11, 14], [11, 11]],
[[1, 2], [1, 5], [6, 5], [6, 12], [8, 14], [12, 14], [11, 11]],
[[1, 2], [0, 0], [14, 0], [13, 4], [9, 6], [6, 5], [6, 14], [11, 13], [11, 11]],
[[1, 2], [0, 3], [2, 5], [9, 5], [8, 6], [7, 8], [5, 8], [6, 14], [12, 14], [11, 11]],
[[1, 2], [1, 4], [2, 5], [9, 5], [5, 9], [6, 14], [12, 13], [11, 11]],
[[1, 2], [1, 0], [12, 0], [14, 1], [13, 4], [7, 6], [5, 10], [9, 14], [11, 14], [11, 11]],
]
def draw_line(points:list[list],plt:matplotlib.pyplot):
    xl = []
    yl = []
    for point in points:
        xl.append(point[0]+0.5)
        yl.append(point[1]+0.5)
    plt.plot(xl,yl,'-')

def display(temp_data,final_sol,plt):
    
    fig, ax = plt.subplots()
    fig.set_size_inches(10,10)

    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax.set_xlim(0, environment.map_size+1)
    ax.set_ylim(0, environment.map_size+1)
    plt.xticks([*range(environment.map_size+1)])
    plt.yticks([*range(environment.map_size+1)])
    draw_line(final_sol,plt)
    plt.imshow(temp_data.transpose(),origin='lower',extent = (0,environment.map_size,0,environment.map_size))
    plt.grid()
    ax.invert_yaxis()

    plt.show(block=False)
    plt.pause(10)

def cdn_id(x,y,map_size):
    '''covert coordinates to index node'''
    if x<0 or x>map_size or y<0 or y> map_size:
        raise('not in (0,%d)'%map_size)
    return (y*15+x)


def id_cdn(id, map_size):
    if id < 0 or id > map_size**map_size:
        '''covert coordinates to index node'''
        raise ('not in (0,%d^2)' % map_size)
    return id % map_size,id//map_size 

def length_line_and_clt(coordinates:list):
    length = 0
    clt = 0
    for index in range(len(coordinates)-1):
        if environment.check_collision(coordinates[index],coordinates[index]):
            clt = 1
            break
        length+=distance(coordinates[index],coordinates[index+1])
    return clt,length

def calculator_node(self, f_sol, f_st, f_dst):
    cdn_sol = f_sol
    f_sol = []
    for sol in cdn_sol:
        f_sol.append(id_cdn(sol,self.map_size))
    s = 0
    # s = distance(f_sol[0],f_st)+distance(f_sol[-1],f_dst)
    # o = 0
    # if self.check_collision(f_sol[0],f_st):
    #     o+=1
    # if self.check_collision(f_sol[-1],f_dst):
    #     o+=1
    
    # for index in range(len(f_sol)-1):
    #     if self.check_collision(f_sol[index],f_sol[index+1]):
    #         o+=1
    #     s+= distance(f_sol[index],f_sol[index+1])
    # return s+o*tl
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


def func_fit(line: np):
    coordinates = []

    for node in line:
        coordinates.append(id_cdn(node, 15))
    return length_line_and_clt(coordinates)

        
def covert_list_to_index_node(points: list):
    resuilt = []
    for point in points:
        resuilt.append(cdn_id(*point,map_size=15))
    return resuilt
def covert_list_index_to_coordinates(ids:list):
    coordinates = []
    for index in ids:
        coordinates.append(id_cdn(index, 15))
    return coordinates




def full_data(data:list[list[list]],end:list[list]):
    max_len = 0
    for line in data:
        if line[0] == start:
            line.pop(0)

        if line[-1]==end:
            line.pop(-1)

        if (len_line:=len(line))>max_len:
            max_len = len_line
    
    for line in data:
        line.extend([end]*(max_len-len(line)))
    return data, max_len

start = [1,2]
end = [11,11]
data, dim = full_data(data,end)

data = np.array(data,dtype=np.int64)
def display_data(data):
    for i in data:
        display_line_array(i)
def display_line_array(line):
    s = ''
    for j in line:
        s +='[%-2s %-2s]  '%(j[0],j[1])
    print(s)

def nan_to_zeros(x:np):
    
    for i in range(x.shape[0]):
        x[i] = 0 if np.isnan(x[i]) else x[i]
    return x

# display_data(data)
# for _ in range(10):
Prey = data.copy()
SearchAgents_no = 15
lb = 0
ub = SearchAgents_no-1
Max_iter = 90
Xmin = np.ones((SearchAgents_no, dim),dtype=int)*lb
Xmax = np.ones((SearchAgents_no, dim),dtype=int)*ub
Top_predator_pos = np.zeros((dim, 2),dtype=int)
Top_predator_fit = math.inf
# print('Top_predator_pos :')
# display_line_array(Top_predator_pos)
# print('Top_predator_fit',Top_predator_pos)
# print('Xmax',Xmax)
# print('Xmin',Xmin)
Convergence_curve = np.zeros((1, Max_iter))
stepsize_x = np.zeros((SearchAgents_no, dim))
stepsize_y = np.zeros((SearchAgents_no, dim))

fitness = np.full((SearchAgents_no, 1), np.inf)
# print('Convergence_curve',Convergence_curve)
# print('stepsize',stepsize)
# print('fitness',fitness)
levy.a = 1.5
levy.b = 1
Iter = 0
FADs = 0.2
P = 0.5
for Iter in range(90):
    for i in range(SearchAgents_no):

        Flag4ubx = (Prey[i, :,0] > ub).astype(int)
        Flag4lbx = (Prey[i, :,0] < lb).astype(int)
        Flag4uby = (Prey[i, :,1] > ub).astype(int)
        Flag4lby = (Prey[i, :,1] < lb).astype(int)

        Prey[i, :, 0] = (Prey[i, :, 0]*(np.logical_not(Flag4ubx +Flag4lbx).astype(int))+ub*Flag4ubx+lb*Flag4lbx)
        Prey[i, :, 1] = (Prey[i, :, 1]*(np.logical_not(Flag4uby +Flag4lby).astype(int))+ub*Flag4uby+lb*Flag4lby)
        v,dis_prey = mpa_obj.calculator(Prey[i, :].tolist(), start, end) 

        if v == 0:
            fitness[i, 0] = dis_prey
        elif v==1:
            fitness[i, 0] = math.inf
        else:
            print(v,dis_prey,Prey[i, :])
        if (fitness[i, 0] < Top_predator_fit):
            Top_predator_fit = fitness[i, 0]

            Top_predator_pos = Prey[i].copy()
    # print(Top_predator_pos.T)
    if Iter == 0:
        fit_old = fitness.copy()
        Prey_old = Prey.copy()

    Inx = np.zeros((fitness.shape[0],1))
    for i in range(fitness.shape[0]):
        if(fit_old[i] < fitness[i]):
            Inx[i] = 0
        else:
            Inx[i] = 1

    Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
    Prey[:,:,0] = Indx*Prey_old[:,:,0] + np.logical_not(Indx).astype(int) * Prey[:,:,0]
    Prey[:,:,1] = Indx*Prey_old[:,:,1] + np.logical_not(Indx).astype(int) * Prey[:,:,1]
    with np.errstate(invalid='ignore'):
        fitness =Inx*fit_old + nan_to_zeros((np.logical_not(Inx).astype(int) * fitness))

    fit_old = fitness.copy()
    Prey_old = Prey.copy()
        

    Elite = np.full((SearchAgents_no, *Top_predator_pos.shape), Top_predator_pos,dtype=np.int64)  # %(Eq. 10)

    CF = (1-Iter/Max_iter)**(2*Iter/Max_iter)
    RLX = np.array(levy.rvs(0, 1, (SearchAgents_no, dim)))
    RLY = np.array(levy.rvs(0, 1, (SearchAgents_no, dim)))
    RBX = np.random.normal(0, 1, (SearchAgents_no, dim))
    RBY = np.random.normal(0, 1, (SearchAgents_no, dim))
    for i in range(SearchAgents_no):
        for j in range(dim):
            R = rd.uniform(0, 1)
            #  %------------------ Phase 1 (Eq.12) -------------------
            if Iter < Max_iter/3:
                stepsize_x[i, j] = RBX[i, j] * (Elite[i, j, 0]-RBX[i, j]*Prey[i, j, 0])
                stepsize_y[i, j] = RBY[i, j] * (Elite[i, j, 1]-RBY[i, j]*Prey[i, j, 1])

                Prey[i, j, 0] = Prey[i, j, 0] + P*R*stepsize_x[i, j]
                Prey[i, j, 1] = Prey[i, j, 1] + P*R*stepsize_y[i, j]
                # print('stepsize[%d, %d]'%(i,j),stepsize[i, j])

            # %--------------- Phase 2 (Eqs. 13 & 14)----------------
            elif (Iter > Max_iter/3) and (Iter < 2*Max_iter/3):
                if i > Prey.shape[0]/2:
                    stepsize_x[i, j] = RBX[i, j] * (Elite[i, j, 0]-RBX[i, j]*Prey[i, j, 0])
                    stepsize_y[i, j] = RBY[i, j] * (Elite[i, j, 1]-RBY[i, j]*Prey[i, j, 1])

                    Prey[i, j, 0] = Elite[i, j, 0] + P*CF*stepsize_x[i, j]
                    Prey[i, j, 1] = Elite[i, j, 1] + P*CF*stepsize_y[i, j]
                else:
                    stepsize_x[i, j] = (RLX[i, j] * (Elite[i, j, 0]-RLX[i, j]*Prey[i, j, 0]))
                    stepsize_y[i, j] = (RLY[i, j] * (Elite[i, j, 1]-RLY[i, j]*Prey[i, j, 1]))
                    try:
                        # fix error cant convert too large int to numpy int
                        Prey[i, j, 0] = Prey[i, j, 0] + (check_large if abs(
                            check_large := P*CF*stepsize_x[i, j]) < 15 else 15 if check_large > 0 else -15)
                        Prey[i, j, 1] = Prey[i, j, 1] + (check_large if abs(
                            check_large := P*CF*stepsize_y[i, j]) < 15 else 15 if check_large > 0 else -15)
                    except Exception as E:
                        print('error 1')
                        print('E', E)
                        print('Prey', Prey[i, j, 0])
                        print('Elite', Elite[i, j, 0])
                        print('P', P)
                        print('CF', CF)
                        print('stepsize_x', stepsize_x[i, j])
                        print('stepsize_y', stepsize_y[i, j])
                        sys.exit()


            #  %----------------- Phase 3 (Eq. 15)-------------------
            else:
                stepsize_x[i, j] = (RLX[i, j] * (RLX[i, j]*Elite[i, j, 0]-Prey[i, j, 0]))
                stepsize_y[i, j] = (RLX[i, j] * (RLX[i, j]*Elite[i, j, 1]-Prey[i, j, 1]))
                try:
                    Prey[i, j, 0] = Elite[i, j, 0] + (check_large if abs(
                        check_large := P*CF*stepsize_x[i, j]) < 15 else 15 if check_large > 0 else -15)
                    Prey[i, j, 1] = Elite[i, j, 1] + (check_large if abs(
                        check_large := P*CF*stepsize_y[i, j]) < 15 else 15 if check_large > 0 else -15)
                except Exception as E:
                    print('error 2')
                    print('E', E)
                    print('Prey', Prey[i, j, 0])
                    print('Elite', Elite[i, j, 0])
                    print('P', P)
                    print('CF', CF)
                    print('stepsize_x', stepsize_x[i, j])
                    print('stepsize_y', stepsize_y[i, j])
                    sys.exit()


    # display_data(Prey)
    for i in range(SearchAgents_no):
        # dis_prey = np.inf
        Flag4ubx = (Prey[i, :,0] > ub).astype(int)
        Flag4lbx = (Prey[i, :,0] < lb).astype(int)
        Flag4uby = (Prey[i, :,1] > ub).astype(int)
        Flag4lby = (Prey[i, :,1] < lb).astype(int)

        Prey[i, :, 0] = (Prey[i, :, 0]*(np.logical_not(Flag4ubx +
                        Flag4lbx).astype(int))+ub*Flag4ubx+lb*Flag4lbx)
        Prey[i, :, 1] = (Prey[i, :, 1]*(np.logical_not(Flag4uby +
                        Flag4lby).astype(int))+ub*Flag4uby+lb*Flag4lby)
        v,dis_prey = mpa_obj.calculator(Prey[i, :].tolist(), start, end) 

        if v == 0:
            fitness[i, 0] = dis_prey
            
        elif v==1:
            fitness[i, 0] = np.inf
        else:
            print(v,dis_prey,Prey[i, :])

        if (fitness[i, 0] < Top_predator_fit):
            
            Top_predator_fit = fitness[i, 0].copy()
            Top_predator_pos = Prey[i].copy()

    # print('Top_predator_pos:')
    # display_line_array(Top_predator_pos)
    # print('fitness',fitness)
    if Iter == 0:
        fit_old = fitness.copy()
        Prey_old = Prey.copy()

    Inx = np.zeros((fitness.shape[0],1))
    for i in range(fitness.shape[0]):
        if(fit_old[i] < fitness[i]):
            Inx[i] = 0
        else:
            Inx[i] = 1

    Indx = np.full((Inx.shape[0], dim), Inx).astype(int)

    Prey[:,:,0] = Indx*Prey_old[:,:,0] + np.logical_not(Indx).astype(int) * Prey[:,:,0]
    Prey[:,:,1] = Indx*Prey_old[:,:,1] + np.logical_not(Indx).astype(int) * Prey[:,:,1]

    with np.errstate(invalid='ignore'):
        fitness =Inx*fit_old + nan_to_zeros((np.logical_not(Inx).astype(int) * fitness))
    Prey_old = Prey.copy()
    #  %---------- Eddy formation and FADs� effect (Eq 16) -----------
    if rd.uniform(0, 1) < FADs:
        U = np.random.rand(SearchAgents_no, dim) < FADs
        Prey[:,:,0] = Prey[:,:,0]+CF * ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
        Prey[:,:,1] = Prey[:,:,1]+CF * ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
    else:
        r = rd.uniform(0, 1)
        Rs = Prey.shape[0]
        stepsize_x = (FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :,0]-Prey[np.random.permutation(Rs), :,0])
        stepsize_y = (FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :,1]-Prey[np.random.permutation(Rs), :,1])
        Prey[:,:,0] = Prey[:,:,0]+stepsize_x
        Prey[:,:,1] = Prey[:,:,1]+stepsize_y
    # # Iter = Iter+1
    Convergence_curve[:, Iter] = Top_predator_fit

# print('Prey')
# display_data(Prey)
# print('Prey_old')
# display_data(Prey_old)
# print('fit_old', fit_old.T)
# print('fitness', fitness.T)
print(Top_predator_pos.T)
print('Top_predator_pos')
display_line_array(Top_predator_pos)
print('Top_predator_fit',Top_predator_fit)
print('Convergence_curve', Convergence_curve)
print(shorten:= mpa_obj.shorten(Top_predator_pos.tolist(),start,end))
print(mpa_obj.calculator(Top_predator_pos.tolist(),start,end))
print(mpa_obj.calculator(shorten,start,end))
display(mpa_obj.data,shorten,plt)

'''
end_node = cdn_id(11,11,15)
data_node = []
    data_node.append(covert_list_to_index_node(line))
# print(data_node)
# print(max_len)
# frmt = "{:>10}"*max_len
for line in data_node:
    line.extend([end_node]*(max_len-len(line)))
st = [1,2]
dst = [11,11]
# for i in data_node:
#     print('\t'.join(str(x)for x in i))
Prey=np.array(data_node,dtype=np.int64)
# print(Prey.shape)
# print('Prey',Prey)
# print('func_fit',func_fit(Prey[1,:]))
# so_sanh = initialization(11,15,0,15**2)
dim = max_len
SearchAgents_no = 15
lb = 0
ub = SearchAgents_no**2-1
Max_iter = 50
# so_sanh = initialization(SearchAgents_no, dim, ub, lb)

# print(so_sanh)
# print(so_sanh.shape)
Xmin = np.ones((SearchAgents_no, dim),dtype=np.int64)*lb
Xmax = np.ones((SearchAgents_no, dim),dtype=np.int64)*ub
Top_predator_pos = np.zeros((1, dim),dtype=np.int64)
Top_predator_fit = math.inf

Convergence_curve = np.zeros((1, Max_iter))
stepsize = np.zeros((SearchAgents_no, dim))
fitness = np.full((SearchAgents_no, 1), np.inf)
levy.a = 1.5
levy.b = 1
Iter = 0
FADs = 0.2
P = 0.5
while Iter<Max_iter:
    for i in range(SearchAgents_no):
        Flag4ub = (Prey[i, :] > ub).astype(np.int64)
        Flag4lb = (Prey[i, :] < lb).astype(np.int64)

        Prey[i, :] = (Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)
        
        v,dis_prey = calculator(environment,Prey[i, :].tolist(), st, dst) 
        if v == 0:
            fitness[i, 0] = dis_prey
        if (fitness[i, 0] < Top_predator_fit):
            Top_predator_fit = fitness[i, 0]
            Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
        # display(environment.data,covert_list_index_to_coordinates(Top_predator_pos[0]),plt)
    # %------------------- Marine Memory saving1 -------------------

    if Iter == 0:
        fit_old = fitness
        Prey_old = Prey

    Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
    for i in range(fitness.shape[0]):
        if(fit_old[i] < fitness[i]):
            Inx[i] = 0
        else:
            Inx[i] = 1

    Indx = np.full((Inx.shape[0], dim), Inx).astype(np.int64)
    Prey = Indx*Prey_old + np.logical_not(Indx).astype(np.int64) * Prey
    fitness = Inx*fit_old + np.logical_not(Inx).astype(np.int64) * fitness
    fit_old = fitness
    Prey_old = Prey

    Elite = np.full((SearchAgents_no, Top_predator_pos.shape[1]), Top_predator_pos,dtype=np.int64)  # %(Eq. 10)
    CF = (1-Iter/Max_iter)**(2*Iter/Max_iter)
    RL = np.array(levy.rvs(0, 1,(SearchAgents_no, dim)))
    # print(RL)
    RB = np.random.normal(0, 1, (SearchAgents_no, dim))
    for i in range(Prey.shape[0]):
        for j in range(Prey.shape[1]):
            R = rd.uniform(0, 1)
            # print('R',R)
            #  %------------------ Phase 1 (Eq.12) -------------------
            if Iter < Max_iter/3:
                stepsize[i, j] = RB[i, j]*(Elite[i, j]-RB[i, j]*Prey[i, j])
                # print('stepsize[%d, %d]'%(i,j),stepsize[i, j])

                Prey[i, j] = Prey[i, j]+2/(P*R*stepsize[i, j])
            # %--------------- Phase 2 (Eqs. 13 & 14)----------------
            elif (Iter > Max_iter/3) and (Iter < 2*Max_iter/3):
                if i > Prey.shape[0]/2:
                    stepsize[i, j] = RB[i, j] * \
                        (RB[i, j]*Elite[i, j]-Prey[i, j])
                    print(P*R*stepsize[i, j])
                    Prey[i, j] = Elite[i, j]+2/(P*CF*stepsize[i, j])
                else:
                    stepsize[i, j] = RL[i, j] * \
                        (Elite[i, j]-RL[i, j]*Prey[i, j])
                    print(P*R*stepsize[i, j])

                    Prey[i, j] = Prey[i, j]+2/(P*CF*stepsize[i, j])
                    
            #  %----------------- Phase 3 (Eq. 15)-------------------
            else:
                stepsize[i, j] = RL[i, j]*(RL[i, j]*Elite[i, j]-Prey[i, j])
                Prey[i, j] = Elite[i, j]+2/(P*CF*stepsize[i, j])
    # %------------------- Detecting top predator2 -----------------
    for i in range(Prey.shape[0]):
        Flag4ub = (Prey[i, :] > ub).astype(int)
        Flag4lb = (Prey[i, :] < lb).astype(int)
        Prey[i, :] = (Prey[i, :]*(np.logical_not(Flag4ub+Flag4lb).astype(int))+ub*Flag4ub+lb*Flag4lb)

        v,dis_prey = calculator(environment,Prey[i, :].tolist(), st, dst) 
        if v == 0:
            fitness[i, 0] = dis_prey

        if(fitness[i, 0] < Top_predator_fit):
            Top_predator_fit = fitness[i, 0]
            Top_predator_pos = Prey[i].reshape((Top_predator_pos.shape))
    # %---------------------- Marine Memory saving2 ----------------
    if Iter == 0:
        fit_old = fitness
        Prey_old = Prey
        # print('fit_old_0', fit_old)
    Inx = np.zeros(fitness.shape[0]).reshape(fitness.shape[0], 1)
    for i in range(fitness.shape[0]):
        if(fit_old[i] < fitness[i]):
            Inx[i] = 0
        else:
            Inx[i] = 1
    # print(Inx)
    Indx = np.full((Inx.shape[0], dim), Inx).astype(int)
    Prey = Indx*Prey_old + np.logical_not(Indx).astype(int) * Prey
    fitness = Inx*fit_old + np.logical_not(Inx).astype(int) * fitness
    fit_old = fitness
    Prey_old = Prey

    #  %---------- Eddy formation and FADs� effect (Eq 16) -----------
    # if rd.uniform(0, 1) < FADs:
    #     U = np.random.rand(SearchAgents_no, dim) < FADs
    #     Prey = Prey+CF * \
    #         ((Xmin+np.random.rand(SearchAgents_no, dim)*(Xmax-Xmin))*U)
    # else:
    #     r = rd.uniform(0, 1)
    #     Rs = Prey.shape[0]
    #     stepsize = (
    #         FADs*(1-r)+r)*(Prey[np.random.permutation(Rs), :]-Prey[np.random.permutation(Rs), :])
    #     Prey = Prey+stepsize
    Iter = Iter+1
    Convergence_curve[:, Iter-1] = Top_predator_fit
    # print('fitness', fitness)

# print({'Top_predator_fit': Top_predator_fit, 'Top_predator_pos': Top_predator_pos, 'Convergence_curve': Convergence_curve, })
print('Top_predator_fit', Top_predator_fit)
print('Top_predator_pos', Top_predator_pos)
print('Convergence_curve', Convergence_curve)
print()
path = environment.shorten(covert_list_index_to_coordinates(Top_predator_pos[0]),[1,2],[11,11])
print(path)
path.insert(0,[1,2])
display(environment.data,path,plt)
print(calculator(environment,covert_list_to_index_node(path),[1,2],[11,11]))
display(environment.data,path,plt)
'''
