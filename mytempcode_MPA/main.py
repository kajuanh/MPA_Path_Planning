import Get_Functions_details as GFd
import MPA as mpa
import numpy as np
import matplotlib.pyplot as plt
# fobj = @YourCostFunction
# dim = number of your variables
# Max_iteration = maximum number of iterations
# SearchAgents_no = number of search agents
# lb=[lb1,lb2,...,lbn] where lbn is the lower bound of variable n
# ub=[ub1,ub2,...,ubn] where ubn is the upper bound of variable n
SearchAgents_no = 25  # Number of search agents
Function_name = "F1"
Max_iteration = 50

ludf = GFd.switcher(int(Function_name[1]))
# ludf=GFd.switcher(int(Function_name[1]))
lb, ub, dim, fobj = ludf['lb'], ludf['ub'], ludf['dim'], ludf['fobj']
print(lb, ub, dim, fobj)
kq = mpa.MPA(SearchAgents_no, Max_iteration, lb, ub, dim, fobj)
Best_score, Best_pos, Convergence_curve = kq['Top_predator_fit'], kq[
    'Top_predator_pos'], kq['Convergence_curve']

print("Best_score", Best_score)
print("Best_pos", Best_pos)
# print("Best_pos",Best_pos)
print("Convergence_curve", Convergence_curve)
# print("Convergence_curve",Convergence_curve.shape)
# for i in range(5):
x_Max_iteration = np.arange(0, Max_iteration, 1)
x_Max_iteration = x_Max_iteration.reshape(1, Max_iteration)
# print(x_Max_iteration)
# plt.plot(x_Max_iteration[0], Convergence_curve[0], 'r--', logy=True)
# s= Series(x_Max_iteration[0], index=Convergence_curve[0])
# plt.subplot(2, 1, 1)
plt.plot(x_Max_iteration[0], Convergence_curve[0], color='blue', lw=2)
plt.yscale('log')
plt.show()
#     print("Best_score", Best_score)
print("kt")
