import math
import random as rd


class GA:

    def __init__(self, f_map):
        self.map_tsp = f_map
        self.n = len(f_map)

    def c_cost(self, sol):
        cost = 0
        pre_city = sol[-1]
        for city in sol:
            cost += self.map_tsp[pre_city][city][0]
            pre_city = city
        return cost

    def evolution(self, father, mother):
        first = rd.randint(0, self.n - 1)
        second = rd.randint(first + 1, self.n)
        older = mother[first:second]
        elder = father[first:second]
        for f_i in range(first):
            x = father[f_i]
            while older.count(x) != 0:
                x = father[mother.index(x)]
            older.insert(f_i, x)
        for f_i in range(second, self.n):
            x = father[f_i]
            while older.count(x) != 0:
                x = father[mother.index(x)]
            older.append(x)

        for f_i in range(first):
            x = mother[f_i]
            while elder.count(x) != 0:
                x = mother[father.index(x)]
            elder.insert(f_i, x)
        for f_i in range(second, self.n):
            x = mother[f_i]
            while elder.count(x) != 0:
                x = mother[father.index(x)]
            elder.append(x)
        if rd.random() < 0.1:
            first = rd.randint(0, self.n - 1)
            second = rd.randint(first + 1, self.n)
            rev = older[first:second]
            rev.reverse()
            older = older[:first] + rev + older[second:]
        if rd.random() < 0.1:
            first = rd.randint(0, self.n - 1)
            second = rd.randint(first + 1, self.n)
            rev = elder[first:second]
            rev.reverse()
            elder = elder[:first] + rev + elder[second:]
        return older, elder

    def solve(self):
        num = int((self.n * (self.n - 1)) / 2)
        if num > 30:
            num = 30
        num_child = int(num / 3)
        list_pa = []
        for i in range(num):
            prey = [x for x in range(self.n)]
            pa = []
            for j in range(self.n):
                x = prey[rd.randint(0, self.n - j - 1)]
                pa.append(x)
                prey.remove(x)
            list_pa.append(pa)
        list_rank = []
        for i in list_pa:
            list_rank.append([self.c_cost(i), i])
        list_rank.sort()
        loop = 100000
        elite = list_rank[0][0]
        loop_break = 0
        for i in range(loop):
            list_child = []
            for j in range(num_child):
                i_child1 = rd.randint(0, int(num / 2))
                i_child2 = rd.randint(i_child1, int(num / 2))
                child1, child2 = self.evolution(list_rank[i_child1][1], list_rank[i_child2][1])
                child = [self.c_cost(child1), child1]
                if list_rank.count(child) == 0:
                    list_child.append(list(child))
                child = [self.c_cost(child2), child2]
                if list_rank.count(child) == 0:
                    list_child.append(list(child))
            list_rank = list_rank[:(num - num_child)] + list_child
            list_rank.sort()
            if elite == list_rank[0][0]:
                loop_break += 1
            else:
                elite = list_rank[0][0]
                loop_break = 0
            if loop_break > 3000:
                print("break", i)
                break
        return list_rank[0]
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
