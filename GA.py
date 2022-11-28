import numpy as np

class GA_TSP:
    def __init__(self, map_tsp, k=20, max_gen=5, mutation_prob=0.1, index_s=None):
        self.mutation_prob = mutation_prob
        self.num_t = len(map_tsp)
        self.map_tsp = map_tsp
        self.max_gen = max_gen
        self.index_s = index_s
        self.map_cost = {}
        self.k = k

    def compute_fitness(self, s):
        if (l := self.map_cost.get(str(s))) is not None:
            return l
        else:
            l = 0
            for i in range(self.num_t-1):
                l += self.map_tsp[s[i]][s[i+1]]
            if self.index_s is not None:
                l += self.map_tsp[self.index_s][s[0]]
                l += self.map_tsp[s[-1]][self.index_s]
            else:
                l += self.map_tsp[s[len(s)-1]][s[0]]

            self.map_cost[str(s)] = l
        return l

    
    def selection(self, gen, k):
        gen = sorted(gen, key=lambda s: self.compute_fitness(s))
        return gen[:k]

    def crossover(self, s1, s2, index):
        m_1, f_2 = s1.copy(), s2.copy()

        c1 = f_2.copy()
        for i in range(index, len(m_1)): c1.remove(m_1[i])
        for i in range(index, len(m_1)): c1.append(m_1[i])

        c2 = m_1.copy()
        for i in range(index, len(f_2)): c2.remove(f_2[i])
        for i in range(index, len(f_2)): c2.append(f_2[i])
        return (c1, c2)

    def mutation(self, s, m, n):
        i, j = min(m, n), max(m, n)
        cs = s.copy()
        while i < j:
            cs[i], cs[j] = cs[j], cs[i]
            i += 1
            j -= 1
        return cs

    def init_population(self):
        path = list(range(self.num_t))
        if self.index_s is not None:
            path.pop(self.index_s)
            self.num_t -=1
        n_p = self.k
        gen = [path]
        while len(gen) < n_p:
            copy_path = path.copy()
            np.random.shuffle(copy_path)
            # try:
            #     gen.index(copy_path)
            # except Exception:
            #     gen.append(copy_path)
            if copy_path not in gen:
                gen.append(copy_path)
        return gen

    def solve(self):
        ''' return best_cost, best '''
        gen = self.init_population()
        for trial in range(self.max_gen):
            gen = self.selection(gen, self.k)
            next_gen = []
            for i in range(len(gen)):
                for j in range(i+1, len(gen)):
                    c1, c2 = self.crossover(gen[i], gen[j], np.random.randint(0, len(gen[i])))
                    next_gen.append(c1)
                    next_gen.append(c2)
                if np.random.rand() < self.mutation_prob:
                    m = np.random.randint(0, len(gen[i]))
                    while True:
                        n = np.random.randint(0, len(gen[i]))
                        if m != n:
                            break
                    c = self.mutation(gen[i], m, n)
                    next_gen.append(c)
            gen = next_gen

        best_gen = self.selection(gen, 1)[0]
        best = (round(self.compute_fitness(best_gen),3), best_gen)
        return best