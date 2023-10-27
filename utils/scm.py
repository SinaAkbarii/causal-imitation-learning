import numpy as np



class SCMFull:

    def __init__(self):
        self.pt = [0.95, 0.05]  # [P(T=0),  P(T=1)]
        self.y_t1 = [0.2, 0.5, 0.3]  # P(Y|T=1)
        self.pc = [0.6, 0.4]  # [P(C=0),  P(C=1)]
        self.pu = [0.2, 0.8]  # [P(U=0),  P(U=1)]
        self.px_c_u = [[0.3, 0.7],  # [P(X=0|C=0, U=0), P(X=1|C=0, U=0)]
                       [0.3, 0.7],  # [P(X=0|C=0, U=1), P(X=1|C=0, U=1)]
                       [1.0, 0.0],  # [P(X=0|C=1, U=0), P(X=1|C=1, U=0)]
                       [0.0, 1.0]]  # [P(X=0|C=1, U=1), P(X=1|C=1, U=1)]

        self.ps_x_u = [[0.0, 1.0],  # [P(S=0|X=0, U=0), P(S=1|X=0, U=0)]
                       [1.0, 0.0],  # [P(S=0|X=0, U=1), P(S=1|X=0, U=1)]
                       [1.0, 0.0],  # [P(S=0|X=1, U=0), P(S=1|X=1, U=0)]
                       [0.0, 1.0]]  # [P(S=0|X=1, U=1), P(S=1|X=1, U=1)]

        self.py_s = [[0.8, 0.1, 0.1],  # [P(Y=0|S=0), P(Y=1|S=0), P(Y=2|S=0)]
                     [0.05, 0.2, 0.75]]  # [P(Y=0|S=1), P(Y=1|S=1), P(Y=2|S=1)]

    def gen_data(self, n):  # generate n samples
        card_t = len(self.pt)
        card_u = len(self.pu)
        card_c = len(self.pc)
        card_x = len(self.px_c_u[0])
        card_s = len(self.ps_x_u[0])
        card_y = len(self.py_s[0])
        samples = np.zeros((n, 6))
        for i in range(n):
            u = np.random.choice(np.arange(0, card_u), p=self.pu)
            t = np.random.choice(np.arange(0, card_t), p=self.pt)
            c = np.random.choice(np.arange(0, card_c), p=self.pc)
            x = np.random.choice(np.arange(0, card_x), p=self.px_c_u[c*card_u+u])
            s = np.random.choice(np.arange(0, card_s), p=self.ps_x_u[x*card_u+u])
            y = np.random.choice(np.arange(0, card_y), p=self.py_s[s])
            samples[i, :] = np.array([u, c, x, s, y, t])
        return samples

    def s_dist(self):
        card_u = len(self.pu)
        p = np.zeros(len(self.ps_x_u[0]))
        for s in range(len(self.ps_x_u[0])):
            for c in range(len(self.pc)):
                for u in range(card_u):
                    for x in range(len(self.px_c_u[0])):
                        p[s] += self.ps_x_u[x*card_u+u][s] * self.px_c_u[c*card_u+u][x] * self.pc[c] * self.pu[u]
        return p

    def y_dist(self):
        ps2 = self.s_dist()
        p = np.zeros(len(self.py_s[0]))
        for y in range(len(self.py_s[0])):
            for s in range(len(self.ps_x_u[0])):
                p[y] += ps2[s] * self.py_s[s][y]
            p[y] = p[y] * self.pt[0] + self.y_t1[y] * self.pt[1]
        return p

    def s_do_pi(self, pi):
        p = np.zeros(len(self.ps_x_u[0]))
        for s in range(len(self.ps_x_u[0])):
            for x in range(len(self.px_c_u[0])):
                for c in range(len(self.pc)):
                    ps_dox = self.pu[1-(s ^ x)]
                    p[s] += ps_dox * pi[c][x] * self.pc[c]
        return p

    def y_do_pi(self, pi):
        psdopi = self.s_do_pi(pi)
        p = np.zeros(len(self.py_s[0]))
        for y in range(len(self.py_s[0])):
            for s in range(len(self.ps_x_u[0])):
                p[y] += psdopi[s] * self.py_s[s][y]
            p[y] = p[y] * self.pt[0] + self.y_t1[y] * self.pt[1]
        return p

    def x_dist(self):
        card_u = len(self.pu)
        p = np.zeros(len(self.px_c_u[0]))
        for x in range(len(self.px_c_u[0])):
            for c in range(len(self.pc)):
                for u in range(card_u):
                    p[x] += self.px_c_u[c*card_u+u][x] * self.pc[c] * self.pu[u]
        return p

    def x_given_c(self):
        card_u = len(self.pu)
        card_c = len(self.pc)
        card_x = len(self.px_c_u[0])
        p = np.zeros((card_c, card_x))
        for c in range(card_c):
            for x in range(card_x):
                for u in range(card_u):
                    p[c][x] += self.px_c_u[c*card_u+u][x] * self.pu[u]
        return p


