from utils.imitable import imitable
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager


"""
    Generate random graphs and check imitability.
    Run this snippet to reproduce results of Figure 4.
"""


if __name__ == '__main__':
    samples = 100
    points = 10
    data1 = np.zeros([points, samples])
    datacsi = np.zeros([points, samples])
    for i in range(1, points+1):
        n = i*20  # number of vertices
        max_d = int(n / 10)
        p = (np.array(range(max_d + 1)) + 1) / sum(np.array(range(max_d + 1)) + 1)
        for s in range(samples):
            mat = np.zeros([n, n])
            for j in range(n):
                deg = np.random.choice(np.arange(0, max_d+1), p=p)
                idx = np.random.choice(range(j+1, n), min(n-1-j, np.random.randint(max_d)), replace=False)
                if len(idx) > 0:
                    mat[j][idx] = 1
            mat = np.triu(mat, 1)
            g = nx.DiGraph(mat)

            # nx.draw(g, with_labels=True)
            #g = nx.DiGraph()

            nodes = list(nx.topological_sort(g))
            Y = nodes[-1]
            anc = nx.ancestors(g, Y)
            X = nodes[int(len(anc)/2)]
            out1, out2 = imitable(g, X, Y)
            data1[i-1][s] = out1
            datacsi[i - 1][s] = out2
    mu1 = np.mean(data1, 1)
    error1 = 3 * np.std(data1, 1) / np.sqrt(samples)
    mu2 = np.mean(datacsi, 1)
    error2 = 3 * np.std(datacsi, 1) / np.sqrt(samples)
    fig, ax = plt.subplots()
    xvals = range(20, 20*(1+points), 20)
    ax.errorbar(xvals, mu1, yerr=error1, marker='^', label="Imitable",
                color='black', markersize=5, mew=1)
    ax.errorbar(xvals, mu2, yerr=error2, marker='o', label="Imitable under CSIs",
                color='red', markersize=5, mew=1)

    ax.set_xlabel("Number of vertices", fontname="Times New Roman", fontsize=27)
    ax.set_ylabel("Fraction of imitable instances", fontname="Times New Roman", fontsize=27)
    ax.set_ylim(bottom=0, top=1.1)
    ax.set_xlim(left=19, right=max(xvals))
    plt.xticks(fontname="Times New Roman", fontsize=19)
    plt.yticks(fontname="Times New Roman", fontsize=19)
    # plt.plot(range(10, 110, 10), np.mean(data1, 1))
    # plt.plot(range(10, 110, 10), np.mean(datacsi, 1))
    font = font_manager.FontProperties(family='Times New Roman',
                                       # weight='bold',
                                       # style='normal',
                                       size=20)
    ax.legend(loc="lower left", numpoints=1, prop=font)
    ax.grid(True, alpha=0.3)
    fig.set_size_inches(8, 6)
    plt.show()
    # plt.savefig('fraction.pdf')
            # nx.draw(g, with_labels=True)