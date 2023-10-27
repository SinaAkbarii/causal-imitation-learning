import networkx as nx
import numpy as np


def roots(g):
    """
    :param g: a networkx graph
    :return: the set of nodes of g that have no parents
    """
    return [r for r in g.nodes if len(list(g.predecessors(r))) == 0]


def pi_exist(g, X, Y, unobs):
    """
    determines whether or not an imitation policy exists through \pi-backdoor criterion
    :param g: networkx graph
    :param X: the action variable
    :param Y: the response (reward) variable
    :param unobs: the unobserved variables
    :return: whether or not an imitation policy exists
    """
    desc = {X}.union(nx.descendants(g, X))
    ancestors = set(nx.ancestors(g, Y)).union(nx.ancestors(g, X)).difference(desc)
    Z = ancestors.difference(unobs)
    k = g.copy()
    out_x = set(k.out_edges(X))
    k.remove_edges_from(out_x)
    return nx.d_separated(k, {X}, {Y}, Z)


def imitable(g, X, Y):
    """
    determines imitability
    :param g: networkx graph
    :param X: action variable
    :param Y: response (reward) variable
    :return: whether or not the instance is imitable
    """
    C = list(set(roots(g)).difference([X]))
    num_cxt = min(3, len(C))  # np.random.randint(len(C))
    CV = np.random.choice(C, num_cxt, replace=False)
    cxt_child = set(sum([list(g.successors(v)) for v in CV], []))
    cxt_coparents = set(sum([list(g.predecessors(v)) for v in cxt_child], []))
    left_roots = cxt_coparents.difference(CV)
    n = len(g.nodes)
    num_hidden = min(max(5, int(n/5)), len(left_roots))
    if num_hidden == 0:
        hiddens = []
    else:
        hiddens = np.random.choice(list(left_roots), num_hidden, replace=False)
    is_g_imitable = pi_exist(g, X, Y, hiddens)
    is_g_csi_imitable = True
    if not is_g_imitable:
        for _ in range(2 ** len(CV)):  # realizations of CV
            h = g.copy()
            removable_edges = sum([list(h.out_edges(v)) for v in set(hiddens).intersection(h.predecessors(X))], [])
            num_remove = min(int(len(removable_edges)/2+4), len(removable_edges))
            edge_to_removeidx = np.random.choice(range(len(removable_edges)), num_remove, replace=False)
            edge_to_remove = np.array(removable_edges)[edge_to_removeidx]
            h.remove_edges_from(edge_to_remove)
            if not pi_exist(h, X, Y, hiddens):
                is_g_csi_imitable = False
                break

    return is_g_imitable, is_g_csi_imitable