import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import math
import pickle


def retrieve_adjacency_matrix(graph, order_nodes=None, weight=False):
    """Retrieve the adjacency matrix from the nx.DiGraph or numpy array."""
    if isinstance(graph, np.ndarray):
        return graph
    elif isinstance(graph, nx.DiGraph):
        if order_nodes is None:
            order_nodes = graph.nodes()
        if not weight:
            return np.array(nx.adjacency_matrix(graph, order_nodes, weight=None).todense())
        else:
            return np.array(nx.adjacency_matrix(graph, order_nodes).todense())
    else:
        raise TypeError("Only networkx.DiGraph and np.ndarray (adjacency matrixes) are supported.")


def SHD(target, pred, double_for_anticausal=True):
    r"""Compute the Structural Hamming Distance.
    The Structural Hamming Distance (SHD) is a standard distance to compare
    graphs by their adjacency matrix. It consists in computing the difference
    between the two (binary) adjacency matrixes: every edge that is either
    missing or not in the target graph is counted as a mistake. Note that
    for directed graph, two mistakes can be counted as the edge in the wrong
    direction is false and the edge in the good direction is missing ; the
    `double_for_anticausal` argument accounts for this remark. Setting it to
    `False` will count this as a single mistake.
    Args:
        target (numpy.ndarray or networkx.DiGraph): Target graph, must be of
            ones and zeros.
        prediction (numpy.ndarray or networkx.DiGraph): Prediction made by the
            algorithm to evaluate.
        double_for_anticausal (bool): Count the badly oriented edges as two
            mistakes. Default: True

    Returns:
        int: Structural Hamming Distance (int).
            The value tends to zero as the graphs tend to be identical.
    Examples:
    """

    true_labels = retrieve_adjacency_matrix(target)
    predictions = retrieve_adjacency_matrix(pred, target.nodes()
    if isinstance(target, nx.DiGraph) else None)

    diff = np.abs(true_labels - predictions)
    if double_for_anticausal:
        return np.sum(diff)
    else:
        diff = diff + diff.transpose()
        diff[diff > 1] = 1  # Ignoring the double edges.
        return np.sum(diff) / 2


def set_graph_comparison(set_graphs):
    matrix = np.full((len(set_graphs), len(set_graphs)), -1)

    for i in range(0, len(set_graphs)):
        for j in range(i+1, len(set_graphs)):
            matrix[i, j] = SHD(set_graphs[i], set_graphs[j])

    return matrix


def x_coord_of_point(D, j):
    return (D[0, j] ** 2 + D[0, 1] ** 2 - D[1, j] ** 2) / (2 * D[0, 1])


def coords_of_point(D, j):
    x = x_coord_of_point(D, j)
    return np.array([x, math.sqrt(D[0, j] ** 2 - x ** 2)])


def calculate_positions(D):
    (m, n) = D.shape
    P = np.zeros((n, 2))
    tr = (min(min(D[2, 0:2]), min(D[2, 3:n])) / 2) ** 2
    P[1, 0] = D[0, 1]
    P[2, :] = coords_of_point(D, 2)
    for j in range(3, n):
        P[j, :] = coords_of_point(D, j)
        if abs(np.dot(P[j, :] - P[2, :], P[j, :] - P[2, :]) - D[2, j] ** 2) > tr:
            P[j, 1] = - P[j, 1]
    return P


def draw_set_graphs(set1, set2):
    total_set = set1 + set2
    matrix = set_graph_comparison(total_set)
    coord = calculate_positions(matrix)
    print(coord)

    for i in range(len(set1)):
        plt.scatter(coord[i][0], coord[i][1], color='blue', label='set1')
    for i in range(len(set1), len(total_set)):
        plt.scatter(coord[i][0], coord[i][1], color='red', label='set2')

    plt.show()


"""graph1_1 = np.array([[0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
graph1_2 = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
graph1_3 = np.array([[0, 1, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

graph2_1 = np.array([[0, 1, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
graph2_2 = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]])
graph2_3 = np.array([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]])"""


def load_adj(list_tuples):
    matrix = np.zeros((30, 30))
    for tuple in list_tuples:
        matrix[int(tuple[0]), int(tuple[1])] = 1
    return matrix


set1_graphs = []
for i in range(2, 10):
    infile = open('aux_' + str(i) + '.pickle', 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    set1_graphs.append(load_adj(new_dict))

set2_graphs = []
for i in range(10, 20):
    infile = open('aux_' + str(i) + '.pickle', 'rb')
    new_dict = pickle.load(infile)
    infile.close()
    set2_graphs.append(load_adj(new_dict))

draw_set_graphs(set1_graphs, set2_graphs)

# draw_set_graphs([graph1_1, graph1_2, graph1_3], [graph2_1, graph2_2, graph2_3])

