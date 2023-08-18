import sys
import numpy as np

sys.path.extend(['../'])
from graph import tools

num_node = 21
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 7), (3, 8),
                    (4, 9), (5, 10), (6, 11), (7, 12), (8, 13), (9, 14),
                    (10, 15), (11, 16), (12, 17), (13, 18), (14, 19), (15, 20),
                    (16, 21)]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

class Graph:
    def __init__(self, labeling_mode='spatial'):
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.A = self.get_adjacency_matrix(labeling_mode)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        else:
            raise ValueError()
        return A
