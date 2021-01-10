#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 12/26/2020
test.
@author: Kang Xiatao (799802172@qq.com)
"""

#

# import numpy
#
# a = numpy.load('trained_model\\NFM\\amazon-book\\nfm_embeddim64_64_lr0.0001_pretrain1\\cf_scores.npy')
#
# print(len(a))
# print(a.shape)


import os
import dgl
import torch
import numpy as np
import pandas as pd

def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
        5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
        24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
        29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
        31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # print((u, v))
    # Construct a DGLGraph
    return dgl.graph((u, v))

# G = build_karate_club_graph()
# print('We have %d nodes.' % G.number_of_nodes())
# print('We have %d edges.' % G.number_of_edges())


def load_kg(filename):
    kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python', nrows=10)
    kg_data = kg_data.drop_duplicates()
    return kg_data


def create_graph(kg_data, n_nodes):
    g = dgl.graph((kg_data['t'], kg_data['h']))
    g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)
    g.edata['type'] = torch.LongTensor(kg_data['r'])
    return g


# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# data_dir = os.path.join('datasets/', 'amazon-book')
# kg_file = os.path.join(data_dir, "kg_final.txt")
#
# kg_data = load_kg(kg_file)
# # print(kg_data)
#
# n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
# kg_graph = create_graph(kg_data, n_entities)

# print(kg_graph)
# print(kg_graph.device)
# cuda_kg_graph = kg_graph.to(device)  # accepts any device objects from backend framework
# print(cuda_kg_graph.device)


# train_graph = kg_graph.to(device)

# train_nodes = torch.LongTensor(train_graph.ndata['id'])
# train_edges = torch.LongTensor(train_graph.edata['type'])
# train_graph.ndata['id'] = train_nodes
# train_graph.edata['type'] = train_edges

# print(train_graph.ndata['id'])
# print(train_graph.edata['type'])

# from dgl import save_graphs, load_graphs
#
# graph_path = os.path.join('kgat_dgl_graph.bin')
# save_graphs(graph_path, train_graph)


# from tqdm import tqdm, trange
# import time
#
# for i in tqdm(range(100), desc='Evaluating Iteration'):
#   time.sleep(0.1)
#   pass


# import torch as th
# import dgl
#
# g = dgl.DGLGraph()
# g.add_edges([0, 1, 2], [2, 2, 1])
# g.ndata['x'] = th.tensor([[0], [1], [2]])
# print(g)
#
# def has_dst_one(edges): return (edges.dst['x'] == 2).squeeze(1)
#
# print(g.filter_edges(has_dst_one))

# g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])))
# g.ndata['h'] = torch.ones(3, 1)
# print(g)

print(np.random.randint(low=0, high=10, size=1))
