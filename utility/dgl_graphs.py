#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 12/27/2020
dgl_graphs.
@author: Kang Xiatao (799802172@qq.com)
"""

#
import os
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info


def save(self, data_dir):
    # save graphs
    graph_path = os.path.join(data_dir, 'kgat_dgl_graph.bin')
    save_graphs(graph_path, self.train_graph)
    # save other information in python dict
    # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    # save_info(info_path, {'num_classes': self.num_classes})


def load(self, data_dir):
    # load processed data from directory `self.save_path`
    graph_path = os.path.join(data_dir, 'kgat_dgl_graph.bin')
    self.train_graph, label_dict = load_graphs(graph_path)
    # self.labels = label_dict['labels']
    # info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
    # self.num_classes = load_info(info_path)['num_classes']


# def has_cache(self):
#     # check whether there are processed data in `self.save_path`
#     graph_path = os.path.join(self.save_path, self.mode + '_dgl_graph.bin')
#     info_path = os.path.join(self.save_path, self.mode + '_info.pkl')
#     return os.path.exists(graph_path) and os.path.exists(info_path)
