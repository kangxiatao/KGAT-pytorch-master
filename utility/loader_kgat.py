import os
import time
import random
import collections

import dgl
import torch
import numpy as np
import pandas as pd

from utility import dgl_graphs


"""
    数据集一共有三个文件（项目是属于实体的）
    - train.txt
        - 用户和项目的交互
        - 第一项为用户，后面为用户交互过的项目
    - test.txt 
        - 同train.txt
    - kg_final.txt
        - 实体和实体的关系
        - 数据对应为：
            - 实体 关系 实体
            
    amazon book 的数据集中
        用户数 n_users: 70679
        项目数 n_items: 24915
        实体数 n_entities: 113487
        关系数 n_relations: 80
"""

class DataLoaderKGAT(object):

    def __init__(self, args, logging):
        self.args = args
        self.data_name = args.data_name  # 数据集名
        self.use_pretrain = args.use_pretrain  # 是否用预处理
        self.pretrain_embedding_dir = args.pretrain_embedding_dir  # 预先通过embedding的数据路径

        self.cf_batch_size = args.cf_batch_size  # 一批cf训练数据的大小
        self.kg_batch_size = args.kg_batch_size  # 一批kg训练数据的大小

        # 获取数据文件路径
        data_dir = os.path.join(args.data_dir, args.data_name)
        train_file = os.path.join(data_dir, 'train.txt')
        test_file = os.path.join(data_dir, 'test.txt')
        kg_file = os.path.join(data_dir, "kg_final.txt")
        print('--', data_dir, '--')

        # 获取数据
        """ 
        self.load_cf:
            第一个返回值为一个元组，里面为用户和项目的两个numpy array，长度相等，它们保持对应关系
                - self.cf_train_data[0][i] 和 self.cf_train_data[1][i] 他们的值表示用户id和项目id，同时这两个有交互关系
            第二个返回值是一个字典，就是把train.txt文件中的关系用字典保存
        """
        self.cf_train_data, self.train_user_dict = self.load_cf(train_file)
        self.cf_test_data, self.test_user_dict = self.load_cf(test_file)
        self.statistic_cf()  # 得到数据的大小长度等
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '-- cf data finish --')

        """ 
        self.load_kg:
            返回的是pandas三元组储存格式，实体-关系-实体
        """
        kg_data = self.load_kg(kg_file)
        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '-- kg data load --')

        # 创建知识图预处理
        self.construct_data(kg_data)

        # 保存读取出来是list，也没有加快速度
        # if args.use_graph == 0:
        #     self.train_graph = self.create_graph(self.kg_train_data, self.n_users_entities)
        #     dgl_graphs.save(self, data_dir)
        #     print('-- save train_graph --')
        # else:
        #     dgl_graphs.load(self, data_dir)
        #     print('-- load train_graph --')

        # 用dgl创建知识图
        # 这一块创建知识图需要十几分钟的时间，目前来说好像通过list批量构建是最快的
        self.train_graph = self.create_graph(self.kg_train_data, self.n_users_entities)
        # self.test_graph = self.create_graph(self.kg_test_data, self.n_users_entities)  # test_graph not use

        print(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()), '-- kg data finish --')

        if self.use_pretrain == 1:
            self.load_pretrained_data()

        self.print_info(logging)

    def load_cf(self, filename):
        user = []
        item = []
        user_dict = dict()

        lines = open(filename, 'r').readlines()
        for l in lines:
            tmp = l.strip()
            inter = [int(i) for i in tmp.split()]

            if len(inter) > 1:
                user_id, item_ids = inter[0], inter[1:]
                item_ids = list(set(item_ids))

                # 这里user和item提取出关系放在list中保持对应关系
                # user_dict作为字典重新保持关系
                for item_id in item_ids:
                    user.append(user_id)
                    item.append(item_id)
                user_dict[user_id] = item_ids

        user = np.array(user, dtype=np.int32)
        item = np.array(item, dtype=np.int32)
        return (user, item), user_dict

    def statistic_cf(self):
        self.n_users = max(max(self.cf_train_data[0]), max(self.cf_test_data[0])) + 1
        self.n_items = max(max(self.cf_train_data[1]), max(self.cf_test_data[1])) + 1
        self.n_cf_train = len(self.cf_train_data[0])
        self.n_cf_test = len(self.cf_test_data[0])

    def load_kg(self, filename):
        kg_data = pd.read_csv(filename, sep=' ', names=['h', 'r', 't'], engine='python')
        kg_data = kg_data.drop_duplicates()  # 去除重复项
        return kg_data

    def construct_data(self, kg_data):
        # plus inverse kg data  相当于做成无向图
        n_relations = max(kg_data['r']) + 1
        reverse_kg_data = kg_data.copy()
        reverse_kg_data = reverse_kg_data.rename({'h': 't', 't': 'h'}, axis='columns')
        # print(reverse_kg_data)  # t r h [2557746 rows x 3 columns]
        reverse_kg_data['r'] += n_relations
        # print(reverse_kg_data)  # t r h [2557746 rows x 3 columns]
        kg_data = pd.concat([kg_data, reverse_kg_data], axis=0, ignore_index=True, sort=False)
        # print(kg_data)  # h r t [5115492 rows x 3 columns]

        # re-map user id
        kg_data['r'] += 2  # 后面还需要添加两种关系
        self.n_relations = max(kg_data['r']) + 1
        self.n_entities = max(max(kg_data['h']), max(kg_data['t'])) + 1
        self.n_users_entities = self.n_users + self.n_entities

        # 把cf_data中的用户id加n_entities，再用numpy array形式储存
        self.cf_train_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_train_data[0]))).astype(np.int32), self.cf_train_data[1].astype(np.int32))
        self.cf_test_data = (np.array(list(map(lambda d: d + self.n_entities, self.cf_test_data[0]))).astype(np.int32), self.cf_test_data[1].astype(np.int32))

        # 把dict中的list转换为数组，用户id加n_entities，np.unique()去重排序
        self.train_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.train_user_dict.items()}
        self.test_user_dict = {k + self.n_entities: np.unique(v).astype(np.int32) for k, v in self.test_user_dict.items()}

        # add interactions to kg data
        cf2kg_train_data = pd.DataFrame(np.zeros((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        cf2kg_train_data['h'] = self.cf_train_data[0]
        cf2kg_train_data['t'] = self.cf_train_data[1]

        reverse_cf2kg_train_data = pd.DataFrame(np.ones((self.n_cf_train, 3), dtype=np.int32), columns=['h', 'r', 't'])
        reverse_cf2kg_train_data['h'] = self.cf_train_data[1]
        reverse_cf2kg_train_data['t'] = self.cf_train_data[0]

        # 测试集数据并不用再建一个知识图
        # cf2kg_test_data = pd.DataFrame(np.zeros((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        # cf2kg_test_data['h'] = self.cf_test_data[0]
        # cf2kg_test_data['t'] = self.cf_test_data[1]
        #
        # reverse_cf2kg_test_data = pd.DataFrame(np.ones((self.n_cf_test, 3), dtype=np.int32), columns=['h', 'r', 't'])
        # reverse_cf2kg_test_data['h'] = self.cf_test_data[1]
        # reverse_cf2kg_test_data['t'] = self.cf_test_data[0]

        # 基于CKG建模
        # kg_data: 关系的无向图三元组 [5115492, 3]
        # cf2kg_train_data: 关系值为零的用户和项目三元组 [None, 3]
        # reverse_cf2kg_train_data: 关系值为一的项目和用户三元组 [None, 3]
        self.kg_train_data = pd.concat([kg_data, cf2kg_train_data, reverse_cf2kg_train_data], ignore_index=True)
        # self.kg_test_data = pd.concat([kg_data, cf2kg_test_data, reverse_cf2kg_test_data], ignore_index=True)

        self.n_kg_train = len(self.kg_train_data)
        # self.n_kg_test = len(self.kg_test_data)

        # construct kg dict
        self.train_kg_dict = collections.defaultdict(list)  # 便于查询，不存在时不会报错
        # self.train_relation_dict = collections.defaultdict(list)
        for row in self.kg_train_data.iterrows():  # 遍历行数据,index, row
            h, r, t = row[1]
            self.train_kg_dict[h].append((t, r))
            # self.train_relation_dict[r].append((h, t))

        # self.test_kg_dict = collections.defaultdict(list)
        # self.test_relation_dict = collections.defaultdict(list)
        # for row in self.kg_test_data.iterrows():
        #     h, r, t = row[1]
        #     self.test_kg_dict[h].append((t, r))
        #     self.test_relation_dict[r].append((h, t))

    def print_info(self, logging):
        logging.info('n_users:            %d' % self.n_users)
        logging.info('n_items:            %d' % self.n_items)
        logging.info('n_entities:         %d' % self.n_entities)
        logging.info('n_users_entities:   %d' % self.n_users_entities)
        logging.info('n_relations:        %d' % self.n_relations)

        logging.info('n_cf_train:         %d' % self.n_cf_train)
        logging.info('n_cf_test:          %d' % self.n_cf_test)

        logging.info('n_kg_train:         %d' % self.n_kg_train)
        # logging.info('n_kg_test:          %d' % self.n_kg_test)

    def create_graph(self, kg_data, n_nodes):
        # g = dgl.DGLGraph()
        # g.add_nodes(n_nodes)
        # g.add_edges(kg_data['t'], kg_data['h'])
        # g.readonly()

        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # g = dgl.graph((kg_data['t'], kg_data['h']), device=device)

        # amazon-book中有六百多万个节点，80种关系
        g = dgl.graph((kg_data['t'], kg_data['h']))
        g.ndata['id'] = torch.arange(n_nodes, dtype=torch.long)  # 节点
        g.edata['type'] = torch.LongTensor(kg_data['r'])  # 边
        return g

    def sample_pos_items_for_u(self, user_dict, user_id, n_sample_pos_items):
        pos_items = user_dict[user_id]
        n_pos_items = len(pos_items)

        sample_pos_items = []
        while True:
            if len(sample_pos_items) == n_sample_pos_items:
                break

            pos_item_idx = np.random.randint(low=0, high=n_pos_items, size=1)[0]
            pos_item_id = pos_items[pos_item_idx]
            if pos_item_id not in sample_pos_items:
                sample_pos_items.append(pos_item_id)
        return sample_pos_items

    def sample_neg_items_for_u(self, user_dict, user_id, n_sample_neg_items):
        pos_items = user_dict[user_id]

        sample_neg_items = []
        while True:
            if len(sample_neg_items) == n_sample_neg_items:
                break

            neg_item_id = np.random.randint(low=0, high=self.n_items, size=1)[0]
            if neg_item_id not in pos_items and neg_item_id not in sample_neg_items:
                sample_neg_items.append(neg_item_id)
        return sample_neg_items

    """
    返回参数：
        1，一批用户的id
        2，正面的项目id（用户与该项目产生了交互）
        3，反面的项目id（用户与该项目没有交互）
    """
    def generate_cf_batch(self, user_dict):
        exist_users = user_dict.keys()
        if self.cf_batch_size <= len(exist_users):
            batch_user = random.sample(exist_users, self.cf_batch_size)
        else:
            batch_user = [random.choice(exist_users) for _ in range(self.cf_batch_size)]

        batch_pos_item, batch_neg_item = [], []
        for u in batch_user:
            batch_pos_item += self.sample_pos_items_for_u(user_dict, u, 1)
            batch_neg_item += self.sample_neg_items_for_u(user_dict, u, 1)

        batch_user = torch.LongTensor(batch_user)  # [cf_batch_size]
        batch_pos_item = torch.LongTensor(batch_pos_item)
        batch_neg_item = torch.LongTensor(batch_neg_item)
        return batch_user, batch_pos_item, batch_neg_item

    def sample_pos_triples_for_h(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_h(self, kg_dict, head, relation, n_sample_neg_triples):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=self.n_users_entities, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    """
    返回参数：
        1，一批实体的id
        2，一批关系的id
        3，正面的实体id（实体之间产生了交互）
        4，反面的实体id（实体之间没有交互）
    """
    def generate_kg_batch(self, kg_dict):
        exist_heads = kg_dict.keys()
        if self.kg_batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, self.kg_batch_size)
        else:
            batch_head = [random.choice(exist_heads) for _ in range(self.kg_batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []
        for h in batch_head:
            relation, pos_tail = self.sample_pos_triples_for_h(kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            neg_tail = self.sample_neg_triples_for_h(kg_dict, h, relation[0], 1)
            batch_neg_tail += neg_tail

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def load_pretrained_data(self):
        pre_model = 'mf'
        pretrain_path = '%s/%s/%s.npz' % (self.pretrain_embedding_dir, self.data_name, pre_model)
        pretrain_data = np.load(pretrain_path)
        self.user_pre_embed = pretrain_data['user_embed']
        self.item_pre_embed = pretrain_data['item_embed']

        assert self.user_pre_embed.shape[0] == self.n_users
        assert self.item_pre_embed.shape[0] == self.n_items
        assert self.user_pre_embed.shape[1] == self.args.entity_dim
        assert self.item_pre_embed.shape[1] == self.args.entity_dim


