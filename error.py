#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Created on 01/07/2021
error.
@author: Kang Xiatao (799802172@qq.com)
"""

"""
Traceback (most recent call last):
  File "D:/MyCode/Project_Python/PaperTest/KGAT-pytorch-master/main_kgat.py", line 270, in <module>
    train(args)
  File "D:/MyCode/Project_Python/PaperTest/KGAT-pytorch-master/main_kgat.py", line 191, in train
    _, precision, recall, ndcg = evaluate(model, train_graph, data.train_user_dict, data.test_user_dict, user_ids_batches, item_ids, args.K)
  File "D:/MyCode/Project_Python/PaperTest/KGAT-pytorch-master/main_kgat.py", line 54, in evaluate
    cf_scores = np.concatenate(cf_scores, axis=0)  #  (70591, 24915)
  File "<__array_function__ internals>", line 5, in concatenate
MemoryError: Unable to allocate 6.55 GiB for an array with shape (70591, 24915) and data type float32
"""


