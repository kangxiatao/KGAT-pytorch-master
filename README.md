# Knowledge Graph Attention Network

这是本文的PyTorch和DGL实现：
>Xiang Wang, Xiangnan He, Yixin Cao, Meng Liu and Tat-Seng Chua (2019). KGAT: Knowledge Graph Attention Network for Recommendation. [Paper in ACM DL](https://dl.acm.org/authorize.cfm?key=N688414) or [Paper in arXiv](https://arxiv.org/abs/1905.07854). In KDD'19, Anchorage, Alaska, USA, August 4-8, 2019.

你可以点击 [这里](https://github.com/xiangwang1223/knowledge_graph_attention_network) 找到论文作者的Tensorflow1.0的实现。
PyTorch和DGL实现代码改自[这里](https://github.com/LunaBlack/KGAT-pytorch) 。

代码优点：
 * 整体框架搭建得很好，之后做实验也可以借鉴这个框架
 * 编程风格好，结构清晰，大致可分为参数、数据和模型三部分
 * 借用argparse编写了相对友好的命令行设置参数的接口
 * logging日志模块输出信息到控制台和文件
 * 在某个epoch达到一定效果时保存模型
 
代码缺点：
 * 大量重复的代码，显得臃肿
 * 测试集评估时，因为底层一些函数原因，把gpu中的张量放到cpu中执行，效率低
 * 一次性读取数据集，大量的全局变量，对内存和显存都不友好

## 介绍

知识图注意力网络（KGAT）是专门针对知识感知的个性化推荐量身定制的新推荐框架。KGAT建立在图神经网络框架之上，对协作知识图中的高阶关系进行了明确的建模，以提供更好的项目侧信息推荐。

如果您想在研究中使用代码和数据集，请联系论文作者，并引用以下论文作为参考：
```
@inproceedings{KGAT19,
  author    = {Xiang Wang and
               Xiangnan He and
               Yixin Cao and
               Meng Liu and
               Tat{-}Seng Chua},
  title     = {{KGAT:} Knowledge Graph Attention Network for Recommendation},
  booktitle = {{KDD}},
  pages     = {950--958},
  year      = {2019}
}
```
代码中做了大量注释，如果对本代码有疑问，请联系我：
```
@author: Kang Xiatao (kangxiatao@gmail.com)
```

## 环境

该代码已经过测试，可以在Python 3.8.5下运行。

所需的软件包如下：
* torch == 1.7.1
* dgl-cu101 == 0.5.3
* numpy == 1.18.5
* pandas == 1.1.3
* sklearn == 0.23.2

## 运行

* FM
```
python main_nfm.py --model_type fm --dataset amazon-book
```
* NFM
```
python main_nfm.py --model_type nfm --dataset amazon-book
```
* KGAT
```
python main_kgat.py --dataset amazon-book
```
## 数据集

作者提供了三个数据集：Amazon-book, Last-FM, and Yelp2018.

我爬取了豆瓣Top250制作了一个数据集：douban250

* 作者用的是公开数据集，你能在这里找到完整的数据集 [Amazon-book](http://jmcauley.ucsd.edu/data/amazon), [Last-FM](http://www.cp.jku.at/datasets/LFM-1b/), [Yelp2018](https://www.yelp.com/dataset/challenge).
* 豆瓣的数据集可能因为过于稀疏的原因，训练效果很差，有时间再重新制作整理一次

| | | Amazon-book | Last-FM | Yelp2018 | douban250 |
|:---:|:---|---:|---:|---:|---:|
|User-Item Interaction| #Users | 70,679 | 23,566 | 45,919| 4,422 |
| | #Items | 24,915 | 48,123 | 45,538| 250 |
| | #Interactions | 847,733 | 3,034,796 | 1,185,068| 55000 |
|Knowledge Graph | #Entities | 88,572 | 58,266 | 90,961| None |
| | #Relations | 39 | 9 | 42 | None |
| | #Triplets | 2,557,746 | 464,567 | 1,853,704| None |


## 结果

原代码中用到了多GPU训练，因条件受限，BPRMF,ECFKG,CKE暂时没有测试结果

* `amazon-book`数据集：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| FM    | sample 1000 test users | 65         | 0.014400000683963299 | 0.14490722119808197 | 0.07222465868746986 |
| NFM   | sample 1000 test users | 52         | 0.013500000350177288 | 0.13786590099334717 | 0.07123670123284831 |
| KGAT  | all test users         | 39         | 0.014916915994618973 | 0.1414667212776353  | 0.07478134080605618 |

* `last-fm`数据集：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| FM    | sample 1000 test users | 39         | 0.03400000184774399  | 0.0831719189882278  | 0.06556513877651045 |
| NFM   | sample 1000 test users | 65         | 0.03230000287294388  | 0.0825699120759964  | 0.06412929073269483 |
| KGAT  | all test users         | 82         | 0.03326826841464287  | 0.08198051536362484 | 0.07016461076103524 |

* `yelp2018`数据集：

| Model | Valid Data             | Best Epoch | Precision@20         | Recall@20           | NDCG@20             |
| :---: | :---                   | :---:      | :---:                | :---:               | :---:               |
| FM    | sample 1000 test users | 19         | 0.016450000926852226 | 0.06791889667510986 | 0.04011075919416859 |
| NFM   | sample 1000 test users | 17         | 0.014950000680983067 | 0.0635601356625557  | 0.03876655643191971 |
| KGAT  | all test users         | 16         | 0.016048173102794366 | 0.06584655151793856 | 0.04193551918102937 |

## 相关论文

* KGAT
    * 提出了 [KGAT: Knowledge Graph Attention Network for Recommendation](https://arxiv.org/abs/1905.07854), KDD2019.
    * 论文作者的实现：[https://github.com/xiangwang1223/knowledge_graph_attention_network](https://github.com/xiangwang1223/knowledge_graph_attention_network)
    * 关键点:
        * 在协作知识图中对高阶关系进行建模，以提供带有项边信息的更好推荐。
        * 依次训练KG部分和CF部分。
        


