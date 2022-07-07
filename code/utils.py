import torch
import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from sklearn import preprocessing
import random
from torch_geometric.utils import dropout_adj
from sklearn.metrics import f1_score
import copy
from collections import defaultdict, Counter

'''
----------------------------- data processing -----------------------------
'''

valid_num_dic = {'Amazon_clothing': 17, 'Amazon_eletronics': 36, 'dblp': 27}

def load_data(dataset_source):
    n1s = []
    n2s = []
    for line in open("../few_shot_data/{}_network".format(dataset_source)):
        n1, n2 = line.strip().split('\t')
        n1s.append(int(n1))
        n2s.append(int(n2))

    edge_index = torch.LongTensor([n1s, n2s])                     # [2, 183360]

    num_nodes = max(max(n1s),max(n2s)) + 1
    # print('num_nodes', num_nodes)                               # 24919
    # print('len(n1s)', len(n1s))                                 # 边的数量 183360 包含两条有向边 一条无向边则为91680
    adj = sp.coo_matrix((np.ones(len(n1s)), (n1s, n2s)),          # np.ones(len(n1s)): entries
                                 shape=(num_nodes, num_nodes))    # coo_matrix((data, (i, j)), [shape=(M, N)]) A[i[k], j[k]] = data[k]


    data_train = sio.loadmat("../few_shot_data/{}_train.mat".format(dataset_source))
    # print(type(data_train), data_train.keys())                    # 'Index', 'Attributes', 'Label'
    train_class = list(set(data_train["Label"].reshape((1,len(data_train["Label"])))[0]))
    # print(type(train_class), len(train_class))                    # list 57
    # print(train_class[:10])

    data_test = sio.loadmat("../few_shot_data/{}_test.mat".format(dataset_source))
    class_list_test = list(set(data_test["Label"].reshape((1,len(data_test["Label"])))[0]))
    # print(type(class_list_test), len(class_list_test))            # list 20 共77

    # print()
    # print(type(data_train['Index']), data_train['Index'][:10])
    # print(type(data_test['Label']), data_test['Label'][:10])
    labels = np.zeros((num_nodes,1))
    labels[data_train['Index']] = data_train["Label"]                  # 训练数据和测试全都放到labels中
    labels[data_test['Index']] = data_test["Label"]
    # print('labels', labels[:10])

    features = np.zeros((num_nodes,data_train["Attributes"].shape[1])) # shape 24919, 9034
    features[data_train['Index']] = data_train["Attributes"].toarray() # features也是训练和测试数据
    features[data_test['Index']] = data_test["Attributes"].toarray()

    class_list = []
    for cla in labels:
        if cla[0] not in class_list:
            class_list.append(cla[0])    # unsorted
    # print(len(class_list))               # 77 包含所有的类

    id_by_class = {}
    for i in class_list:
        id_by_class[int(i)] = []
    for id, cla in enumerate(labels):
        id_by_class[int(cla[0])].append(id)
    # print(id_by_class)                 # 77个keys，keys是类，values是所有这个类的节点的id的list
    
    # output
    # for (k, v) in id_by_class.items():
    #     print(k, len(v))

    class_by_id = torch.Tensor([-1 for i in range(num_nodes)]).cuda()
    for c, nodes in id_by_class.items():
        for n in nodes:
            class_by_id[n] = int(c)

    lb = preprocessing.LabelBinarizer()
    labels = lb.fit_transform(labels)    # 每一行是one-hot，1的地方是标签
    # print('labels', labels[:10])

    degree = np.sum(adj, axis=1)
    degree = torch.FloatTensor(degree)

    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    features = torch.FloatTensor(features)
    labels = torch.LongTensor(np.where(labels)[1]) # np.where返回不为0的index，[0]就是横坐标，1-24918，[1]中坐标即标签
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    class_list_valid = random.sample(train_class, valid_num_dic[dataset_source])   # 从训练类中抽出一些作为验证类

    class_list_train = list(set(train_class).difference(set(class_list_valid)))    # 其他就是训练类

    print('\n[{} dataset]  nodes:{}  edges:{}  all class:{}  train class:{}  valid class:{}  test class:{}'\
            .format(dataset_source, num_nodes, len(n1s), len(class_list), len(class_list_train), len(class_list_valid), len(class_list_test)))

    return adj, features, labels, degree, class_list_train, class_list_valid, class_list_test, id_by_class, class_by_id, edge_index


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):          # (2708, 1433)
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))      # (2708, 1)
    r_inv = np.power(rowsum, -1).flatten()  # (2708,)
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)             # 对角矩阵 稀疏
    features = r_mat_inv.dot(features)      # 每个元素*每行，对应相乘，features上为1的元素 都除以 这一行为1的个数 （正则化）
    # print(sparse_to_tuple(features))      # (coords, values, shape)，其中coords是2*n的narray，即features的稀疏矩阵；shape是(2708, 1433)
    return torch.Tensor(features.todense()), sparse_to_tuple(features)

def sparse_to_tuple(sparse_mx, insert_batch=False):
    """Convert sparse matrix to tuple representation."""
    """Set insert_batch=True if you want to insert a batch dimension."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        if insert_batch:
            coords = np.vstack((np.zeros(mx.row.shape[0]), mx.row, mx.col)).transpose()
            values = mx.data
            shape = (1,) + mx.shape
        else:
            coords = np.vstack((mx.row, mx.col)).transpose()
            values = mx.data
            shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def get_train_nodes(train_pool, nodes_num):
    train_class_by_id = torch.Tensor([-1 for i in range(nodes_num)]).cuda()
    for task in train_pool:          # np.array(id_support), np.array(id_query), class_selected；只有在task中的节点的标签才已知
        id_support, id_query, class_selected = task[0], task[1], task[2]
        k_way = len(class_selected)
        
        support_set_num = len(id_support)/k_way
        j = 0
        for i in range(len(id_support)):
            train_class_by_id[id_support[i]] = class_selected[j]
            if i % support_set_num == support_set_num-1:
                j += 1
        
        query_set_num = len(id_query)/k_way
        j = 0
        for i in range(len(id_query)):                           # 但是task中的支持节点和查询节点的标签都已知
            train_class_by_id[id_query[i]] = class_selected[j]
            if i % query_set_num == query_set_num-1:
                j += 1      
        
    return train_class_by_id

'''
----------------------------- graph augmentation -----------------------------
'''

def drop_feature(x, drop_prob):
    # print(x[:10, :100])
    # 每个数∈[0,1]，小于0.1置True，所以很多大部分是false，小部分是true
    drop_mask = torch.empty((x.size(1), ), dtype=torch.float32, device=x.device).uniform_(0, 1) < drop_prob
    # print(drop_mask[:100])
    x = x.clone()
    # true的地方置为0，false的地方不变，所以少部分置为0
    x[:, drop_mask] = 0
    # print(x[0, :100])


    return x 


def drop_edge(edge_index, drop_prob):
    return dropout_adj(edge_index, p=drop_prob)[0]


def drop_adj(adj, drop_prob):
    # print()
    # print(adj.is_sparse)
    # print(adj)
    # print(adj.coalesce().indices())

    indices = adj.coalesce().indices()       # .coalesce()：如果indices中存在两条相同的边，则合并为一条，对应的values相加
    values = adj.coalesce().values()         # 在这里的数据不会出现这种情况
    # print(indices[:20])
    # print(values[:20])
    
    # 每个数∈[0,1]，大于0.1置True，所以很多大部分是true，小部分是false
    drop_mask = torch.empty((indices.size(1), ), dtype=torch.float32, device=adj.device).uniform_(0, 1) > drop_prob
    # true的地方可以取，false的地方取不了，所以得到的新值的长度下降了
    indices_n = indices[:, drop_mask]
    values_n = values[drop_mask]
    
    return torch.sparse.FloatTensor(indices_n, values_n, adj.coalesce().size())


'''
----------------------------- few-shot -----------------------------
'''


def pr_generator(edge_index, id_by_class, class_num, degrees):
    # pr
    # G = nx.Graph()
    # for i in range(len(edge_index[0])):
    #     G.add_edge(int(edge_index[0][i]), int(edge_index[1][i]))
    # pr = nx.pagerank(G)

    # class_pr = defaultdict(list)
    # for c in range(class_num):
    #     for n in id_by_class[c]:
    #         class_pr[c].append([int(n), pr[n]])

    class_d = defaultdict(list)
    for c in range(class_num):
        de = []
        for n in id_by_class[c]:
            de.append(degrees[n][0].item())
        de_count = Counter(de)
        de_max = max(de_count, key=de_count.get)

        for n in id_by_class[c]:
            if degrees[n][0].item() == de_max:     # 度为1，这样的节点很多
                class_d[c].append(int(n))

    return class_d


def pr_generator_2(edge_index, id_by_class, class_num, degrees):
    class_d = defaultdict(list)
    for c in range(class_num):
        de = []
        for n in id_by_class[c]:
            de.append(degrees[n][0].item())
        de_count = Counter(de)                                # 某个度有多少个节点
        de_max = max(de_count, key=de_count.get)

        # print(de_count.values())
        # print(sum(de_count.values()))
        # thres = np.percentile(list(de_count.values()), 20)     # 确保有10%的节点
        # print(thres)
        
        thres = 0
        while len(class_d[c]) < len(id_by_class[c])/10:
            thres += 1
            for n in id_by_class[c]:
                if de_count[degrees[n][0].item()] <= thres:         # 和它具有相同度的节点 的数量不多
                    class_d[c].append(int(n))

    # for (k, v) in class_d.items():
    #     print(k, len(v), len(v)/len(id_by_class[k]))

    return class_d


def task_generator_bad(id_by_class, class_list, n_way, k_shot, m_query, class_pr):
    # sample class indices
    class_selected = random.sample(class_list, n_way)                # 抽出n_way个类 list  ##### 这里是随机的所以交叉验证了？
    id_support = []
    id_query = []
    
    for cla in class_selected:                           # 每个类
        
        temp_sup = random.sample(class_pr[cla], k_shot)
        id_support.extend(temp_sup)
        # temp_sup = list(temp_c[-k_shot:, 0].astype(np.int_))
        # id_support.extend(temp_sup)

        temp_qry = list(set(id_by_class[cla]) - set(temp_sup))
        temp_qry = random.sample(list(temp_qry), m_query)
        id_query.extend(temp_qry)

    # 因为是for extend这样增加，所以id_support共有类别数*k_shot个节点，其中前k_shot个是同一个类
    # 这样分存在的问题：id_query中属于每个类的数量是等分的
    np.random.shuffle(id_query)
    return np.array(id_support), np.array(id_query), class_selected  # 返回tuple


def task_generator(id_by_class, class_list, n_way, k_shot, m_query):

    # sample class indices
    class_selected = random.sample(class_list, n_way)                # 抽出n_way个类 list  ##### 这里是随机的所以交叉验证了？
    id_support = []
    id_query = []
    for cla in class_selected:                                       # 每个类
        temp = random.sample(id_by_class[cla], k_shot + m_query)     # 从节点中抽出k_shot + m_query个节点
        id_support.extend(temp[:k_shot])                             # 前面的k_shot个节点是support
        id_query.extend(temp[k_shot:])                               # 后面的是query
    # 因为是for extend这样增加，所以id_support共有类别数*k_shot个节点，其中前k_shot个是同一个类
    # 这样分存在的问题：id_query中属于每个类的数量是等分的
    np.random.shuffle(id_query)
    return np.array(id_support), np.array(id_query), class_selected  # 返回tuple


def euclidean_dist(x, y):
    # x: N x D query
    # y: M x D prototype
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)  # [100, 16] [100, 1, 16] [100, 5, 16]
    y = y.unsqueeze(0).expand(n, m, d)  # [5, 16] [1, 5, 16] [100, 5, 16]

    return torch.pow(x - y, 2).sum(2)   # N x M [100 5] 这100个查询节点到5个原型的欧拉距离


def accuracy(output, labels):
    assert torch.isinf(output).any() == False
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def f1(output, labels):
    preds = output.max(1)[1].type_as(labels)
    f1 = f1_score(labels, preds, average='weighted')
    return f1


'''
----------------------------- args -----------------------------
'''

def repeat(args0):
    # dropout = [0.1, 0.2, 0.3, 0.4]                     
    # tau = [0.2, 0.3, 0.4]                       
    # lr_c = [0.00005]    #[0.0001, 0.0003]   # 0.00005比0.0001差
    # epochs_c = [50, 60, 70, 80]                     # 30比50差
    # batch_size = [256, 512, 1024]                  # 会极大影响时间，倒序；512普遍一般，比256小则慢
    # w_pt_init = [0.001, 0.0001]
    w_loss = [10000.]      # 1000. 10000.

    drop_edge_rate_1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    drop_edge_rate_2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    drop_feature_rate_1 = [0.1, 0.2, 0.3, 0.4, 0.5]
    drop_feature_rate_2 = [0.1, 0.2, 0.3, 0.4, 0.5]
    drop_edge_rate = [[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.3, 0.3]]
    drop_feature_rate = [[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.3, 0.3]]

    # fix
    dropout = [0.4]        # 0.1 0.4
    tau = [0.4]            # 0.6 0.4
    lr_c = [0.0001]
    epochs = [300]         # 100 300
    batch_size = [512]     
    w_pt_init = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.2, 0.5, 1.0]           # 0.1 不要改这个，改了出大事
    # drop_edge_rate = [[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.3, 0.3]]
    # drop_feature_rate = [[0.1, 0.2], [0.1, 0.3], [0.2, 0.2], [0.2, 0.3], [0.2, 0.4], [0.3, 0.3]]
    drop_edge_rate = [[0.2, 0.4]]
    drop_feature_rate = [[0.1, 0.3]]

    all_args = []
    for der in drop_edge_rate:
        for dfr in drop_feature_rate:
            for lr in lr_c:
                for t in tau:
                    for w in w_pt_init:
                        for wl in w_loss:
                            for e in epochs:
                                for d in dropout:
                                    for b in batch_size:
                                        args0.lr_c, args0.tau, args0.w_pt_init, args0.epochs, args0.dropout, args0.batch_size = \
                                                                                                                        lr, t, w, e, d, b
                                        args0.w_loss = wl
                                        args0.drop_edge_rate_1, args0.drop_edge_rate_2 = der[0], der[1]
                                        args0.drop_feature_rate_1, args0.drop_feature_rate_2 = dfr[0], dfr[1]
                                        yield args0
                            
        #                     args_temp = copy.deepcopy(args0)
        #                     args_temp.lr_c, args_temp.tau, args_temp.epochs_c, args_temp.dropout, args_temp.batch_size = lr, t, e, d, b
        #                     all_args.append(args_temp)
    
    # print(len(all_args))
    # return all_args


def average(dataset, log_name, n=5):
    results = 0.
    f1_results = 0.
    with open('./log/{}/{}.txt'.format(dataset, log_name), 'r') as file:
        count = 0
        for line in file.readlines()[::-1]:

            if len(line) < 7 or line[:7] != 'RESULTS':
                continue
            elif line[:7] == 'RESULTS':
                results += float(line[34:42])
                f1_results += float(line[48:56])
                count += 1

                if count == n:
                    break

    print('Average ACC: {}  F1: {}'.format(str(results/n), str(f1_results/n)))
    with open('./log/{}/{}.txt'.format(dataset, log_name), 'a') as file:
        file.write('Average ACC: {}  F1: {}'.format(str(results/n), str(f1_results/n)))
        file.write('\n\n\n')

'''
class Args(object):
    dataset = 'Amazon_clothing'
    hid_dim = 16
    dropout =0.3
    proj_hid_dim = 16
    tau = 0.7
    lr_c = 0.0001
    weight_decay = 0.0005
    epochs_c = 50
    batch_size = 256
    drop_edge_rate_1 = 0.2
    drop_edge_rate_2 = 0.4
    drop_feature_rate_1 = 0.3
    drop_feature_rate_2 = 0.3

    way = 5
    shot = 3
    query = 20
    epochs = 1000
    seed = 1234
'''