from asyncio import protocols
import torch
import numpy as np
import argparse
import random
import time
import sys
import pickle

import utils
# import prototype as pt
from models import *

FILENAME = 'log/dblp/v6.4.txt'
# PS = 'pretrain GNN with given params'
ALL_LOG = 'v6.4-all'

class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a+")
 
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()                     #每次写入后刷新到文件中，防止程序意外结束

    def flush(self):
        self.log.flush()

sys.stdout = Logger(FILENAME)
print(FILENAME)
# print(PS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,  default='dblp', help='Dataset:Amazon_clothing/Amazon_eletronics/dblp')

    # GNN
    parser.add_argument('--hid_dim', type=int, default=16, help='16')
    parser.add_argument('--dropout', type=float, default=0.3)

    # contrastive
    parser.add_argument('--proj_hid_dim', type=int, default=16, help='16')
    parser.add_argument('--tau', type=float, default=0.7)
    parser.add_argument('--lr_c', type=float, default=0.0001, help='0.005')   # 0.0001
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='0.0005')
    parser.add_argument('--epochs_c', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--drop_edge_rate_1', type=float, default=0.2)
    parser.add_argument('--drop_edge_rate_2', type=float, default=0.4)
    parser.add_argument('--drop_feature_rate_1', type=float, default=0.3)
    parser.add_argument('--drop_feature_rate_2', type=float, default=0.3)

    # distance loss
    parser.add_argument('--w_pt_init', type=float, default=0.2)
    parser.add_argument('--w_thres_init', type=float, default=1.)

    # few-shot
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=3)
    parser.add_argument('--query', type=int, default=20)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--w_loss', type=float, default=1000.)

    parser.add_argument('--seed', type=int, default=1234)

    args = parser.parse_args()
    return args


def run(args):
    dataset, hid_dim, dropout = args.dataset, args.hid_dim, args.dropout
    proj_hid_dim, tau, lr_c, weight_decay, epochs_c, batch_size = args.proj_hid_dim, args.tau, args.lr_c, args.weight_decay, args.epochs_c, args.batch_size
    drop_edge_rate_1, drop_edge_rate_2, drop_feature_rate_1, drop_feature_rate_2 = args.drop_edge_rate_1, args.drop_edge_rate_2, args.drop_feature_rate_1, args.drop_feature_rate_2
    w_pt_init, w_thres_init = args.w_pt_init, args.w_thres_init
    way, shot, query, epochs, w_loss, seed = args.way, args.shot, args.query, args.epochs, args.w_loss, args.seed
    cuda = True

    start_time = time.time()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dataset = dataset
    adj, features, labels, degrees, class_list_train, class_list_valid, class_list_test, id_by_class, class_by_id, edge_index = utils.load_data(dataset)
    adj, features, labels, edge_index = adj.cuda(), features.cuda(), labels.cuda(), edge_index.cuda()
    edges_num = edge_index.size(1)
    nodes_num = features.size(0)
    # print()
    # print('adj', type(adj), adj.shape)
    # print('features', type(features), features.shape)
    # print('labels', type(labels), labels.shape)


    feat_dim = features.size(1)
    # encoder = Encoder(feat_dim, args.hid_dim, activation=F.relu).cuda()
    encoder = Encoder(feat_dim, hid_dim, dropout).cuda()
    contrastive = Contrastive(hid_dim, proj_hid_dim, tau, dropout)
    prototype = Prototype(edge_index, w_pt_init, w_thres_init, way, shot, hid_dim).cuda()
    model = Model(encoder, contrastive, prototype).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_c, weight_decay=weight_decay)

    n_way = way
    k_shot = shot
    n_query = query
    meta_test_num = 50        # 任务数量
    meta_valid_num = 50

    # class_pr = utils.pr_generator_2(edge_index, id_by_class, len(id_by_class.keys()), degrees)

    train_pool = [utils.task_generator(id_by_class, class_list_train, n_way, k_shot, n_query) for i in range(epochs)]
    valid_pool = [utils.task_generator(id_by_class, class_list_valid, n_way, k_shot, n_query) for i in range(meta_valid_num)]
    test_pool = [utils.task_generator(id_by_class, class_list_test, n_way, k_shot, n_query) for i in range(meta_test_num)]

    bsize = batch_size
    train_class_by_id = utils.get_train_nodes(train_pool, nodes_num) 

    '''
    -------------------------- train & test --------------------------
    '''

    def get_loss_main(embeddings, id_support, id_query, episode):
        z_dim = embeddings.size()[1]

        # embedding lookup
        support_embeddings = embeddings[id_support]                            # 共有id_support*k_shot个
        support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])   # 按类别分开
        query_embeddings = embeddings[id_query]

        # compute loss 计算
        prototype_embeddings = model.prototype(embeddings, support_embeddings, id_support, class_selected, id_by_class, degrees, episode)       # shape n_way z_dim

        dists = utils.euclidean_dist(query_embeddings, prototype_embeddings)     # [100, 5]
        output = F.log_softmax(-dists, dim=1)                                    # [100, 5]

        # print('labels[id_query]', labels[id_query])    
        labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).cuda()

        return output, labels_new, F.nll_loss(output, labels_new)

    def get_loss_aux(x_1, adj_1, x_2, adj_2, index, i, bsize, train_class_by_id):
        
        z1 = model(x_1, adj_1)
        z2 = model(x_2, adj_2)
        loss = model.contrastive(z1[index[i:i+bsize]], z2[index[i:i+bsize]], train_class_by_id[index[i:i+bsize]]) / bsize

        return loss


    def train(class_selected, id_support, id_query, \
                x_1, adj_1, x_2, adj_2, index, i, bsize, episode):
        optimizer.zero_grad()
        embeddings = model(features, adj)    # 实际上就是encoder
        
        output, labels_new, loss_train_main = get_loss_main(embeddings, id_support, id_query, episode) 
        loss_train_aux = get_loss_aux(x_1, adj_1, x_2, adj_2, index, i, bsize, train_class_by_id)
        # print('loss_train_main:{} loss_train_aux:{}'.format(loss_train_main, loss_train_aux))

        loss_train = loss_train_main + loss_train_aux * w_loss
        loss_train.backward()
        optimizer.step()

        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
        acc_train = utils.accuracy(output, labels_new)
        f1_train = utils.f1(output, labels_new)

        return acc_train, f1_train, loss_train_main, loss_train_aux


    def test(class_selected, id_support, id_query, \
                x_1, adj_1, x_2, adj_2, index, i, bsize, episode):
        model.eval()
        embeddings = model(features, adj)
        z_dim = embeddings.size()[1]

        # embedding lookup
        support_embeddings = embeddings[id_support]        # 3*5=15
        support_embeddings = support_embeddings.view([n_way, k_shot, z_dim])
        query_embeddings = embeddings[id_query]            # 20*5=100

        # compute loss
        prototype_embeddings = model.prototype(embeddings, support_embeddings, id_support, class_selected, id_by_class, degrees)
        
        dists = utils.euclidean_dist(query_embeddings, prototype_embeddings)
        output = F.log_softmax(-dists, dim=1)

        labels_new = torch.LongTensor([class_selected.index(i) for i in labels[id_query]]).cuda()
        loss_test = F.nll_loss(output, labels_new)

        output = output.cpu().detach()
        labels_new = labels_new.cpu().detach()
        acc_test = utils.accuracy(output, labels_new)
        f1_test = utils.f1(output, labels_new)

        return acc_test, f1_test, embeddings, prototype_embeddings

    '''
    -------------------------- few-shot --------------------------
    '''

    def store_emb(embeddings, class_selected, id_by_class, id_support, id_query, all_pt, store):
        if store == None:
            return
        file = open('./log/{}/emb/emb-{}-{}-{}-{}.pickle'.format(dataset, way, shot, store, str(seed)), 'wb')
        data = {
            'embeddings': embeddings,
            'class_selected': class_selected,
            'id_by_class': id_by_class,
            'class_by_id': class_by_id,
            'id_support': id_support,
            'id_query': id_query,
            'test_pool': test_pool,
            'all_pt': all_pt
        }
        pickle.dump(data, file)


    meta_train_acc = []
    best_valid_acc = -1
    best_acc = -1
    best_f1 = -1
    best_acc_epoch = -1
    for episode in range(epochs):
        id_support, id_query, class_selected = train_pool[episode]       # 一个任务 for main
        
        adj_1 = utils.drop_adj(adj, drop_edge_rate_1)                    # aux：一个epoch里面遍历所有的batch
        adj_2 = utils.drop_adj(adj, drop_edge_rate_2)
        x_1 = utils.drop_feature(features, drop_feature_rate_1)
        x_2 = utils.drop_feature(features, drop_feature_rate_2)
        assert x_1.size(0) == nodes_num

        losses = []
        index = [i for i in range(nodes_num)]
        np.random.shuffle(index)
        
        # 一个task所有batch
        loss_epi_main, loss_epi_aux = 0, 0
        for i in range(0, nodes_num, bsize):
            acc_train, f1_train, loss_train_main, loss_train_aux = train(class_selected, id_support, id_query,  \
                                                                        x_1, adj_1, x_2, adj_2, index, i, bsize, episode)
            loss_epi_main += loss_train_main
            loss_epi_aux += loss_train_aux
        
        meta_train_acc.append(acc_train)

        if (episode >= 0 and episode % 1 == 0) or episode==epochs-1:              # 用10个meta-train-tasks进行训练之后，进行测试
            print("-------Episode {}-------".format(episode))
            print("Main Loss:{} Aux Loss:{}".format(loss_epi_main/(nodes_num/bsize), loss_epi_aux/(nodes_num/bsize)))
            print("Meta-Train_Accuracy: {}".format(np.array(meta_train_acc).mean(axis=0)))

            # validation
            meta_test_acc = []
            meta_test_f1 = []
            for idx in range(meta_valid_num):
                id_support, id_query, class_selected = valid_pool[idx]
                acc_test, f1_test, _, _ = test(class_selected, id_support, id_query,  \
                                                x_1, adj_1, x_2, adj_2, index, i, bsize, episode)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
            acc_v = np.array(meta_test_acc).mean(axis=0)
            f1_v = np.array(meta_test_f1).mean(axis=0)
            print("Meta-valid_Accuracy: {}, Meta-valid_F1: {}".format(acc_v, f1_v))
            
            # testing
            meta_test_acc = []
            meta_test_f1 = []
            all_pt = torch.Tensor([]).cuda()
            for idx in range(meta_test_num):                                 # 所有任务的acc的均值
                id_support, id_query, class_selected = test_pool[idx]
                acc_test, f1_test, embeddings, pt = test(class_selected, id_support, id_query,  \
                                                            x_1, adj_1, x_2, adj_2, index, i, bsize, episode)
                meta_test_acc.append(acc_test)
                meta_test_f1.append(f1_test)
                all_pt = torch.cat((all_pt, pt.unsqueeze(0)), 0)
            
            acc_t = np.array(meta_test_acc).mean(axis=0)
            f1_t = np.array(meta_test_f1).mean(axis=0)
            print("Meta-Test_Accuracy: {}, Meta-Test_F1: {}".format(acc_t, f1_t))

            # if episode%20==0 or episode==epochs-1:
            #     store = None # 'test_nolbl_' + str(episode)   # sup+qry scores / noscores
            #     store_emb(embeddings, class_selected, id_by_class, id_support, id_query, all_pt, store)
            # else:
            #     store = None

            if acc_v >= best_valid_acc:
                best_acc_epoch = episode
                best_acc = acc_t
                best_f1 = f1_t
                best_valid_acc = acc_v
            print('w_pt:{:8f} w_thres:{}'.format(model.prototype.w_pt.item(), model.prototype.w_thres.data))

    end_time = time.time()
    print(' Time: {:4f}'.format(end_time - start_time))
    print(' AGAIN ', args)
    print(' RESULTS from epoch {:3d} \n Accuracy: {:8f} \n F1: {:8f} \n'.format(best_acc_epoch, best_acc, best_f1))

    with open('./log/{}/{}.txt'.format(dataset, ALL_LOG), 'a') as f:
        f.write(str(args))
        f.write('\n')
        f.write('RESULTS from epoch {:3d}  Accuracy: {:8f}  F1: {:8f}  w_pt:{}  w_thres:{}' .format(best_acc_epoch, \
                    best_acc, best_f1, model.prototype.w_pt.item(), model.prototype.w_thres.data))
        f.write('\n')


if __name__ == '__main__':
    args0 = parse_args()
    seeds = [806, 865, 510, 5, 274, 326, 491, 725, 931, 74]      # random.sample(range(1, 1000), 5)
    print(seeds)
    for n in [5]:
        for k in [3]:
            for args in iter(utils.repeat(args0)):
                for s in seeds:
                    args.way, args.shot = n, k
                    args.seed = s
                    print(args)
                    run(args)
                utils.average(args0.dataset, ALL_LOG, n=len(seeds))


        
            

    




