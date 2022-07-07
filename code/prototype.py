import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
import time
import os
import networkx as nx
import utils

class Prototype(nn.Module):
    def __init__(self, edge_index, w_pt_init, w_thres_init, n_way, k_shot, hid_dim):
        super(Prototype, self).__init__()
        self.edge_index = edge_index
        self.n_way = n_way
        self.k_shot = k_shot
        self.hid_dim = hid_dim
        
        # refined pt
        self.w_pt = torch.Tensor([w_pt_init]).cuda() #          
        
        # threshold
        self.w_thres_all = nn.Parameter(torch.Tensor([w_thres_init]).cuda(), requires_grad=True)
        self.w_thres_mlp = nn.Sequential(nn.Linear(k_shot*hid_dim, hid_dim),
                                        nn.LeakyReLU(0.2),
                                        nn.Linear(hid_dim, 1),
                                        nn.Tanh()
                                        )
        # self.mlp_info = nn.Sequential(nn.Linear(6, 3),
        #                               nn.LeakyReLU(0.2),
        #                               nn.Linear(3, 1),
        #                               nn.Tanh()
        #                              )
        # self.fc = nn.Linear(k_shot*hid_dim, 1)
        # torch.nn.init.xavier_uniform_(self.fc.weight)
        
        # sup scores
        self.pr = None
        self.hop2_num = None

        # self.W = self.get_param((hid_dim, hid_dim))
        # self.a = self.get_param((hid_dim*2, 1))
        # self.act = nn.LeakyReLU(0.2)
    
    def sup_assign_impt(self, support_embeddings, degrees, id_support):
        # degree adjustment, weight computation
        support_degrees = torch.log(degrees[id_support].view([self.n_way, self.k_shot])).cuda()     # [5, 3]
        support_scores = torch.sigmoid(support_degrees).unsqueeze(-1)                     # [5, 3, 1]
        support_scores = support_scores / torch.sum(support_scores, dim=1, keepdim=True)  # [5, 3, 1] / [5, 1, 1] = [5, 3, 1]
        hop1_scores = support_scores
        # support_embeddings = support_embeddings * support_scores
        
        # PageRank
        if self.pr == None:
            G = nx.Graph()
            for i in range(len(self.edge_index[0])):
                G.add_edge(int(self.edge_index[0][i]), int(self.edge_index[1][i]))
            self.pr = nx.pagerank(G)

        pr_sup = torch.zeros(self.n_way, self.k_shot).cuda()
        for i in range(self.n_way):
            for j in range(self.k_shot):
                pr_sup[i][j] = self.pr[id_support[i*self.k_shot+j]]
        
        pr_scores = pr_sup / pr_sup.sum(1).unsqueeze(1)
        pr_scores = pr_scores.unsqueeze(2)

        # 2-hop
        if self.hop2_num == None:
            G = nx.Graph()
            for i in range(len(self.edge_index[0])):
                G.add_edge(int(self.edge_index[0][i]), int(self.edge_index[1][i]))

            self.hop2_num = defaultdict(int)
            for c_n in G.nodes():
                hop2_nei = []
                hop1_nei = list(G.neighbors(c_n))
                for h1_n in hop1_nei:
                    hop2_nei.extend(list(G.neighbors(h1_n)))
                self.hop2_num[c_n] = len(set(hop2_nei))
        
        hop2_sup = torch.zeros(self.n_way, self.k_shot).cuda()
        for i in range(self.n_way):
            for j in range(self.k_shot):
                hop2_sup[i][j] = self.hop2_num[id_support[i*self.k_shot+j]]
        
        hop2_scores = hop2_sup / hop2_sup.sum(1).unsqueeze(1)
        hop2_scores = hop2_scores.unsqueeze(2)
        
        return support_embeddings * (hop2_scores+pr_scores)/2

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape));
        nn.init.xavier_normal_(param.data)
        return param
    
    def get_w_pt(self, sup_embs, dist_sup2pt):
        k_shot = self.k_shot
        dist_sup2pt_min = dist_sup2pt.min(1).values
        dist_sup2pt_max = dist_sup2pt.max(1).values
        dist_sup2pt_mean = dist_sup2pt.mean(1).values
        dist_sup2pt_var = torch.var(dist_sup2pt, dim=1)
        dist_avg_sup = torch.norm(sup_embs.unsqueeze(2)-sup_embs.unsqueeze(1), dim=-1).sum(-1).sum(-1)/(k_shot*(k_shot-1))
        # print(dist_sup2pt_min.shape)
        # print(dist_sup2pt_max.shape)
        # print(dist_sup2pt_mean.shape)
        # print(dist_sup2pt_var.shape)
        # print(dist_avg_sup.shape)
        
        input1 = torch.cat((dist_sup2pt_min.unsqueeze(0), dist_sup2pt_max.unsqueeze(0), dist_sup2pt_mean.unsqueeze(0), \
                            dist_sup2pt_var.unsqueeze(0), dist_avg_sup.unsqueeze(0)), dim=0)
        out = self.fc(input1)
        # out = self.act(out)
        # print(out.shape)
        # print(out)

        return out

    
    def get_attn_pt(self, sup_embs):                   # [5, 3, dim]  [5, dim]
        init_pt = sup_embs.mean(1)
        n, k, d = self.n_way, self.k_shot, self.hid_dim
        supW = torch.matmul(sup_embs.unsqueeze(-2), self.W)    # [5, 3, 1, dim] [5, 3, 1, dim]
        ptW = torch.matmul(init_pt.unsqueeze(-2), self.W)      # [5, 1, dim] [5, 1, dim]
        ptW = ptW.unsqueeze(1).expand(n, k, 1, d)              # [5, 1, 1, dim] [5, 3, 1,dim]
        concat = torch.cat((supW, ptW), dim=-1)                # [5, 3, 1, dim*2]
        attnscore = torch.matmul(concat, self.a)               # [5, 3, 1, 1]
        exp_pow  = torch.exp(self.act(attnscore))              # [5, 3, 1, 1]
        attnsum = exp_pow.sum(-1).sum(-1).sum(-1).repeat(k, 1).T   # [5, 3]
        attn = exp_pow.squeeze(-1).squeeze(-1) / attnsum       # [5, 3] [5, 3] [5, 3]
        res = attn.unsqueeze(-1) * sup_embs                    # [5, 3, dim]
        
        return res.sum(1), res.mean(1)                                      # [5, dim]
    

    def get_slabel1(self, embeddings, support_embeddings, init_pt, id_unlabelled):
        k = self.k_shot
        dist_un2pt = torch.norm(embeddings.unsqueeze(1)-init_pt, dim=2)    # [24919, 5, 16] [24919, 5] 每个un到n个pt的距离
        # print(dist_un2pt.shape)

        dist_sup2pt = torch.norm(support_embeddings-init_pt.unsqueeze(1), dim=2)  # [5, 3, 16] [5, 3] n个类中，k个sup到pt的距离
        # print(dist_sup2pt.shape)

        # threshold = dist_sup2pt.mean(1)       # [5] support到原型的平均值
        # threshold = dist_sup2pt.max(1).values   # [5] support到原型的最大值
        # threshold = dist_sup2pt.min(1).values   # [5] support到原型的最小值
        threshold = torch.norm(support_embeddings.unsqueeze(2)-support_embeddings.unsqueeze(1), dim=-1).sum(-1).sum(-1)/(k*(k-1))   # support之间的平均距离
        # print(threshold.shape)

        slabel1 = torch.where(threshold - dist_un2pt>0, threshold-dist_un2pt, torch.Tensor([0.0])[0].cuda())  # [24919, 5]
        # slabel1 = torch.where(temp>0, dist_un2pt, torch.Tensor([0.0])[0].cuda())  # [24919, 5] 后面这个更不合理些，应d越大slabel越小
        slabel1_nor = F.softmax(slabel1, dim=1)
        slabel1_nor = torch.where(slabel1_nor==0.2, torch.Tensor([0.]).cuda()[0], slabel1_nor)

        # slabel1_nor = slabel1 / (slabel1.sum(1).unsqueeze(1)+1e-9)   # [24919, 5]
        slabel1_nor = slabel1_nor[id_unlabelled]

        # print('slabel1_nor')
        # self.slabel_static(slabel1_nor)
        # self.check_node_class(id_by_class, id_unlabelled, class_selected, slabel1_nor)   
        return slabel1_nor   


    def get_w_thres(self, support_embeddings, sim_sup2sup, sim_sup2pt):
        # sup_concat = support_embeddings.reshape(support_embeddings.size(0), -1)
        # re_thres = self.w_thres_mlp(sup_concat).squeeze()  # -1~1 期望0
        
        # k = self.k_shot
        # info1 = sim_sup2sup.min(2).values.min(1).values                                                       # sup之间相似度最小值 [5, ]
        # info2 = torch.where(sim_sup2sup==1, torch.Tensor([-1]).cuda()[0], sim_sup2sup).max(2).values.max(1).values   # sup之间相似度最大值 [5, ]
        # info3 = (torch.sum(sim_sup2sup, dim=(1, 2))-k)/(k*(k-1))                               # sup之间相似度平均值 [5, ]

        # info4 = sim_sup2pt.min(1).values.squeeze()                  # sup和pt的相似度的最小值 [5, ]
        # info5 = sim_sup2pt.max(1).values.squeeze()
        # info6 = sim_sup2pt.mean(1).squeeze()
        
        # info = torch.Tensor([]).cuda()
        # info = torch.cat((info, info1.unsqueeze(1)), dim=1)
        # info = torch.cat((info, info2.unsqueeze(1)), dim=1)
        # info = torch.cat((info, info3.unsqueeze(1)), dim=1)
        # info = torch.cat((info, info4.unsqueeze(1)), dim=1)
        # info = torch.cat((info, info5.unsqueeze(1)), dim=1)
        # info = torch.cat((info, info6.unsqueeze(1)), dim=1)
        # re_thres = self.mlp_info(info).squeeze()

        # w_thres = self.w_thres_all + re_thres 
        
        return self.w_thres_all



    def get_slabel2(self, un_embeddings, support_embeddings, init_pt):
        sim_un2sup = F.cosine_similarity(un_embeddings.unsqueeze(1).unsqueeze(1), support_embeddings, dim=-1)  # [24904, 1, 1, 16] [24904, 5, 3]
        sim_un2c = sim_un2sup.max(2).values               # [24904, 5]

        sim_sup2sup = torch.Tensor([]).cuda()
        for i in range(5):
            sim_sup2sup = torch.cat((sim_sup2sup, F.cosine_similarity(support_embeddings[i].unsqueeze(1), support_embeddings[i], dim=-1).unsqueeze(0)))
                          # [5, 3, 3]
        sim_sup2pt = torch.Tensor([]).cuda()
        for i in range(5):
            sim_sup2pt = torch.cat((sim_sup2pt, F.cosine_similarity(support_embeddings[i].unsqueeze(1), init_pt[i], dim=-1).unsqueeze(0)))
                         # [5, 3, 1]

        # threshold
        # threshold = sim_sup2sup.min(2).values.min(1).values                                                       # sup之间相似度最小值 [5, ]
        # threshold = torch.where(sim_sup2sup==1, torch.Tensor([-1]).cuda()[0], sim_sup2sup).max(2).values.max(1).values   # sup之间相似度最大值 [5, ]
        # threshold = (torch.sum(sim_sup2sup, dim=(1, 2))-k_shot)/(k_shot*(k_shot-1))                               # sup之间相似度平均值 [5, ]

        # threshold = sim_sup2pt.max(1).values.squeeze()                  # sup和pt的相似度的最小值 [5, ]
        threshold = sim_sup2pt.max(1).values.squeeze()
        # threshold = sim_sup2pt.mean(1).squeeze()

        # print('threshold')
        # print(sim_sup2sup.min(2).values.min(1).values)
        # print(torch.where(sim_sup2sup==1, torch.Tensor([-1]).cuda()[0], sim_sup2sup).max(2).values.max(1).values)
        # print((torch.sum(sim_sup2sup, dim=(1, 2))-k_shot)/(k_shot*(k_shot-1)))
        # print(sim_sup2pt.min(1).values.squeeze())
        # print(sim_sup2pt.max(1).values.squeeze())
        # print(sim_sup2pt.mean(1).squeeze())

        inf = 9999999999
        slabel2 = torch.where(sim_un2c<threshold, torch.Tensor([-inf]).cuda()[0], sim_un2c)   # [24904, 5]
        slabel2_nor = F.softmax(slabel2, dim=1)
        slabel2_nor = torch.where(slabel2_nor==0.2, torch.Tensor([0.]).cuda()[0], slabel2_nor)

        # print('slabel2_nor')
        # self.slabel_static(slabel2_nor)
        # self.check_node_class(id_by_class, id_unlabelled, class_selected, slabel2_nor)  
        return slabel2_nor


    def get_slabel3(self, un_embeddings, support_embeddings, init_pt):
        sim_un2pt = F.cosine_similarity(un_embeddings.unsqueeze(1), init_pt, dim=-1)  # [24904, 5]      

        sim_sup2sup = torch.Tensor([]).cuda()
        for i in range(self.n_way):
            sim_sup2sup = torch.cat((sim_sup2sup, F.cosine_similarity(support_embeddings[i].unsqueeze(1), support_embeddings[i], dim=-1).unsqueeze(0)))
                          # [5, 3, 3]
        sim_sup2pt = torch.Tensor([]).cuda()
        for i in range(self.n_way):
            sim_sup2pt = torch.cat((sim_sup2pt, F.cosine_similarity(support_embeddings[i].unsqueeze(1), init_pt[i], dim=-1).unsqueeze(0)))
                         # [5, 3, 1]
        # print(sim_sup2sup)
        # print(sim_sup2pt)

        # threshold
        # threshold = sim_sup2sup.min(2).values.min(1).values                                                       # sup之间相似度最小值 [5, ]
        # threshold = torch.where(sim_sup2sup==1, torch.Tensor([-1]).cuda()[0], sim_sup2sup).max(2).values.max(1).values   # sup之间相似度最大值 [5, ]
        # threshold = (torch.sum(sim_sup2sup, dim=(1, 2))-k_shot)/(k_shot*(k_shot-1))                               # sup之间相似度平均值 [5, ]

        # threshold = sim_sup2pt.min(1).values.squeeze()                  # sup和pt的相似度的最小值 [5, ]
        # threshold = sim_sup2pt.max(1).values.squeeze()
        threshold = sim_sup2pt.mean(1).squeeze()

        # print('threshold')
        # print(sim_sup2sup.min(2).values.min(1).values)
        # print(torch.where(sim_sup2sup==1, torch.Tensor([-1]).cuda()[0], sim_sup2sup).max(2).values.max(1).values)
        # print((torch.sum(sim_sup2sup, dim=(1, 2))-k_shot)/(k_shot*(k_shot-1)))
        # print(sim_sup2pt.min(1).values.squeeze())
        # print(sim_sup2pt.max(1).values.squeeze())
        # print(sim_sup2pt.mean(1).squeeze())

        
        # w_thres = self.get_w_thres(support_embeddings, sim_sup2sup, sim_sup2pt)
        w_thres = self.w_thres_all
        self.w_thres = w_thres

        inf = 9999999999
        slabel3 = torch.where(sim_un2pt<w_thres*threshold, torch.Tensor([-inf]).cuda()[0], sim_un2pt-w_thres*threshold)   # [24904, 5]
        slabel3_nor = F.softmax(slabel3, dim=1)
        slabel3_nor = torch.where(slabel3_nor==1/self.n_way, torch.Tensor([0.]).cuda()[0], slabel3_nor)

        # slabel3 = torch.where(sim_un2pt<threshold, torch.Tensor([0.]).cuda()[0], sim_un2pt)
        # slabel3_nor = slabel3 / (slabel3.sum(1).unsqueeze(1) + 1e-10)

        return slabel3_nor


    def forward(self, embeddings, support_embeddings, id_support, class_selected, id_by_class, degrees, episode=None):
                  # [24919, 16]      [5, 3, 16]         [15]               [5]                
        support_embeddings_ipt = self.sup_assign_impt(support_embeddings, degrees, id_support)

        n_way = self.n_way
        k_shot = self.k_shot

        id_unlabelled = np.delete(np.arange(embeddings.size(0)), id_support)
        id_unlabelled = torch.LongTensor(id_unlabelled)
        un_embeddings = embeddings[id_unlabelled]      # 无标签节点的嵌入 [24904, 16]

        # init_sup_sum = support_embeddings.sum(1)         # 初始原型嵌入 [5, 16]
        # init_sup_sum, init_pt = self.get_attn_pt(support_embeddings) 
        ipt_pt = support_embeddings_ipt.sum(1)
        init_pt = support_embeddings.mean(1)

        '''
        ------------------------------ soft label ------------------------------
        '''

        # slabel1_nor = self.get_slabel1(embeddings, support_embeddings, init_pt, id_unlabelled)
        # slabel2_nor = self.get_slabel2(un_embeddings, support_embeddings, init_pt)
        slabel3_nor = self.get_slabel3(un_embeddings, support_embeddings, init_pt)
        # print('slabel2_nor')
        # self.slabel_static(slabel2_nor)
        # self.check_node_class(id_by_class, id_unlabelled, class_selected, slabel2_nor)  
        # exit()
        # if episode == 20:
        #     print('slabel3_nor')
        #     self.slabel_static(slabel3_nor)
        #     self.check_node_class(id_by_class, id_unlabelled, class_selected, slabel3_nor)  
        # exit()
        '''
        ------------------------------ refine ------------------------------
        '''

        # slabel = (slabel1_nor + slabel2_nor) / 2
        slabel = slabel3_nor

        # print('slabel')
        # self.slabel_static(slabel)
        # self.check_node_class(id_by_class, id_unlabelled, class_selected, slabel)
        # print('class_selected', class_selected)
        
        unemb_pt = (un_embeddings.unsqueeze(1) * slabel.unsqueeze(2)).sum(0)  # unlabelled label 对n个原型的贡献微调
        # print(un_embeddings.unsqueeze(1).shape)
        # print(slabel1_nor.unsqueeze(2).shape)
        # print((un_embeddings.unsqueeze(1) * slabel1_nor[id_unlabelled].unsqueeze(2)).shape)
        # print(unemb_pt.shape)
        
        # 看一下support的贡献和unlabelled的贡献
        # print(init_pt)      # [0, 1]小数
        # print(unemb_pt)     # 几百几千

        w_pt = self.w_pt[0]
        # w_pt = self.get_w_pt(support_embeddings, dist_sup2pt)
        # self.w_pt = w_pt     # for output
        # print('w_pt.shape', w_pt.shape)

        # print('init_pt', init_pt)                   # 想比较 微调无标签嵌入 对 原来的pt 的贡献，要pretrain之后才看
        # print('w_pt', w_pt)
        # print('unemb_pt', unemb_pt)
        # print('w_pt * unemb_pt', w_pt * unemb_pt) 
        # print(unemb_pt.shape)
        # print(slabel.sum(0).shape)
        # print('除以1', unemb_pt.unsqueeze(1) / slabel.sum(0).unsqueeze(1)) 
        # print('w除以1', w_pt * unemb_pt.unsqueeze(1) / w_pt * slabel.sum(0).unsqueeze(1)) 
        
        refined_pt = (ipt_pt + w_pt * unemb_pt) / (1 + w_pt * slabel.sum(0)).unsqueeze(1)
        # refined_pt = ipt_pt
        # print(refined_pt.shape)
        # print(refined_pt)

        assert torch.isnan(refined_pt).any() == False
        assert torch.isinf(refined_pt).any() == False

        # if episode == 20:
        #     self.compare(id_by_class, class_selected, embeddings, init_pt, refined_pt)
        # print('************************************')
        # exit()
        return refined_pt
    

    def slabel_static(self, slabel):
        # 有多少节点具有非0的slabel
        count = 0
        for i in slabel:
            if torch.equal(i, torch.Tensor([0. for _ in range(5)]).cuda()) == False:
                count += 1
        print('有多少节点具有非0的slabel', count) 

        # 每个类被分到的节点的数量
        d = defaultdict(int)
        for i in range(slabel.size(0)):
            for j in range(slabel.size(1)):
                if slabel[i][j] != 0.:
                    d[j] += 1
        print('每个类被分到的节点的数量', d)
        
        # 每个类被分到的节点的加权数量
        d1 = defaultdict(float)
        for i in range(slabel.size(0)):
            for j in range(slabel.size(1)):
                if slabel[i][j] != 0.:
                    d1[j] += float(slabel[i][j])
        print('每个类被分到的节点的加权数量', d1)


    def check_node_class(self, id_by_class, id_unlabelled, class_selected, slabel):
        print('实际节点数')
        for i in range(len(class_selected)):
            print(i, class_selected[i], len(id_by_class[class_selected[i]]))
        print('check_node_class')
        assert len(id_unlabelled) == slabel.size(0)
        class_correct = defaultdict(int)
        class_uncorrect = defaultdict(int)
        class_miss = defaultdict(int)
        class_argmax = defaultdict(int)
        for i in range(slabel.size(0)):
            for j in range(slabel.size(1)):
                if slabel[i, j] != 0.:
                    if id_unlabelled[i] in id_by_class[class_selected[j]]:
                        class_correct[j] += 1
                    else:
                        class_uncorrect[j] += 1
                elif slabel[i, j] == 0.:
                    if id_unlabelled[i] in id_by_class[class_selected[j]]:
                        class_miss[j] += 1
            if id_unlabelled[i] in id_by_class[class_selected[torch.argmax(slabel[i]).item()]]:
                class_argmax[torch.argmax(slabel[i]).item()] += 1
        for i in range(len(class_selected)):
            print('{} {}: corr {:6f} uncorr {:6f} miss {:6f} argmaxcorr {:6f}'.\
                    format(i, class_selected[i], class_correct[i]/len(id_by_class[class_selected[i]]),  \
                        class_uncorrect[i]/(class_correct[i]+class_uncorrect[i]), class_miss[i]/len(id_by_class[class_selected[i]]), \
                        class_argmax[i]/len(id_by_class[class_selected[i]])  )) 


    def compare(self, id_by_class, class_selected, embeddings, init_pt, refined_pt):
        print('compare')
        # 所有属于这个类的节点的中心
        pt_select = torch.Tensor([]).cuda()
        for c in class_selected:
            nodes_index = id_by_class[c]
            pt_select = torch.cat((pt_select, embeddings[nodes_index].mean(0).unsqueeze(0)), dim=0)
        print('pt_select')
        print(pt_select)

        print('init_pt')
        print(init_pt)
        print('dis', torch.norm(pt_select-init_pt, dim=1))
        print('cos', F.cosine_similarity(pt_select, init_pt))
        print('refined_pt')
        print(refined_pt)
        print('dis', torch.norm(pt_select-refined_pt, dim=1))
        print('cos', F.cosine_similarity(pt_select, refined_pt))


'''
def get_prototype(embeddings, support_embeddings, id_support, class_selected, n_way, k_shot):
    # print(embeddings.shape)                     # [24919, 16]
    # print(support_embeddings.shape)             # [5, 3, 16]
    # print(id_support.shape, id_support)         # 15
    # print(len(class_selected), class_selected)  # 5
    # print(n_way, k_shot)                        # 5 3

    init_pt = support_embeddings.sum(1)         # 初始原型嵌入 [5, 16]
    # print(init_pt.shape)
    
    dist_un2pt = torch.norm(embeddings.unsqueeze(1)-init_pt, dim=2)    # [24919, 5, 16] [24919, 5] 每个un到n个pt的距离
    # print(dist_un2pt.shape)

    dist_sup2pt = torch.norm(support_embeddings-init_pt.unsqueeze(1), dim=2)  # [5, 3, 16] [5, 3] n个类中，k个sup到pt的距离
    # print(dist_sup2pt.shape)


    # threshold = dist_sup2pt.mean(1)       # [5] support到原型的平均值
    # threshold = dist_sup2pt.max(1).values   # [5] support到原型的最大值
    threshold = dist_sup2pt.min(1).values   # [5] support到原型的最小值
    # threshold = torch.norm(support_embeddings.unsqueeze(1)-support_embeddings, dim=2).mean()   # support之间的平均距离
    # print(threshold.shape)

    temp = threshold - dist_un2pt
    slabel1 = torch.where(temp>0, temp, torch.Tensor([0.0])[0].cuda())  # [24919, 5]
    # print(slabel1.shape)
    # print(slabel1[:5])

    slabel1_nor = slabel1 / (slabel1.sum(1).unsqueeze(1)+1e-9)   # [24919, 5]
    # print(slabel1_nor.shape)
    # print(slabel1_nor[:5])
    # statistic(slabel1_nor)

    # id_unlabelled = torch.LongTensor([x for x in range(embeddings.size(0)) if x not in id_support]).cuda()
    id_unlabelled = np.delete(np.arange(embeddings.size(0)), id_support)
    id_unlabelled = torch.LongTensor(id_unlabelled)
    un_embeddings = embeddings[id_unlabelled]      # 无标签节点的嵌入 [24904, 16]

    unemb_pt = (un_embeddings.unsqueeze(1) * slabel1_nor[id_unlabelled].unsqueeze(2)).sum(0)  # unlabelled label 对n个原型的贡献
    # print(un_embeddings.unsqueeze(1).shape)
    # print(slabel1_nor.unsqueeze(2).shape)
    # print(unemb_pt.shape)
    
    # 看一下support的贡献和unlabelled的贡献
    # print(init_pt)      # [0, 1]小数
    # print(unemb_pt)     # 几百几千

    w_pt = 0.1
    refined_pt = (init_pt + w_pt * unemb_pt) / (k_shot + w_pt * slabel1_nor.sum(0)).unsqueeze(1)
    # print(refined_pt.shape)
    # print(refined_pt)

    return refined_pt



def statistic(slabel1_nor):
    # 看一下有多少节点贡献了这n个类
    index = torch.nonzero(slabel1_nor.T).tolist()
    d = defaultdict(int)
    for i in index:
        d[i[0]] += 1
    print(d)
    
    # 看一下有多少节点属于两个类以上
    index2 = torch.nonzero(slabel1_nor).tolist()
    d2 = defaultdict(int)
    for i in index2:
        d2[i[0]] += 1
    print(len(d2.keys()))   # 属于某一个类的节点数
    count = 0
    for (k, v) in d2.items():
        if v > 1:
            count += 1
    print(count)

'''