# *_*coding:utf-8 *_*
import torch
import torch.nn as nn
from torch.nn import functional as F, Parameter
from torch.nn.init import xavier_normal_, xavier_uniform_
import numpy as np
import sys
from utils import *
import math
import copy
from time_encoder import TimeEncode
from decoder import TimeConvTransE
from gcn_layer import CandRGCNLayer, RGCNBlockLayer, UnionRGCNLayer
from rgcn.model import BaseRGCN
from utils import get_entity_relation_set

torch.set_printoptions(profile='full')
"""
class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, 2 * input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(2 * input_dim, input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)
"""

class RGCNCell(BaseRGCN):
    def build_hidden_layer(self, idx):
        act = F.rrelu
        if idx:
            self.num_basis = 0
        return UnionRGCNLayer(self.h_dim, self.h_dim, 4, self.num_bases,
                            activation=act, dropout=self.dropout, self_loop=True, skip_connect=False)

    def forward(self, g, init_ent_emb, init_rel_emb):
        node_id = g.ndata['id'].squeeze()
        g.ndata['h'] = init_ent_emb[node_id]
        x, r = init_ent_emb, init_rel_emb
        for i, layer in enumerate(self.layers):
            layer(g, [], r[i])
        return g.ndata.pop('h')

class Oracle(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(Oracle, self).__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, input_dim),
                                    nn.BatchNorm1d(input_dim),
                                    nn.Dropout(0.2),
                                    nn.LeakyReLU(0.2),
                                    nn.Linear(input_dim, out_dim),
                                    )

    def forward(self, x):
        return self.linear(x)

class GatingMechanism(nn.Module):
    def __init__(self, num_e, num_rel, h_dim):
        super(GatingMechanism, self).__init__()
        # gating 的参数
        self.gate_theta = nn.Parameter(torch.Tensor(num_e, h_dim), requires_grad=True).float()
        self.num_rels = num_rel
        nn.init.xavier_uniform_(self.gate_theta)
        self.linear = nn.Linear(h_dim, 1)

    def forward(self, X: torch.LongTensor, Y: torch.LongTensor):
        '''
        :param X:   LSTM 的输出tensor   |E| * H
        :param Y:   Entity 的索引 id    |E|,
        :return:    Gating后的结果      |E| * H
        '''
        gate = torch.sigmoid(self.linear(self.gate_theta[X]))
        return gate

class PLEASING(nn.Module):
    def __init__(self, num_e, num_rel, num_t, args, num_static_rels, num_words):
        super(PLEASING, self).__init__()
        # stats
        self.num_e = num_e
        self.num_t = num_t
        self.num_rel = num_rel
        self.args = args
        self.num_static_rels = num_static_rels
        self.num_words = num_words

        self.words_emb = torch.nn.Parameter(torch.Tensor(self.num_words, args.embedding_dim), requires_grad=True).float()
        torch.nn.init.xavier_normal_(self.words_emb)
        self.static_rgcn_layer = RGCNBlockLayer(args.embedding_dim, args.embedding_dim, self.num_static_rels*2, 100,
                                                activation=F.rrelu, dropout=0.2, self_loop=False, skip_connect=False)
        self.static_loss = torch.nn.MSELoss()
        
        self.historical_weight = nn.Parameter(torch.Tensor(args.batch_size, self.num_e), requires_grad=True).float()
        nn.init.xavier_uniform_(self.historical_weight, gain=nn.init.calculate_gain('sigmoid'))
        self.time_gate_weight = nn.Parameter(torch.Tensor(self.num_e, self.num_e))
        nn.init.xavier_uniform_(self.time_gate_weight, gain=nn.init.calculate_gain('relu'))
        self.time_gate_bias = nn.Parameter(torch.Tensor(self.num_e))
        nn.init.zeros_(self.time_gate_bias)
        self.rel_embeds = nn.Parameter(torch.Tensor(2 * num_rel, args.embedding_dim), requires_grad=True).float()
        nn.init.xavier_uniform_(self.rel_embeds, gain=nn.init.calculate_gain('relu'))
        self.entity_embeds = nn.Parameter(torch.Tensor(self.num_e, args.embedding_dim), requires_grad=True).float()
        nn.init.xavier_uniform_(self.entity_embeds, gain=nn.init.calculate_gain('relu'))
        # add initial time embedding
        self.time_embeds = nn.Parameter(torch.Tensor(args.timestamps, 32), requires_grad=True).float()
        nn.init.xavier_uniform_(self.time_embeds, gain=nn.init.calculate_gain('relu'))
        self.time_constraint = nn.Linear(32, args.embedding_dim)

        self.linear_frequency = nn.Linear(self.num_e, args.embedding_dim)

        self.contrastive_hidden_layer = nn.Linear(3 * args.embedding_dim, args.embedding_dim)
        self.contrastive_output_layer = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.oracle_layer = Oracle(3 * args.embedding_dim, 1)
        self.oracle_layer.apply(self.weights_init)

        self.linear_pred_layer_s1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o1 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o2 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.linear_pred_layer_s3 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_pred_layer_o3 = nn.Linear(2 * args.embedding_dim, args.embedding_dim)

        self.time_encoder = TimeEncode(args.embedding_dim)

        self.weight_t1 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.bias_t1 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.weight_t2 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.bias_t2 = nn.parameter.Parameter(torch.randn(1, args.embedding_dim))
        self.decoder1 = TimeConvTransE(self.num_e, args.embedding_dim, 0.2, 0.2, 0.2)
        self.decoder2 = TimeConvTransE(self.num_e, args.embedding_dim, 0.2, 0.2, 0.2)
        self.time_linear1 = nn.Linear(2* args.embedding_dim, args.embedding_dim)
        self.time_linear2 = nn.Linear(2* args.embedding_dim, args.embedding_dim)
        self.cand_layer_s = CandRGCNLayer(args.embedding_dim, args.embedding_dim, self.num_rel, 100,
                            activation=F.rrelu, dropout=0.2, self_loop=True, skip_connect=False)
        self.cand_layer_o = CandRGCNLayer(args.embedding_dim, args.embedding_dim, self.num_rel, 100,
                            activation=F.rrelu, dropout=0.2, self_loop=True, skip_connect=False)
        self.rgcn_s = RGCNCell(self.num_e,
                             args.embedding_dim,
                             args.embedding_dim,
                             self.num_rel * 2,
                             100,
                             2,
                             args.dropout)
        self.rgcn_o = RGCNCell(self.num_e,
                             args.embedding_dim,
                             args.embedding_dim,
                             self.num_rel * 2,
                             100,
                             2,
                             args.dropout)

        self.rgcn_t = RGCNBlockLayer(args.embedding_dim, args.embedding_dim, 4*2, 100,
                                                activation=F.rrelu, dropout=0.2, self_loop=False, skip_connect=False)
        self.gate = GatingMechanism(self.num_e, self.num_rel, args.embedding_dim)
        self.con_gate_s = ConcurrentGating(self.num_e, args.embedding_dim)
        self.con_gate_o = ConcurrentGating(self.num_e, args.embedding_dim)

        #* new contrastive
        self.linear_er = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_ef = nn.Linear(2 * args.embedding_dim, args.embedding_dim)
        self.linear_rf = nn.Linear(args.embedding_dim, args.embedding_dim)
        self.weights_init(self.linear_er)
        self.weights_init(self.linear_rf)
        self.weights_init(self.linear_ef)

        self.weights_init(self.linear_frequency)
        self.weights_init(self.linear_pred_layer_s1)
        self.weights_init(self.linear_pred_layer_o1)
        self.weights_init(self.linear_pred_layer_s2)
        self.weights_init(self.linear_pred_layer_o2)
        self.weights_init(self.linear_pred_layer_s3)
        self.weights_init(self.linear_pred_layer_o3)
        self.weights_init(self.time_linear1)
        self.weights_init(self.time_linear2)

        self.dropout = nn.Dropout(args.dropout)
        self.logSoftmax = nn.LogSoftmax()
        self.softmax = nn.Softmax()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.crossEntropy = nn.BCELoss()
        self.oracle_mode = args.oracle_mode
        self.celoss = nn.CrossEntropyLoss()
        self.gru_s = nn.GRUCell(2 * args.embedding_dim, args.embedding_dim)
        self.gru_o = nn.GRUCell(2 * args.embedding_dim, args.embedding_dim)
        self.leakyrelu = nn.LeakyReLU(0.2)
        print('PLEASING Initiated')

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

    def softmax_focal_loss(self, input, gamma=2., alpha=0.75):
        logp = input
        p = torch.exp(-logp)
        loss = (1 - p) ** gamma * logp
        return loss.mean()

    def get_init_time(self, quadrupleList):
        T_idx = quadrupleList.unsqueeze(1).float()
        t1 = self.weight_t1 * T_idx + self.bias_t1
        t2 = torch.sin(self.weight_t2 * T_idx + self.bias_t2)
        t = self.time_constraint(torch.cat([t1, t2], dim=1))
        return t


    def forward(self, batch_block, mode_lk, static_graph, t_graph, total_data=None):
        quadruples, s_history_event_o, o_history_event_s, \
        s_history_label_true, o_history_label_true, s_frequency, o_frequency = batch_block
        if isListEmpty(s_history_event_o) or isListEmpty(o_history_event_s):
            sub_rank, obj_rank, batch_loss = [None] * 3
            if mode_lk == 'Training':
                return batch_loss
            elif mode_lk in ['Valid', 'Test']:
                return sub_rank, batch_loss
            else:
                return None

        s = quadruples[:, 0]
        r = quadruples[:, 1]
        o = quadruples[:, 2]

        """
        t = (quadruples[:, 3] / 24.0).long()
        time_embedding = self.pe[t]
        """
        t = (quadruples[:, 3] / self.args.time_span).long()
        self.init_time_emb = self.time_encoder(self.time_embeds)
        t_graph = t_graph.to(self.args.gpu)
        t_graph.ndata['h'] = self.init_time_emb
        self.rgcn_t(t_graph, [])
        all_time = F.normalize(t_graph.ndata.pop('h'))
        self.fact_time = all_time[t]

        if self.args.static:
            static_graph = static_graph.to(self.args.gpu)
            static_graph.ndata['h'] = torch.cat((self.entity_embeds, self.words_emb), dim=0)  # 演化得到的表示，和wordemb满足静态图约束
            self.static_rgcn_layer(static_graph, [])
            static_emb = static_graph.ndata.pop('h')[:self.num_e, :]
            static_emb = F.normalize(static_emb)
            self.init_ent_emb = static_emb
        else:
            self.init_ent_emb = self.entity_embeds

        s_history_tag = copy.deepcopy(s_frequency)
        o_history_tag = copy.deepcopy(o_frequency)
        s_non_history_tag = copy.deepcopy(s_frequency)
        o_non_history_tag = copy.deepcopy(o_frequency)

        s_history_tag[s_history_tag != 0] = self.args.lambdax
        o_history_tag[o_history_tag != 0] = self.args.lambdax

        s_non_history_tag[s_history_tag == 1] = -self.args.lambdax
        s_non_history_tag[s_history_tag == 0] = self.args.lambdax

        o_non_history_tag[o_history_tag == 1] = -self.args.lambdax
        o_non_history_tag[o_history_tag == 0] = self.args.lambdax

        s_history_tag[s_history_tag == 0] = -self.args.lambdax
        o_history_tag[o_history_tag == 0] = -self.args.lambdax

        s_frequency = F.softmax(s_frequency, dim=1)
        o_frequency = F.softmax(o_frequency, dim=1)
        s_frequency_hidden = self.tanh(self.linear_frequency(s_frequency))
        o_frequency_hidden = self.tanh(self.linear_frequency(o_frequency))

        if mode_lk == 'Training':
            s_nce_loss, s_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                    self.linear_pred_layer_s1, self.linear_pred_layer_s2, self.linear_pred_layer_s3,
                                                    s_history_tag, s_non_history_tag)
            o_nce_loss, o_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                    self.linear_pred_layer_o1, self.linear_pred_layer_o2, self.linear_pred_layer_o3,
                                                    o_history_tag, o_non_history_tag)
            # calculate_spc_loss(self, hidden_lk, actor1, r, rel_embeds, targets):
            s_spc_loss = self.calculate_spc_loss(s, r, self.rel_embeds[:self.num_rel],
                                                 s_history_label_true, s_frequency_hidden)
            o_spc_loss = self.calculate_spc_loss(o, r, self.rel_embeds[self.num_rel:],
                                                 o_history_label_true, o_frequency_hidden)

            s_cand_loss, _, s_ent_emb = self.calculate_cand_loss(s, o, r, self.rel_embeds[:self.num_rel], s_preds, self.cand_layer_s, self.decoder1, self.gru_s, self.time_linear1)
            o_cand_loss, _, o_ent_emb = self.calculate_cand_loss(o, s, r, self.rel_embeds[self.num_rel:], o_preds, self.cand_layer_o, self.decoder2, self.gru_o, self.time_linear2)

            nce_loss = (s_nce_loss + o_nce_loss) / 2.0
            spc_loss = (s_spc_loss + o_spc_loss) / 2.0
            cand_loss = (s_cand_loss + o_cand_loss) / 2.0
            # print('nce loss', nce_loss.item(), ' spc loss', spc_loss.item())

            if self.args.static:
                s_static_loss = self.calculate_static_loss(static_emb, s_ent_emb)
                o_static_loss = self.calculate_static_loss(static_emb, o_ent_emb)
                static_loss = (s_static_loss + o_static_loss) /2.0
                total_loss = self.args.alpha * nce_loss + (1 - self.args.alpha - self.args.beta) * spc_loss + self.args.beta * cand_loss + static_loss
            else:
                total_loss = self.args.alpha * nce_loss + (1 - self.args.alpha - self.args.beta) * spc_loss + self.args.beta * cand_loss
            return total_loss

        elif mode_lk in ['Valid', 'Test']:
            s_history_oid = []
            o_history_sid = []

            for i in range(quadruples.shape[0]):
                s_history_oid.append([])
                o_history_sid.append([])
                for con_events in s_history_event_o[i]:
                    s_history_oid[-1] += con_events[:, 1].tolist()
                for con_events in o_history_event_s[i]:
                    o_history_sid[-1] += con_events[:, 1].tolist()

            s_nce_loss, ss_preds = self.calculate_nce_loss(s, o, r, self.rel_embeds[:self.num_rel],
                                                          self.linear_pred_layer_s1, self.linear_pred_layer_s2, self.linear_pred_layer_s3,
                                                          s_history_tag, s_non_history_tag)
            o_nce_loss, oo_preds = self.calculate_nce_loss(o, s, r, self.rel_embeds[self.num_rel:],
                                                          self.linear_pred_layer_o1, self.linear_pred_layer_o2, self.linear_pred_layer_o3,
                                                          o_history_tag, o_non_history_tag)

            s_cand_loss, s_preds, s_ent_emb = self.calculate_cand_loss(s, o, r, self.rel_embeds[:self.num_rel], ss_preds, self.cand_layer_s, self.decoder1, self.gru_s, self.time_linear1)
            o_cand_loss, o_preds, o_ent_emb = self.calculate_cand_loss(o, s, r, self.rel_embeds[self.num_rel:], oo_preds, self.cand_layer_o, self.decoder2, self.gru_o, self.time_linear2)

            s_ce_loss, s_pred_history_label, s_ce_all_acc = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                                                             s_history_label_true, s_frequency_hidden)
            o_ce_loss, o_pred_history_label, o_ce_all_acc = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                                                             o_history_label_true, o_frequency_hidden)

            s_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask = to_device(torch.zeros(quadruples.shape[0], self.num_e))

            for i in range(quadruples.shape[0]):
                if s_pred_history_label[i].item() > 0.5:
                    s_mask[i, s_history_oid[i]] = 1
                else:
                    s_mask[i, :] = 1
                    s_mask[i, s_history_oid[i]] = 0

                if o_pred_history_label[i].item() > 0.5:
                    o_mask[i, o_history_sid[i]] = 1
                else:
                    o_mask[i, :] = 1
                    o_mask[i, o_history_sid[i]] = 0

            if self.oracle_mode == 'hard':
                s_mask = s_mask
                o_mask = o_mask

            if self.oracle_mode == 'soft':
                s_mask = self.sigmoid(s_mask)
                o_mask = self.sigmoid(o_mask)
            if mode_lk == 'Valid':
                s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                            s_mask, total_data, 's', True)
                o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                            o_mask, total_data, 'o', True)
            if mode_lk == 'Test':
                s_total_loss1, sub_rank1 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                            s_mask, total_data, 's', True, history_tag=s_history_tag, case_study=True)
                o_total_loss1, obj_rank1 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                            o_mask, total_data, 'o', True, history_tag=o_history_tag, case_study=True)
            batch_loss1 = (s_total_loss1 + o_total_loss1) / 2.0

            s_total_loss2, sub_rank2 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask, total_data, 's', False)
            o_total_loss2, obj_rank2 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask, total_data, 'o', False)
            batch_loss2 = (s_total_loss2 + o_total_loss2) / 2.0

            # Ground Truth
            s_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))
            o_mask_gt = to_device(torch.zeros(quadruples.shape[0], self.num_e))


            for i in range(quadruples.shape[0]):
                if o[i] in s_history_oid[i]:
                    s_mask_gt[i, s_history_oid[i]] = 1
                else:
                    s_mask_gt[i, :] = 1
                    s_mask_gt[i, s_history_oid[i]] = 0

                if s[i] in o_history_sid[i]:
                    o_mask_gt[i, o_history_sid[i]] = 1
                else:
                    o_mask_gt[i, :] = 1
                    o_mask_gt[i, o_history_sid[i]] = 0

            s_total_loss3, sub_rank3 = self.link_predict(s_nce_loss, s_preds, s_ce_loss, s, o, r,
                                                         s_mask_gt, total_data, 's', True)
            o_total_loss3, obj_rank3 = self.link_predict(o_nce_loss, o_preds, o_ce_loss, o, s, r,
                                                         o_mask_gt, total_data, 'o', True)
            batch_loss3 = (s_total_loss3 + o_total_loss3) / 2.0

            return sub_rank1, obj_rank1, batch_loss1, \
                   sub_rank2, obj_rank2, batch_loss2, \
                   sub_rank3, obj_rank3, batch_loss3, \
                   (s_ce_all_acc + o_ce_all_acc) / 2

        elif mode_lk == 'Oracle':
            print('Oracle Training')
            s_ce_loss, _, _ = self.oracle_loss(s, r, self.rel_embeds[:self.num_rel],
                                               s_history_label_true, s_frequency_hidden)
            o_ce_loss, _, _ = self.oracle_loss(o, r, self.rel_embeds[self.num_rel:],
                                               o_history_label_true, o_frequency_hidden)
            return (s_ce_loss + o_ce_loss) / 2.0 + self.oracle_l1(0.01)

    def calculate_nce_loss(self, actor1, actor2, r, rel_embeds, linear1, linear2, linear3, history_tag, non_history_tag):
        preds_raw1 = self.tanh(linear1(
            self.dropout(torch.cat((self.init_ent_emb[actor1], rel_embeds[r]), dim=1))))
        preds1 = F.softmax(preds_raw1.mm(self.init_ent_emb.transpose(0, 1)) + history_tag, dim=1)

        preds_raw2 = self.tanh(linear2(
            self.dropout(torch.cat((self.init_ent_emb[actor1], rel_embeds[r]), dim=1))))
        preds2 = F.softmax(preds_raw2.mm(self.init_ent_emb.transpose(0, 1)) + non_history_tag, dim=1)

        gate = self.gate(actor1, r)
        fgate = open("gate.txt", "a+")
        print(gate,file=fgate)
        preds_two = (torch.mul(gate, preds1) + torch.mul(-gate + 1, preds2))

        nce = torch.sum(torch.gather(torch.log(preds_two), 1, actor2.view(-1, 1)))
        nce /= -1. * actor2.shape[0]
        if torch.any(torch.isnan(nce)):
            nce = torch.tensor(0.0, requires_grad=True)

        return nce, preds_two

    def contrastive_layer(self, x):
        x = self.contrastive_hidden_layer(x)
        return x

    def calculate_spc_loss(self, actor1, r, rel_embeds, targets, frequency_hidden):
        projections = self.contrastive_layer(
            torch.cat((self.init_ent_emb[actor1], rel_embeds[r], frequency_hidden), dim=1))
        targets = torch.squeeze(targets)
        """if np.random.randint(0, 10) < 1 and torch.sum(targets) / targets.shape[0] < 0.65 and torch.sum(targets) / targets.shape[0] > 0.35:
            np.savetxt("xx.tsv", projections.detach().cpu().numpy(), delimiter="\t")
            np.savetxt("yy.tsv", targets.detach().cpu().numpy(), delimiter="\t")
        """
        dot_product_tempered = torch.mm(projections, projections.T) / 1.0
        # Minus max for numerical stability with exponential. Same done in cross entropy. Epsilon added to avoid log(0)
        exp_dot_tempered = (
                torch.exp(dot_product_tempered - torch.max(dot_product_tempered, dim=1, keepdim=True)[0]) + 1e-5
        )
        mask_similar_class = to_device(targets.unsqueeze(1).repeat(1, targets.shape[0]) == targets)
        mask_anchor_out = to_device(1 - torch.eye(exp_dot_tempered.shape[0]))
        mask_combined = mask_similar_class * mask_anchor_out
        cardinality_per_samples = torch.sum(mask_combined, dim=1)
        log_prob = -torch.log(exp_dot_tempered / (torch.sum(exp_dot_tempered * mask_anchor_out, dim=1, keepdim=True)))
        supervised_contrastive_loss_per_sample = torch.sum(log_prob * mask_combined, dim=1) / cardinality_per_samples

        supervised_contrastive_loss = torch.mean(supervised_contrastive_loss_per_sample)

        if torch.any(torch.isnan(supervised_contrastive_loss)):
            return torch.tensor(0.0, requires_grad=True)
        return supervised_contrastive_loss

    def calculate_static_loss(self, static_emb, evolve_emb):
        # step = (self.angle * math.pi / 180) * (time_step + 1)
        step = (10 * math.pi / 180)
        sim_matrix = torch.sum(static_emb * F.normalize(evolve_emb), dim=1)
        mask = (math.cos(step) - sim_matrix) > 0
        loss_static = torch.sum(torch.masked_select(math.cos(step) - sim_matrix, mask))
        return loss_static

    def calculate_cand_loss(self, actor1, actor2, r, rel_embeds, preds, cand_layer, decoder, gru, gate):
        #! add a candidate graph based on the prediction result of preds1 and preds2 5&
        triples = [actor1, r]
        cand_graph = build_candidate_subgraph(self.num_e, triples, preds, self.args.k, 1)
        total_feature = cand_layer.forward(cand_graph, self.init_ent_emb, rel_embeds, self.args.k)
        avg_feature = torch.split_with_sizes(total_feature, cand_graph.batch_num_nodes().tolist())
        neigh_feats = torch.stack(avg_feature, dim=0).mean(dim=0)
        enhanced_ent_emb = neigh_feats
        # enhanced_ent_emb[actor1] = gate(torch.cat([enhanced_ent_emb[actor1], self.fact_time], dim=1))
        enhanced_ent_emb = gru(torch.cat([enhanced_ent_emb, self.init_ent_emb], dim=1), enhanced_ent_emb)

        entity_feature = F.normalize(enhanced_ent_emb)

        score_enhanced = F.softmax(decoder.forward(entity_feature, actor1, rel_embeds[r], self.fact_time), dim=1)

        preds_total = self.args.gamma * score_enhanced + (1 - self.args.gamma) * preds

        scores_enhanced = torch.log(score_enhanced)
        loss_cand = F.nll_loss(scores_enhanced, actor2) + self.regularization_loss(reg_param=0.01)

        pred_actor2 = torch.argmax(preds_total, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print('# Batch accuracy', accuracy)
        if torch.any(torch.isnan(loss_cand)):
            loss_cand = torch.tensor(0.0, requires_grad=True)
        return loss_cand, preds_total, entity_feature

    def oracle_loss(self, actor1, r, rel_embeds, history_label, frequency_hidden):
        history_label_pred = self.sigmoid(
            self.oracle_layer(torch.cat((self.init_ent_emb[actor1], rel_embeds[r], frequency_hidden), dim=1)))
        tmp_label = torch.squeeze(history_label_pred).clone().detach()
        tmp_label[torch.where(tmp_label > 0.5)[0]] = 1
        tmp_label[torch.where(tmp_label < 0.5)[0]] = 0
        # print('# Bias Ratio', torch.sum(tmp_label).item() / tmp_label.shape[0])
        ce_correct = torch.sum(torch.eq(tmp_label, torch.squeeze(history_label)))
        ce_accuracy = 1. * ce_correct.item() / tmp_label.shape[0]
        print('# CE Accuracy', ce_accuracy)
        ce_loss = self.crossEntropy(torch.squeeze(history_label_pred), torch.squeeze(history_label))
        return ce_loss, history_label_pred, ce_accuracy * tmp_label.shape[0]

    def link_predict(self, nce_loss, preds, ce_loss, actor1, actor2, r, trust_musk, all_triples, pred_known, oracle,
                     history_tag=None, case_study=False):
        if case_study:
            f = open("case_study.txt", "a+")
            entity2id, relation2id = get_entity_relation_set(self.args.dataset)

        if oracle:
            preds = torch.mul(preds, trust_musk)
            print('$Batch After Oracle accuracy:', end=' ')
        else:
            print('$Batch No Oracle accuracy:', end=' ')
        # compute the correct triples
        if case_study:
            fff  = open("hits10.txt", "a+")
            _, hits10 = torch.topk(preds, k=10)
            np.set_printoptions(threshold=np.inf)
            print(hits10,file=fff)

        pred_actor2 = torch.argmax(preds, dim=1)  # predicted result
        correct = torch.sum(torch.eq(pred_actor2, actor2))
        accuracy = 1. * correct.item() / actor2.shape[0]
        print(accuracy)
        # print('Batch Error', 1 - accuracy)

        total_loss = nce_loss + ce_loss

        ranks = []
        for i in range(preds.shape[0]):
            cur_s = actor1[i]
            cur_r = r[i]
            cur_o = actor2[i]
            if case_study:
                in_history = torch.where(history_tag[i] > 0)[0]
                not_in_history = torch.where(history_tag[i] < 0)[0]
                print('---------------------------', file=f)
                for hh in range(in_history.shape[0]):
                    print('his:', entity2id[in_history[hh].item()], file=f)

                print(pred_known,
                      'Truth:', entity2id[cur_s.item()], '--', relation2id[cur_r.item()], '--', entity2id[cur_o.item()],
                      'Prediction:', entity2id[pred_actor2[i].item()], file=f)

            o_label = cur_o
            ground = preds[i, cur_o].clone().item()
            if self.args.filtering:
                if pred_known == 's':
                    s_id = torch.nonzero(all_triples[:, 0] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 2]
                else:
                    s_id = torch.nonzero(all_triples[:, 2] == cur_s).view(-1)
                    idx = torch.nonzero(all_triples[s_id, 1] == cur_r).view(-1)
                    idx = s_id[idx]
                    idx = all_triples[idx, 0]

                preds[i, idx] = 0
                preds[i, o_label] = ground

            ob_pred_comp1 = (preds[i, :] > ground).data.cpu().numpy()
            ob_pred_comp2 = (preds[i, :] == ground).data.cpu().numpy()
            ranks.append(np.sum(ob_pred_comp1) + ((np.sum(ob_pred_comp2) - 1.0) / 2) + 1)
        sys.exit(0)
        return total_loss, ranks

    def regularization_loss(self, reg_param):
        regularization_loss = torch.mean(self.rel_embeds.pow(2)) + torch.mean(self.entity_embeds.pow(2))# + torch.mean(self.time_embeds.pow(2))
        return regularization_loss * reg_param

    def oracle_l1(self, reg_param):
        reg = 0
        for param in self.oracle_layer.parameters():
            reg += torch.sum(torch.abs(param))
        return reg * reg_param

    # contrastive
    def freeze_parameter(self):
        self.rel_embeds.requires_grad_(False)
        self.entity_embeds.requires_grad_(False)
        self.time_embeds.requires_grad_(False)
        self.init_ent_emb.requires_grad_(False)
        self.words_emb.requires_grad_(False)
        self.linear_pred_layer_s1.requires_grad_(False)
        self.linear_pred_layer_o1.requires_grad_(False)
        self.linear_pred_layer_s2.requires_grad_(False)
        self.linear_pred_layer_o2.requires_grad_(False)
        self.linear_pred_layer_s3.requires_grad_(False)
        self.linear_pred_layer_o3.requires_grad_(False)
        self.linear_frequency.requires_grad_(False)
        self.contrastive_hidden_layer.requires_grad_(False)
        self.contrastive_output_layer.requires_grad_(False)
        self.static_rgcn_layer.requires_grad_(False)
        self.time_encoder.requires_grad_(False)
        self.rgcn_t.requires_grad_(False)
        self.cand_layer_s.requires_grad_(False)
        self.cand_layer_o.requires_grad_(False)
        self.decoder1.requires_grad_(False)
        self.decoder2.requires_grad_(False)
        self.gru_s.requires_grad_(False)
        self.gru_o.requires_grad_(False)
        self.gate.requires_grad_(False)
        self.time_linear1.requires_grad_(False)
        self.time_linear2.requires_grad_(False)
        self.linear_er.requires_grad_(False)
        self.linear_ef.requires_grad_(False)
        self.linear_rf.requires_grad_(False)
        self.weight_t1.requires_grad_(False)
        self.weight_t2.requires_grad_(False)
        self.bias_t1.requires_grad_(False)
        self.bias_t2.requires_grad_(False)
        self.time_constraint.requires_grad_(False)
