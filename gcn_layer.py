import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CandRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(CandRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        #* add attention
        self.fc = nn.Linear(in_feat, out_feat, bias=False)
        self.attn_fc = nn.Linear(out_feat, 1, bias=False)
        self.pos_proj = nn.Linear(2*in_feat, out_feat, bias=False)
        self.reset_parameters()

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(2 * self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
        self.weight_relation = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_relation, gain=nn.init.calculate_gain('relu'))
        self.time_weight = nn.Linear(2 * self.in_feat, self.out_feat)

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        channels = 1
        kernel_size = 1
        self.conv1d = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                                      padding=int(math.floor(kernel_size / 2)))
        self.gnn_bn0 = torch.nn.BatchNorm1d(2)

    def forward(self, g, prev_h, emb_rel, k):
        # with g.local_scope():
        g.ndata['h'] = prev_h
        self.rel_emb = emb_rel
        # self.time_emb = torch.repeat_interleave(emb_time, k, 0)
        # self.rel_emb = torch.mm(emb_rel, self.weight_relation)
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        g.apply_edges(self.edge_attention)
        # g.edata['e'] = self.edge_attention
        g.update_all(self.msg_func, self.reduce_func)
        node_repr = g.ndata['h']
        # edge_repr = g.edata['r']
        # edge_repr = torch.mm(edge_repr, self.weight_neighbor_r)
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        # g.edata['r'] = self.rel_emb
        # print('11',node_repr)
        # g.edata['r'] = edge_repr
        return node_repr#, self.rel_emb

    def msg_func(self, edges):
        relation = self.rel_emb.index_select(0, edges.data['rid']).view(-1, self.out_feat)
        # time_relation = self.time_weight(torch.cat([relation, self.time_emb], 1)).unsqueeze(1)
        # relation = edges.data['r'].view(-1, self.out_feat)
        # time_now = self.time_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        node = edges.src['h'].view(-1, self.out_feat)
        batch_size = node.shape[0]
        node = node.unsqueeze(1)
        relation = relation.unsqueeze(1)
        # stacked_inputs = torch.cat([node, time_relation], 1)
        stacked_inputs = torch.cat([node, relation], 1)

        # stacked_inputs = self.gnn_bn0(stacked_inputs)

        # msg = self.conv1d(stacked_inputs)
        msg = stacked_inputs
        msg = msg.view(batch_size, -1)

        # msg = self.proj_time(msg, self.time_emb)
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg, 'e': edges.data['e']}

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.fc_p.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        # nn.init.xavier_normal_(self.attn_fc_p.weight, gain=gain)
        nn.init.xavier_normal_(self.pos_proj.weight, gain=gain)

    def edge_attention(self, edges):
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        a = self.pos_proj(z2)
        return {'e': self.attn_fc(F.leaky_relu(a))}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1) # 归一化每一条入边的注意力系数
        h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
        return {'h': h}

#* regcn
    #     self.in_feat = in_feat
    #     self.out_feat = out_feat
    #     self.bias = bias
    #     self.activation = activation
    #     self.self_loop = self_loop
    #     self.num_rels = num_rels
    #     self.rel_emb = None
    #     self.skip_connect = skip_connect
    #     self.ob = None
    #     self.sub = None

    #     # WL
    #     self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
    #     nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

    #     if self.self_loop:
    #         self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
    #         nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
    #         self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
    #         nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

    #     if self.skip_connect:
    #         self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
    #         nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
    #         self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
    #         nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

    #     if dropout:
    #         self.dropout = nn.Dropout(dropout)
    #     else:
    #         self.dropout = None

    # def propagate(self, g):
    #     g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    # def forward(self, g, prev_h, emb_rel, k):
    #     g.ndata['h'] = prev_h
    #     self.rel_emb = emb_rel
    #     # self.sub = sub
    #     # self.ob = ob
    #     if self.self_loop:
    #         #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
    #         # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
    #         masked_index = torch.masked_select(
    #             torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
    #             (g.in_degrees(range(g.number_of_nodes())) > 0))
    #         loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
    #         loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
    #     if len(prev_h) != 0 and self.skip_connect:
    #         skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

    #     # calculate the neighbor message with weight_neighbor
    #     self.propagate(g)
    #     node_repr = g.ndata['h']

    #     # print(len(prev_h))
    #     if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
    #         if self.self_loop:
    #             node_repr = node_repr + loop_message
    #         node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
    #     else:
    #         if self.self_loop:
    #             node_repr = node_repr + loop_message

    #     if self.activation:
    #         node_repr = self.activation(node_repr)
    #     if self.dropout is not None:
    #         node_repr = self.dropout(node_repr)
    #     g.ndata['h'] = node_repr
    #     return node_repr

    # def msg_func(self, edges):
    #     # if reverse:
    #     #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
    #     # else:
    #     #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
    #     relation = self.rel_emb.index_select(0, edges.data['rid']).view(-1, self.out_feat)
    #     node = edges.src['h'].view(-1, self.out_feat)
    #     # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
    #     #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
    #     # node = torch.matmul(node, self.sub)

    #     # after add inverse edges, we only use message pass when h as tail entity
    #     # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
    #     # msg = torch.cat((node, relation), dim=1)
    #     msg = node + relation
    #     # calculate the neighbor message with weight_neighbor
    #     msg = torch.mm(msg, self.weight_neighbor)
    #     return {'msg': msg}

    # def apply_func(self, nodes):
    #     return {'h': nodes.data['h'] * nodes.data['norm']}

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, bias=None, activation=None,
                 self_loop=False, skip_connect=False, dropout=0.0, layer_norm=False):
        super(RGCNLayer, self).__init__()
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.skip_connect = skip_connect
        self.layer_norm = layer_norm

        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            # self.loop_weight = nn.Parameter(torch.eye(out_feat), requires_grad=False)

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,
                                    gain=nn.init.calculate_gain('relu'))

            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

        if self.layer_norm:
            self.normalization_layer = nn.LayerNorm(out_feat, elementwise_affine=False)

    # define how propagation is done in subclass
    def propagate(self, g):
        raise NotImplementedError

    def forward(self, g, prev_h=[]):
        if self.self_loop:
            #print(self.loop_weight)
            loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            if self.dropout is not None:
                loop_message = self.dropout(loop_message)
        # self.skip_connect_weight.register_hook(lambda g: print("grad of skip connect weight: {}".format(g)))
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        self.propagate(g)

        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.bias
        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:   # 两次计算loop_message的方式不一样，前者激活后再加权
            previous_node_repr = (1 - skip_weight) * prev_h
            if self.activation:
                node_repr = self.activation(node_repr)
            if self.self_loop:
                if self.activation:
                    loop_message = skip_weight * self.activation(loop_message)
                else:
                    loop_message = skip_weight * loop_message
                node_repr = node_repr + loop_message
            node_repr = node_repr + previous_node_repr
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.layer_norm:
                node_repr = self.normalization_layer(node_repr)
            if self.activation:
                node_repr = self.activation(node_repr)
            # print("node_repr")
            # print(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

class RGCNBlockLayer(RGCNLayer):
    def __init__(self, in_feat, out_feat, num_rels, num_bases, bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, layer_norm=False):
        super(RGCNBlockLayer, self).__init__(in_feat, out_feat, bias,
                                             activation, self_loop=self_loop, skip_connect=skip_connect,
                                             dropout=dropout)
        self.num_rels = num_rels
        self.num_bases = num_bases

        assert self.num_bases > 0

        self.out_feat = out_feat
        self.submat_in = in_feat // self.num_bases
        self.submat_out = out_feat // self.num_bases

        # assuming in_feat and out_feat are both divisible by num_bases
        self.weight = nn.Parameter(torch.Tensor(
            self.num_rels, self.num_bases * self.submat_in * self.submat_out))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def msg_func(self, edges):
        weight = self.weight.index_select(0, edges.data['type']).view(
                    -1, self.submat_in, self.submat_out)    # [edge_num, submat_in, submat_out]
        node = edges.src['h'].view(-1, 1, self.submat_in)   # [edge_num * num_bases, 1, submat_in]->
        msg = torch.bmm(node, weight).view(-1, self.out_feat)   # [edge_num, out_feat]
        return {'msg': msg}

    def propagate(self, g):
        g.update_all(self.msg_func, fn.sum(msg='msg', out='h'), self.apply_func)

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

# class UnionRGCNLayer(nn.Module):
#     def __init__(self, in_feat, out_feat, num_rels, bias=None,
#                  activation=None, self_loop=False, dropout=0.0, skip_connect=False):
#         super(UnionRGCNLayer, self).__init__()

#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.bias = bias
#         self.activation = activation
#         self.self_loop = self_loop
#         self.num_rels = num_rels
#         self.rel_emb = None
#         self.skip_connect = skip_connect
#         self.ob = None
#         self.sub = None

#         #* add attention
#         self.fc = nn.Linear(in_feat, out_feat, bias=False)
#         self.attn_fc = nn.Linear(2*out_feat, 1, bias=False)
#         self.pos_proj = nn.Linear(2*in_feat, 2*out_feat, bias=False)
#         self.reset_parameters()

#         # WL
#         self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
#         nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

#         # Time Weight
#         self.time_constraints = nn.Linear(2*self.in_feat, self.out_feat, bias=True)
#         nn.init.xavier_uniform_(self.time_constraints.weight)

#         self.fact_evolve_weight = nn.Linear(self.in_feat, self.out_feat, bias=True)
#         nn.init.xavier_uniform_(self.fact_evolve_weight.weight)


#         if self.self_loop:
#             self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
#             nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
#             self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
#             nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

#         if self.skip_connect:
#             self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
#             nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
#             self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
#             nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

#         if dropout:
#             self.dropout = nn.Dropout(dropout)
#         else:
#             self.dropout = None

#         channels = 1
#         kernel_size = 1
#         self.conv1d = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
#                                       padding=int(math.floor(kernel_size / 2)))
#         self.fc = torch.nn.Linear(self.out_feat * channels, self.out_feat)
#         self.gnn_bn0 = torch.nn.BatchNorm1d(2)
#         self.gnn_bn1 = torch.nn.BatchNorm1d(channels)
#         self.gnn_bn2 = torch.nn.BatchNorm1d(self.out_feat)

#     def forward(self, g, prev_h, emb_rel):
#         self.rel_emb = emb_rel
#         # self.sub = sub
#         # self.ob = ob
#         if self.self_loop:
#             masked_index = torch.masked_select(
#                 torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
#                 (g.in_degrees(range(g.number_of_nodes())) > 0))
#             loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
#             loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
#         if len(prev_h) != 0 and self.skip_connect:
#             skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

#         # calculate the neighbor message with weight_neighbor
#         g.apply_edges(self.edge_attention)
#         g.update_all(self.msg_func, self.reduce_func)
#         node_repr = g.ndata['h']

#         if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
#             if self.self_loop:
#                 node_repr = node_repr + loop_message
#             node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
#         else:
#             if self.self_loop:
#                 node_repr = node_repr + loop_message

#         if self.activation:
#             node_repr = self.activation(node_repr)
#         if self.dropout is not None:
#             node_repr = self.dropout(node_repr)
#         g.ndata['h'] = node_repr
#         return node_repr

#     def msg_func(self, edges):
#         relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
#         node = edges.src['h'].view(-1, self.out_feat)
#         batch_size = node.shape[0]
#         node = node.unsqueeze(1)
#         relation = relation.unsqueeze(1)
#         # stacked_inputs = torch.cat([node, relation, t1], 1)
#         stacked_inputs = torch.cat([node, relation], 1)

#         stacked_inputs = self.gnn_bn0(stacked_inputs)

#         msg = self.conv1d(stacked_inputs)
#         msg = msg.view(batch_size, -1)
#         msg = torch.mm(msg, self.weight_neighbor)
#         return {'msg': msg, 'e': edges.data['e']}

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         # nn.init.xavier_normal_(self.fc_p.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#         # nn.init.xavier_normal_(self.attn_fc_p.weight, gain=gain)
#         nn.init.xavier_normal_(self.pos_proj.weight, gain=gain)

#     def edge_attention(self, edges):
#         z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
#         a = self.pos_proj(z2)
#         # a = z2
#         return {'e': self.attn_fc(F.leaky_relu(a))}

#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['e'], dim=1) # 归一化每一条入边的注意力系数
#         h = torch.sum(alpha * nodes.mailbox['msg'], dim=1)
#         return {'h'}

class UnionRGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
                 activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
        super(UnionRGCNLayer, self).__init__()

        self.in_feat = in_feat
        self.out_feat = out_feat
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.num_rels = num_rels
        self.rel_emb = None
        self.skip_connect = skip_connect
        self.ob = None
        self.sub = None

        # WL
        self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
        nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))

        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
            self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

        if self.skip_connect:
            self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
            nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
            self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

        if dropout:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None

    def propagate(self, g):
        g.update_all(lambda x: self.msg_func(x), fn.sum(msg='msg', out='h'), self.apply_func)

    def forward(self, g, prev_h, emb_rel):
        self.rel_emb = emb_rel
        # self.sub = sub
        # self.ob = ob
        if self.self_loop:
            #loop_message = torch.mm(g.ndata['h'], self.loop_weight)
            # masked_index = torch.masked_select(torch.arange(0, g.number_of_nodes(), dtype=torch.long), (g.in_degrees(range(g.number_of_nodes())) > 0))
            masked_index = torch.masked_select(
                torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
                (g.in_degrees(range(g.number_of_nodes())) > 0))
            loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
            loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
        if len(prev_h) != 0 and self.skip_connect:
            skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

        # calculate the neighbor message with weight_neighbor
        self.propagate(g)
        node_repr = g.ndata['h']

        # print(len(prev_h))
        if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
            if self.self_loop:
                node_repr = node_repr + loop_message
            node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
        else:
            if self.self_loop:
                node_repr = node_repr + loop_message

        if self.activation:
            node_repr = self.activation(node_repr)
        if self.dropout is not None:
            node_repr = self.dropout(node_repr)
        g.ndata['h'] = node_repr
        return node_repr

    def msg_func(self, edges):
        # if reverse:
        #     relation = self.rel_emb.index_select(0, edges.data['type_o']).view(-1, self.out_feat)
        # else:
        #     relation = self.rel_emb.index_select(0, edges.data['type_s']).view(-1, self.out_feat)
        relation = self.rel_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
        edge_type = edges.data['type']
        edge_num = edge_type.shape[0]
        node = edges.src['h'].view(-1, self.out_feat)
        # node = torch.cat([torch.matmul(node[:edge_num // 2, :], self.sub),
        #                  torch.matmul(node[edge_num // 2:, :], self.ob)])
        # node = torch.matmul(node, self.sub)

        # after add inverse edges, we only use message pass when h as tail entity
        # 这里计算的是每个节点发出的消息，节点发出消息时其作为头实体
        # msg = torch.cat((node, relation), dim=1)
        msg = node + relation
        # calculate the neighbor message with weight_neighbor
        msg = torch.mm(msg, self.weight_neighbor)
        return {'msg': msg}

    def apply_func(self, nodes):
        return {'h': nodes.data['h'] * nodes.data['norm']}

# class CandRGCNLayer(nn.Module):
#     def __init__(self, in_feat, out_feat, num_rels, num_bases=-1,  bias=None,
#                  activation=None, self_loop=False, dropout=0.0, skip_connect=False, rel_emb=None):
#         super(CandRGCNLayer, self).__init__()

#         self.in_feat = in_feat
#         self.out_feat = out_feat
#         self.bias = bias
#         self.activation = activation
#         self.self_loop = self_loop
#         self.num_rels = num_rels
#         self.rel_emb = None
#         self.skip_connect = skip_connect
#         self.ob = None
#         self.sub = None

#         #* add attention
#         self.fc = nn.Linear(in_feat, out_feat, bias=False)
#         self.attn_fc = nn.Linear(out_feat, 1, bias=False)
#         self.attn_fc2 = nn.Linear(out_feat, 1, bias=False)
#         self.pos_proj = nn.Linear(2*in_feat, out_feat, bias=False)
#         self.pos_proj2 = nn.Linear(2*in_feat, out_feat, bias=False)
#         self.reset_parameters()

#         # WL
#         self.weight_neighbor = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
#         nn.init.xavier_uniform_(self.weight_neighbor, gain=nn.init.calculate_gain('relu'))
#         self.weight_relation = nn.Parameter(torch.Tensor(self.in_feat, self.out_feat))
#         nn.init.xavier_uniform_(self.weight_relation, gain=nn.init.calculate_gain('relu'))

#         if self.self_loop:
#             self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
#             nn.init.xavier_uniform_(self.loop_weight, gain=nn.init.calculate_gain('relu'))
#             self.evolve_loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
#             nn.init.xavier_uniform_(self.evolve_loop_weight, gain=nn.init.calculate_gain('relu'))

#         if self.skip_connect:
#             self.skip_connect_weight = nn.Parameter(torch.Tensor(out_feat, out_feat))   # 和self-loop不一样，是跨层的计算
#             nn.init.xavier_uniform_(self.skip_connect_weight,gain=nn.init.calculate_gain('relu'))
#             self.skip_connect_bias = nn.Parameter(torch.Tensor(out_feat))
#             nn.init.zeros_(self.skip_connect_bias)  # 初始化设置为0

#         if dropout:
#             self.dropout = nn.Dropout(dropout)
#         else:
#             self.dropout = None

#         channels = 1
#         kernel_size = 1
#         self.conv1d = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
#                                       padding=int(math.floor(kernel_size / 2)))
#         self.gnn_bn0 = torch.nn.BatchNorm1d(2)

#     def forward(self, g, prev_h, emb_rel):
#         # with g.local_scope():
#         # g.ndata['h'] = prev_h
#         self.rel_emb = emb_rel
#         g.edata['r'] = self.rel_emb[g.edata['rid']]
#         # self.rel_emb = torch.mm(emb_rel, self.weight_relation)
#         # self.sub = sub
#         # self.ob = ob
#         if self.self_loop:
#             masked_index = torch.masked_select(
#                 torch.arange(0, g.number_of_nodes(), dtype=torch.long).cuda(),
#                 (g.in_degrees(range(g.number_of_nodes())) > 0))
#             loop_message = torch.mm(g.ndata['h'], self.evolve_loop_weight)
#             loop_message[masked_index, :] = torch.mm(g.ndata['h'], self.loop_weight)[masked_index, :]
#         if len(prev_h) != 0 and self.skip_connect:
#             skip_weight = F.sigmoid(torch.mm(prev_h, self.skip_connect_weight) + self.skip_connect_bias)     # 使用sigmoid，让值在0~1

#         # calculate the neighbor message with weight_neighbor
#         g.apply_edges(self.edge_attention)
#         # g.edata['e'] = self.edge_attention
#         g.update_all(self.msg_func, self.reduce_func)
#         node_repr = g.ndata['h']
#         # edge_repr = g.edata['r']
#         # edge_repr = torch.mm(edge_repr, self.weight_neighbor_r)
#         # print(len(prev_h))
#         if len(prev_h) != 0 and self.skip_connect:  # 两次计算loop_message的方式不一样，前者激活后再加权
#             if self.self_loop:
#                 node_repr = node_repr + loop_message
#             node_repr = skip_weight * node_repr + (1 - skip_weight) * prev_h
#         else:
#             if self.self_loop:
#                 node_repr = node_repr + loop_message

#         if self.activation:
#             node_repr = self.activation(node_repr)
#         if self.dropout is not None:
#             node_repr = self.dropout(node_repr)
#         g.ndata['h'] = node_repr
#         # g.edata['r'] = self.rel_emb
#         # print('11',node_repr)
#         # g.edata['r'] = edge_repr
#         return node_repr#, self.rel_emb

#     def msg_func(self, edges):
#         relation = self.rel_emb.index_select(0, edges.data['rid']).view(-1, self.out_feat)
#         # relation = edges.data['r'].view(-1, self.out_feat)
#         # time_now = self.time_emb.index_select(0, edges.data['type']).view(-1, self.out_feat)
#         node = edges.src['h'].view(-1, self.out_feat)
#         batch_size = node.shape[0]
#         node = node.unsqueeze(1)
#         relation = relation.unsqueeze(1)
#         stacked_inputs = torch.cat([node, relation], 1)

#         stacked_inputs = self.gnn_bn0(stacked_inputs)

#         msg = self.conv1d(stacked_inputs)
#         msg = msg.view(batch_size, -1)

#         # msg = self.proj_time(msg, self.time_emb)
#         msg = torch.mm(msg, self.weight_neighbor)
#         return {'msg': msg, 'ea': edges.data['ea'], 'eb': edges.data['eb']}

#     def reset_parameters(self):
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.fc.weight, gain=gain)
#         # nn.init.xavier_normal_(self.fc_p.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
#         nn.init.xavier_normal_(self.attn_fc2.weight, gain=gain)
#         # nn.init.xavier_normal_(self.attn_fc_p.weight, gain=gain)
#         nn.init.xavier_normal_(self.pos_proj.weight, gain=gain)
#         nn.init.xavier_normal_(self.pos_proj2.weight, gain=gain)

#     def edge_attention(self, edges):
#         z2 = torch.cat([edges.src['h'], edges.data['r']], dim=1)
#         a = self.pos_proj(z2)
#         z3 = torch.cat([a, edges.dst['h']], dim=1)
#         b = self.pos_proj2(z3)
#         return {'ea': self.attn_fc(F.leaky_relu(a)), 'eb': self.attn_fc2(F.leaky_relu(b))}

#     def reduce_func(self, nodes):
#         alpha = F.softmax(nodes.mailbox['ea'], dim=1) # 归一化每一条入边的注意力系数
#         beta = F.softmax(nodes.mailbox['eb'], dim=1)
#         h = torch.sum(alpha * beta * nodes.mailbox['msg'], dim=1)
#         return {'h': h}