# *_*coding:utf-8 *_*

import os
import numpy as np
import torch
import argparse
import dgl
import random
from collections import defaultdict

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1]), int(line_split[2])


def load_quadruples(inPath, fileName, fileName2=None, fileName3=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)

    if fileName3 is not None:
        with open(os.path.join(inPath, fileName3), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)


def make_batch(a, b, c, d, e, f, g, batch_size, valid1=None, valid2=None):
    # idx = [_ for _ in range(0, len(a), batch_size)]
    # random.shuffle(idx)
    #* new
    # count_list = []
    # count_num = 0
    # count_time = 0
    # for jj in enumerate(a):
    #     if jj[0] == 0 and jj[1][3] != 0:
    #         count_time = jj[1][3]
    #     if jj[1][3] == count_time:
    #         count_num = count_num + 1
    #         # print(jj[1][1])
    #     else:
    #         count_time = jj[1][3]
    #         count_list.append(count_num)
    #         count_num = 1
    # count_list.append(count_num)
    # if valid1 is None and valid2 is None:
    #     k = 0
    #     for i in range(0, len(count_list)):
    #     # for i in idx:
    #         print(count_list[i])
    #         boom = count_list[i]
    #         yield [a[k:k + boom], b[k:k + boom], c[k:k + boom],
    #                d[k:k + boom], e[k:k + boom], f[k:k + boom], g[k:k + boom]]
    #         k = k + count_list[i]
    # else:
    #     k = 0
    #     for i in range(0, len(count_list)):
    #     # for i in idx:
    #         boom = count_list[i]
    #         yield [a[k:k + boom], b[k:k + boom], c[k:k + boom],
    #                d[k:k + boom], e[k:k + boom], f[k:k + boom], g[k:k + boom],
    #                valid1[k:k + boom], valid2[k:k + boom]]
    #         k = k + count_list[i]
    #*raw
    if valid1 is None and valid2 is None:
        for i in range(0, len(a), batch_size):
        # for i in idx:
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size]]
    else:
        for i in range(0, len(a), batch_size):
        # for i in idx:
            yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
                   d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],
                   valid1[i:i + batch_size], valid2[i:i + batch_size]]

# def make_batch(a, b, c, d, e, f, g, batch_size, valid1=None, valid2=None):
#     idx = [_ for _ in range(0, len(a), batch_size)]
#     random.shuffle(idx)
#     if valid1 is None and valid2 is None:
#         for i in range(0, len(a), batch_size):
#         # for i in idx:
#             yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
#                    d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size]]
#     else:
#         for i in range(0, len(a), batch_size):
#         # for i in idx:
#             yield [a[i:i + batch_size], b[i:i + batch_size], c[i:i + batch_size],
#                    d[i:i + batch_size], e[i:i + batch_size], f[i:i + batch_size], g[i:i + batch_size],
#                    valid1[i:i + batch_size], valid2[i:i + batch_size]]

# def make_batch(a, b, c, d, e, f, g, batch_size, valid1=None, valid2=None):
#     ii = [_ for _ in range(0, len(a))]
#     random.shuffle(ii)
#     if valid1 is None and valid2 is None:
#         for idx in range(0, len(a), batch_size):
#             yield [a[ii[idx:idx+batch_size]], b[ii[idx:idx+batch_size]], c[ii[idx:idx+batch_size]],
#                    d[ii[idx:idx+batch_size]], e[ii[idx:idx+batch_size]], f[ii[idx:idx+batch_size]], g[ii[idx:idx+batch_size]]]
#     else:
#         for idx in range(0, len(a), batch_size):
#             yield [a[ii[idx:idx+batch_size]], b[ii[idx:idx+batch_size]], c[ii[idx:idx+batch_size]],
#                    d[ii[idx:idx+batch_size]], e[ii[idx:idx+batch_size]], f[ii[idx:idx+batch_size]], g[ii[idx:idx+batch_size]],
#                    valid1[ii[idx:idx+batch_size]], valid2[ii[idx:idx+batch_size]]]

def to_device(tensor):
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor.cpu()


def isListEmpty(inList):
    if isinstance(inList, list):
        return all(map(isListEmpty, inList))
    return False


def get_sorted_s_r_embed_limit(s_hist, s, r, ent_embeds, limit):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]
    s_len_non_zero = torch.where(s_len_non_zero > limit, to_device(torch.tensor(limit)), s_len_non_zero)

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist[-limit:]:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def get_sorted_s_r_embed(s_hist, s, r, ent_embeds):
    s_hist_len = to_device(torch.LongTensor(list(map(len, s_hist))))
    s_len, s_idx = s_hist_len.sort(0, descending=True)
    num_non_zero = len(torch.nonzero(s_len))
    s_len_non_zero = s_len[:num_non_zero]

    s_hist_sorted = []
    for idx in s_idx[:num_non_zero]:
        s_hist_sorted.append(s_hist[idx.item()])

    flat_s = []
    len_s = []

    for hist in s_hist_sorted:
        for neighs in hist:
            len_s.append(len(neighs))
            for neigh in neighs:
                flat_s.append(neigh[1])
    s_tem = s[s_idx]
    r_tem = r[s_idx]

    embeds = ent_embeds[to_device(torch.LongTensor(flat_s))]
    embeds_split = torch.split(embeds, len_s)
    """
    s_idx: id of descending by length in original list.  1 * batch
    s_len_non_zero: number of events having history  any
    s_tem: sorted s by length  batch
    r_tem: sorted r by length  batch
    embeds: event->history->neighbor
    lens_s: event->history_neighbor length
    embeds_split split by history neighbor length
    s_hist_dt_sorted: history interval sorted by history length without non
    """
    return s_idx, s_len_non_zero, s_tem, r_tem, embeds, len_s, embeds_split


def str2bool(v: str) -> bool:
    v = v.lower()
    if v == "true":
        return True
    elif v == "false":
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected, got" + str(v) + ".")


def write2file(s_ranks, o_ranks, all_ranks, file_test):
    s_ranks = np.asarray(s_ranks)
    s_mr_lk = np.mean(s_ranks)
    s_mrr_lk = np.mean(1.0 / s_ranks)

    print("Subject test MRR (lk): {:.6f}".format(s_mrr_lk))
    print("Subject test MR (lk): {:.6f}".format(s_mr_lk))
    file_test.write("Subject test MRR (lk): {:.6f}".format(s_mrr_lk) + '\n')
    file_test.write("Subject test MR (lk): {:.6f}".format(s_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_sub_lk = np.mean((s_ranks <= hit))
        print("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk))
        file_test.write("Subject test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_sub_lk) + '\n')

    o_ranks = np.asarray(o_ranks)
    o_mr_lk = np.mean(o_ranks)
    o_mrr_lk = np.mean(1.0 / o_ranks)

    print("Object test MRR (lk): {:.6f}".format(o_mrr_lk))
    print("Object test MR (lk): {:.6f}".format(o_mr_lk))
    file_test.write("Object test MRR (lk): {:.6f}".format(o_mrr_lk) + '\n')
    file_test.write("Object test MR (lk): {:.6f}".format(o_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_obj_lk = np.mean((o_ranks <= hit))
        print("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk))
        file_test.write("Object test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_obj_lk) + '\n')

    all_ranks = np.asarray(all_ranks)
    all_mr_lk = np.mean(all_ranks)
    all_mrr_lk = np.mean(1.0 / all_ranks)

    print("ALL test MRR (lk): {:.6f}".format(all_mrr_lk))
    print("ALL test MR (lk): {:.6f}".format(all_mr_lk))
    file_test.write("ALL test MRR (lk): {:.6f}".format(all_mrr_lk) + '\n')
    file_test.write("ALL test MR (lk): {:.6f}".format(all_mr_lk) + '\n')
    for hit in [1, 3, 10]:
        avg_count_all_lk = np.mean((all_ranks <= hit))
        print("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk))
        file_test.write("ALL test Hits (lk) @ {}: {:.6f}".format(hit, avg_count_all_lk) + '\n')
    return all_mrr_lk

def comp_deg_norm(g):
    in_deg = g.in_degrees(range(g.number_of_nodes())).float()
    in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def build_candidate_subgraph(
    num_nodes: int,
    total_triples: np.array,
    total_obj_logit: torch.Tensor,
    k: int,
    num_partitions: int,
) -> dgl.DGLGraph:
    # if pred_sub:
    #     total_obj = total_triples[0]
    #     # total_sub = total_sub_emb[total_triples[:, 0]].unsqueeze(1)
    #     total_rel = total_triples[1]
    #     # total_rel = total_rel_emb[total_triples[:, 1]].unsqueeze(1)

    #     num_queries = total_obj.size(0)
    #     # k = int(num_queries/2)
    #     _, total_topk_sub = torch.topk(total_obj_logit, k=k)
    #     rng = torch.Generator().manual_seed(1234)
    #     total_indices = torch.randperm(num_queries, generator=rng)

    #     graph_list = []
    #     for indices in torch.tensor_split(total_indices, num_partitions):
    #         topk_sub = total_topk_sub[indices]
    #         obj = torch.repeat_interleave(total_obj[indices], k)
    #         rel = torch.repeat_interleave(total_rel[indices], k)
    #         sub = topk_sub.view(-1)
    #         graph = dgl.graph(
    #             (sub, obj),
    #             num_nodes=num_nodes,
    #             device=total_obj.device,
    #         )
    #         graph.ndata["eid"] = torch.arange(num_nodes, device=graph.device)
    #         graph.edata["rid"] = rel
    #         norm = comp_deg_norm(graph)
    #         graph.ndata['norm'] = norm.view(-1, 1)
    #         # graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    #         graph_list.append(graph)
    # else:
    total_sub = total_triples[0]
    # total_sub = total_sub_emb[total_triples[:, 0]].unsqueeze(1)
    total_rel = total_triples[1]
    # total_rel = total_rel_emb[total_triples[:, 1]].unsqueeze(1)

    num_queries = total_sub.size(0)
    # k = int(num_queries/2)
    _, total_topk_obj = torch.topk(total_obj_logit, k=k)
    rng = torch.Generator().manual_seed(1234)
    total_indices = torch.randperm(num_queries, generator=rng)

    graph_list = []
    for indices in torch.tensor_split(total_indices, num_partitions):
        topk_obj = total_topk_obj[indices]
        sub = torch.repeat_interleave(total_sub[indices], k)
        rel = torch.repeat_interleave(total_rel[indices], k)
        obj = topk_obj.view(-1)
        graph = dgl.graph(
            (sub, obj),
            num_nodes=num_nodes,
            device=total_sub.device,
        )
        graph.ndata["eid"] = torch.arange(num_nodes, device=graph.device)
        graph.edata["rid"] = rel
        norm = comp_deg_norm(graph)
        graph.ndata['norm'] = norm.view(-1, 1)
        graph.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
        graph_list.append(graph)
    return dgl.batch(graph_list)


def r2e(triplets, num_rels):
    src, rel, dst = triplets.transpose()
    # get all relations
    uniq_r = np.unique(rel)
    uniq_r = np.concatenate((uniq_r, uniq_r+num_rels))
    # generate r2e
    r_to_e = defaultdict(set)
    for j, (src, rel, dst) in enumerate(triplets):
        r_to_e[rel].add(src)
        r_to_e[rel+num_rels].add(src)
    r_len = []
    e_idx = []
    idx = 0
    for r in uniq_r:
        r_len.append((idx,idx+len(r_to_e[r])))
        e_idx.extend(list(r_to_e[r]))
        idx += len(r_to_e[r])
    return uniq_r, r_len, e_idx

def build_sub_graph(num_nodes, num_rels, triples, use_cuda, gpu):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    # print(triples.shape)
    triples = triples[:, :3]
    src, rel, dst = triples.transpose()
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))

    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    g.ndata.update({'id': node_id, 'norm': norm.view(-1, 1)})
    g.apply_edges(lambda edges: {'norm': edges.dst['norm'] * edges.src['norm']})
    g.edata['type'] = torch.LongTensor(rel)

    uniq_r, r_len, r_to_e = r2e(triples, num_rels)
    g.uniq_r = uniq_r
    g.r_to_e = r_to_e
    g.r_len = r_len
    if use_cuda:
        g.to(gpu)
        g.r_to_e = torch.from_numpy(np.array(r_to_e)).long()
    return g

#* future work T graph
def build_time_graph(timestamps, r_types, r_num, period):
    def comp_deg_norm(g):
        in_deg = g.in_degrees(range(g.number_of_nodes())).float()
        in_deg[torch.nonzero(in_deg == 0).view(-1)] = 1
        norm = 1.0 / in_deg
        return norm
    t_id = torch.arange(0, timestamps, dtype=torch.long).view(-1, 1)
    # r1 = r_types[0]
    # r2 = r_types[1]
    # period1 = period[0]
    # period2 = period[1]
    g = dgl.DGLGraph()
    g.add_nodes(timestamps)
    src = []
    dst = []
    rel = []
    for i in range(0, len(r_types)):
        r = r_types[i]
        p = period[i]
        for ii in range(0, timestamps, p):
            if ii+p < timestamps:
                src.append(ii)
                dst.append(ii+p)
                rel.append(r)
    # for j in range(0, timestamps, period2):
    #     if j+period2 < timestamps:
    #         src.append(j)
    #         dst.append(j+period2)
    #         rel.append(r2)
    src = np.array(src)
    dst = np.array(dst)
    rel = np.array(rel)
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + r_num))
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    g.ndata.update({'id': t_id, 'norm': norm.view(-1, 1)})
    g.edata['type'] = torch.LongTensor(rel)

    return g

def get_entity_relation_set(dataset):
    inPath = './data/' + dataset
    entity_file = 'entity2id.txt'
    relation_file = 'relation2id.txt'
    with open(os.path.join(inPath, entity_file), 'r') as fr:
        entity = []
        for line in fr:
            line_split = line.split()
            head = int(line_split[-1])
            entity.append([head])

    with open(os.path.join(inPath, relation_file), 'r') as fr:
        relation = []
        for line in fr:
            line_split = line.split()
            head = int(line_split[-1])
            relation.append([head])

    return np.asarray(entity), np.asarray(relation)