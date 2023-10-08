import random

from torch.nn import functional as F
import torch
from torch.nn.parameter import Parameter
import math
import os

class TimeConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(TimeConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(3, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(3)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, eid, emb_rel, emb_time, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        # e1_embedded_all = embedding
        # emb_pos = F.tanh(emb_pos)
        e1_embedded = e1_embedded_all[eid].unsqueeze(1)  # batch_size,1,h_dim
        batch_size = e1_embedded.shape[0]
        rel_embedded = emb_rel.unsqueeze(1)  # batch_size,1,h_dim
        time_emb = emb_time.unsqueeze(1)
        # print(e1_embedded.shape)
        # print(rel_embedded.shape)
        # stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # batch_size,2,h_dim
        stacked_inputs = torch.cat([e1_embedded, rel_embedded, time_emb], 1)  # batch_size,2,h_dim
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x

class ConvTransE(torch.nn.Module):
    def __init__(self, num_entities, embedding_dim, input_dropout=0, hidden_dropout=0, feature_map_dropout=0, channels=50, kernel_size=3, use_bias=True):

        super(ConvTransE, self).__init__()

        self.inp_drop = torch.nn.Dropout(input_dropout)
        self.hidden_drop = torch.nn.Dropout(hidden_dropout)
        self.feature_map_drop = torch.nn.Dropout(feature_map_dropout)
        self.loss = torch.nn.BCELoss()

        self.conv1 = torch.nn.Conv1d(2, channels, kernel_size, stride=1,
                               padding=int(math.floor(kernel_size / 2)))
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.bn2 = torch.nn.BatchNorm1d(embedding_dim)
        self.register_parameter('b', Parameter(torch.zeros(num_entities)))
        self.fc = torch.nn.Linear(embedding_dim * channels, embedding_dim)

    def forward(self, embedding, eid, emb_rel, nodes_id=None, mode="train", negative_rate=0, partial_embeding=None):
        e1_embedded_all = F.tanh(embedding)
        # e1_embedded_all = embedding
        # emb_pos = F.tanh(emb_pos)
        e1_embedded = e1_embedded_all[eid].unsqueeze(1)  # batch_size,1,h_dim
        batch_size = e1_embedded.shape[0]
        rel_embedded = emb_rel.unsqueeze(1)  # batch_size,1,h_dim
        # print(e1_embedded.shape)
        # print(rel_embedded.shape)
        # stacked_inputs = torch.cat([e1_embedded, emb_time], 1)  # batch_size,2,h_dim
        stacked_inputs = torch.cat([e1_embedded, rel_embedded], 1)  # batch_size,2,h_dim
        stacked_inputs = self.bn0(stacked_inputs)  # batch_size,2,h_dim
        x = self.inp_drop(stacked_inputs)  # batch_size,2,h_dim
        x = self.conv1(x)  # batch_size,2,h_dim
        x = self.bn1(x)  # batch_size,channels,h_dim
        x = F.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(batch_size, -1)  # batch_size,channels*h_dim
        x = self.fc(x)  # batch_size,channels*h_dim
        x = self.hidden_drop(x)  # batch_size,h_dim
        if batch_size > 1:
            x = self.bn2(x)
        x = F.relu(x)
        if partial_embeding is None:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
        else:
            x = torch.mm(x, e1_embedded_all.transpose(1, 0))
            x = torch.mul(x, partial_embeding)
        return x