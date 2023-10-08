import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  # Time Encoding proposed by TGAT
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(32, dimension)

    # self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
    #                                    .float().reshape(dimension, -1))
    # self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    # t = t.unsqueeze(1)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output

# class TimeEncode(torch.nn.Module):
#     '''
#     This class implemented the Bochner's time embedding
#     expand_dim: int, dimension of temporal entity embeddings
#     enitity_specific: bool, whether use entith specific freuency and phase.
#     num_entities: number of entities.
#     '''

#     def __init__(self, expand_dim, entity_specific=False, num_entities=None, device='cpu'):
#         """
#         :param expand_dim: number of samples draw from p(w), which are used to estimate kernel based on MCMC
#         :param entity_specific: if use entity specific time embedding
#         :param num_entities: number of entities
#         refer to Self-attention with Functional Time Representation Learning for more detail
#         """
#         super(TimeEncode, self).__init__()
#         self.time_dim = expand_dim
#         self.entity_specific = entity_specific

#         if entity_specific:
#             self.basis_freq = torch.nn.Parameter(
#                 torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float().unsqueeze(dim=0).repeat(
#                     num_entities, 1))
#             self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float().unsqueeze(dim=0).repeat(num_entities, 1))
#         else:
#             self.basis_freq = torch.nn.Parameter(
#                 torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim)).float())  # shape: num_entities * time_dim
#             self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())

#     def forward(self, ts, entities=None):
#         '''
#         :param ts: [batch_size, seq_len]
#         :param entities: which entities do we extract their time embeddings.
#         :return: [batch_size, seq_len, time_dim]
#         '''
#         # ts.unsqueeze(1)
#         # print("Forward in TimeEncode: ts is on ", ts.get_device())
#         if self.entity_specific:
#             map_ts = ts * self.basis_freq[entities].unsqueeze(
#                 dim=1)  # self.basis_freq[entities]:  [batch_size, time_dim]
#             map_ts += self.phase[entities].unsqueeze(dim=1)
#         else:
#             map_ts = ts * self.basis_freq
#             map_ts += self.phase
#         harmonic = torch.cos(map_ts)
#         return harmonic