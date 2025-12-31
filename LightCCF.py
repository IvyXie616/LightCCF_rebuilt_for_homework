
# -*- coding: utf-8 -*-
"""
LightCCF (BaseReader version for ReChorus)
- Simple user/item embedding layer
- Loss: L = L_BPR + na_weight * L_NA + l2 * ||E||_2^2
- NA negative set B_p: other users' (v, i_v) positive pairs in the same batch

Drop this file into: src/models/general/LightCCF.py
and ensure src/models/general/__init__.py contains:
    import models.general.LightCCF as LightCCF
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.BaseModel import GeneralModel


class LightCCFBase(object):
    @staticmethod
    def parse_model_args(parser):
        # embedding
        parser.add_argument('--emb_size', type=int, default=64)
        # NA loss
        parser.add_argument('--na_weight', type=float, default=0.05,
                            help='weight for neighborhood aggregation loss L_NA')
        parser.add_argument('--na_temp', type=float, default=0.2,
                            help='temperature tau used in L_NA')
        return parser

    def _base_init(self, args, corpus):
        self.emb_size = args.emb_size
        self.dropout = args.dropout
        self.na_weight = getattr(args, 'na_weight', 0.05)
        self.na_temp = getattr(args, 'na_temp', 0.2)

        # "Simple embedding layer": map user/item id -> vector
        self.u_embeddings = nn.Embedding(corpus.n_users, self.emb_size)
        self.i_embeddings = nn.Embedding(corpus.n_items, self.emb_size)

        # init
        nn.init.xavier_uniform_(self.u_embeddings.weight)
        nn.init.xavier_uniform_(self.i_embeddings.weight)

        self.dropout_layer = nn.Dropout(p=self.dropout)

    def forward(self, feed_dict):
        """
        Expected feed_dict from BaseReader/BaseRunner:
          - user_id: [B]
          - item_id: [B, K] where K = 1 + num_neg (first column is positive)
        """
        user_id = feed_dict['user_id']                              # [B]
        item_id = feed_dict['item_id']                              # [B, K]

        u_vec = self.u_embeddings(user_id)                          # [B, D]
        i_vec = self.i_embeddings(item_id)                          # [B, K, D]
        u_vec = self.dropout_layer(u_vec)
        i_vec = self.dropout_layer(i_vec)

        # dot-product scores
        # [B, K] = sum over D of u_vec[:,None,:] * i_vec
        prediction = (u_vec.unsqueeze(1) * i_vec).sum(dim=-1)

        out = {
            'prediction': prediction,   # [B, K]
            'u_emb': u_vec,             # [B, D]
            'i_emb': i_vec,             # [B, K, D]
        }
        return out

    @staticmethod
    def _cosine_sim(a, b, eps=1e-8):
        # a: [B, D], b: [B, D] or [B, M, D]
        if b.dim() == 2:
            a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
            b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
            return (a_n * b_n).sum(dim=-1)  # [B]
        elif b.dim() == 3:
            a_n = a / (a.norm(dim=-1, keepdim=True) + eps)          # [B, D]
            b_n = b / (b.norm(dim=-1, keepdim=True) + eps)          # [B, M, D]
            return (a_n.unsqueeze(1) * b_n).sum(dim=-1)             # [B, M]
        else:
            raise ValueError("b must be 2D or 3D")

    def _bpr_loss(self, pred):
        """
        pred: [B, K], first column positive, remaining negatives
        """
        if pred.size(1) < 2:
            # fallback (shouldn't happen if num_neg >= 1)
            return pred.new_tensor(0.0)
        pos = pred[:, 0:1]             # [B, 1]
        neg = pred[:, 1:]              # [B, K-1]
        # -log(sigmoid(pos - neg)) averaged over negatives and batch
        loss = -F.logsigmoid(pos - neg).mean()
        return loss

    def _na_loss(self, u_emb, pos_i_emb):
        """
        Neighborhood Aggregation loss (InfoNCE-like):
          L_NA(u) = - sim(e_u, e_i)/tau + log sum_{p in B_p} exp(sim(e_u, e_p)/tau)

        We realize e_p as an embedding of a (v, i_v) pair from the same batch:
          e_p = normalize(e_v + e_{i_v})
        where v != u.

        Inputs:
          u_emb:     [B, D]  user embeddings for current batch
          pos_i_emb: [B, D]  positive item embeddings for current batch
        """
        B, D = u_emb.shape
        tau = self.na_temp

        # positive similarity sim(e_u, e_i)
        pos_sim = self._cosine_sim(u_emb, pos_i_emb)  # [B]

        # build B_p using "other users' positive pairs" within the same batch
        pair_emb = u_emb + pos_i_emb                  # [B, D]
        # sim(e_u, e_p) for all (u, p) pairs in batch -> [B, B]
        sim_mat = self._cosine_sim(u_emb, pair_emb.unsqueeze(0).expand(B, B, D))  # [B, B]

        # mask diagonal (exclude itself from negatives)
        mask = torch.eye(B, device=sim_mat.device, dtype=torch.bool)
        sim_mat = sim_mat.masked_fill(mask, float('-inf'))

        # logsumexp over negatives in the batch
        lse = torch.logsumexp(sim_mat / tau, dim=1)   # [B]

        loss_na = (-pos_sim / tau + lse).mean()
        return loss_na

    def loss(self, out_dict):
        """
        Called by runner (ReChorus BaseRunner calls model.loss(...) if present).
        """
        pred = out_dict['prediction']                 # [B, K]
        u_emb = out_dict['u_emb']                     # [B, D]
        i_emb = out_dict['i_emb']                     # [B, K, D]

        bpr = self._bpr_loss(pred)

        # NA uses only positive item embeddings (first column)
        pos_i_emb = i_emb[:, 0, :]                    # [B, D]
        # if batch size is 1, NA isn't defined -> 0
        if u_emb.size(0) <= 1:
            na = bpr.new_tensor(0.0)
        else:
            na = self._na_loss(u_emb, pos_i_emb)

        # L2 regularization already handled by GeneralModel in many configs,
        # but we keep it explicit to match L = ... + ||E||_2^2 (scaled by args.l2)
        l2 = getattr(self, 'l2', 0.0)
        reg = l2 * (u_emb.pow(2).mean() + i_emb.pow(2).mean())

        return bpr + self.na_weight * na + reg


class LightCCF(GeneralModel, LightCCFBase):
    """
    BaseReader + BaseRunner version for implicit CF datasets (e.g., Grocery_and_Gourmet_Food).
    """
    reader = 'BaseReader'
    runner = 'BaseRunner'
    extra_log_args = ['emb_size', 'batch_size', 'na_weight', 'na_temp', 'dropout']

    @staticmethod
    def parse_model_args(parser):
        parser = LightCCFBase.parse_model_args(parser)
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        GeneralModel.__init__(self, args, corpus)
        self._base_init(args, corpus)

    def forward(self, feed_dict):
        return LightCCFBase.forward(self, feed_dict)

    def loss(self, out_dict):
        return LightCCFBase.loss(self, out_dict)