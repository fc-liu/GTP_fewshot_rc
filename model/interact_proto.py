import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from configure import FLAGS
import framework
import utils
from model.relation_representation_model import RRModel
from model.utils import *
import json
import os


class InteractiveContrastiveNet(framework.FewShotREModel):
    def __init__(self, tokenizer, embedder, rel_rep_model, max_length):
        super(InteractiveContrastiveNet, self).__init__(None)
        self.tokenizer = tokenizer
        self.rr_model = RRModel(embedder, rel_rep_model)
        self.seg_num_emb = FLAGS.seg_emb_size
        self.instance_emb_size = self.rr_model.output_size
        self.hidden_size = self.instance_emb_size+self.seg_num_emb
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=FLAGS.n_head)
        # encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.instance_emb_size, nhead=4)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=FLAGS.layer)

        self.cont_net = nn.Linear(2*self.instance_emb_size, 1)
        self.active = nn.Sigmoid()

        self.seg_embedding = nn.Embedding(2, self.seg_num_emb)

        self.q_seg_idx = torch.LongTensor([19])
        self.seg_idxs = None
        self.drop = nn.Dropout(1-FLAGS.dropout_keep_prob)
        self.layNorm = nn.LayerNorm(self.instance_emb_size)
        self.layNorm_hidden = nn.LayerNorm(self.hidden_size)
        self.seg_norm = nn.LayerNorm(self.seg_num_emb)
        self.N = 0
        self.K = 0
        # self.layNorm_out = nn.LayerNorm(self.hidden_size)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def encoder(self, tokens, pos1, pos2, mask):

        hidden = self.rr_model(tokens, pos1, pos2, mask)
        # hidden = self.drop(hidden)
        # hidden = self.FC(hidden)
        # hidden = torch.tanh(hidden)
        return hidden

    def forward(self, support, query, B, N, K, total_Q):
        """
        batch_samples: B*Q*N*K*seq_length
        """
        support = self.encoder(*support)
        query = self.encoder(*query)
        support = self.layNorm(support)
        query = self.layNorm(query)
        # support = self.drop(support)
        # query = self.drop(query)
        # (B, N, K, D)
        batch_support = support.view(B, N, K, self.instance_emb_size)
        # batch_support = batch_support.view(B, N*(K), self.instance_emb_size)
        # (B, tot_Q, D)
        batch_query = query.view(B, total_Q, self.instance_emb_size)

        if self.seg_idxs is None or K != self.K or N != self.N:
            seg_idxs = torch.arange(N).expand(K, N).t().reshape(-1)
            seg_idxs = torch.cat(
                (seg_idxs, self.q_seg_idx))
            self.seg_idxs = seg_idxs.expand(
                B, N*(K)+1).cuda(FLAGS.paral_cuda[0])
        seg_emb = self.seg_embedding(self.seg_idxs)  # (B, N*K+1, seg_emb_D)
        seg_emb = self.seg_norm(seg_emb)

        # pos_emb = self.pos_embedding(self.pos_idxs)
        logits_list = []
        for i in range(total_Q):
            sing_query = batch_query[:, i:i+1, :]  # (B, 1，D)
            sing_query = sing_query.repeat(1, N, 1).reshape(
                B, N, 1, -1)  # (B, N, 1, D)
            instances = torch.cat(
                (batch_support, sing_query), dim=2)  # (B, N, K+1, D)
            instances = instances.reshape(B, N*(K+1), -1)
            instances = torch.cat((instances, seg_emb), dim=-1)
            # (B,N*K+2, hidden_size)

            input_emb = instances

            input_emb = input_emb.transpose_(1, 0)  # (N*(K+1), B, hidden_size)
            hidden = self.transformer(input_emb)  # (N*(K+1), B, hidden_size)
            hidden = hidden.transpose_(1, 0)  # (B,N*(K+1), hidden_size)

            hidden = hidden.reshape(B, N, K+1, -1)

            support = hidden[:, :, :K]  # (B, N, K, D)
            # (B, N, D) the sep tensor
            query = hidden[:, :, -1:].reshape(B, N, -1)

            support = torch.mean(support, 2)  # (B, N, D)

            support = self.drop(support)
            query = self.drop(query)
            logits = -self.__dist__(support, query, 2)  # (B, N)
            # logits=logits.reshape(B,1,N,K)
            # logits,_=torch.max()

            logits_list.append(logits)
        logits = torch.stack(logits_list, dim=1)  # (B, total_Q, N)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred


class Proto(framework.FewShotREModel):

    def __init__(self, tokenizer, embedder, rel_rep_model, max_length):
        super(Proto, self).__init__(None)
        self.sentence_encoder = RRModel(embedder, rel_rep_model)
        self.hidden_size = self.sentence_encoder.output_size
        # self.fc = nn.Linear(hidden_size, hidden_size)
        self.drop = nn.Dropout()

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, B, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_emb = self.sentence_encoder(
            *support)  # (B * N * K, D), where D is the hidden size
        query_emb = self.sentence_encoder(*query)  # (B * total_Q, D)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, self.hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, self.hidden_size)  # (B, total_Q, D)

        B = support.size(0)  # Batch size

        # Prototypical Networks
        # Ignore NA policy
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = -self.__batch_dist__(support, query)  # (B, total_Q, N)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1],
                           2)  # (B, total_Q, N + 1)
        _, pred = torch.max(logits.view(-1, N+1), 1)
        return logits, pred


class GlobalTransformedProtoNet(framework.FewShotREModel):
    def __init__(self, tokenizer, embedder, rel_rep_model, max_length):
        super(GlobalTransformedProtoNet, self).__init__(None)
        self.tokenizer = tokenizer
        self.rr_model = RRModel(embedder, rel_rep_model)
        self.hidden_size = self.rr_model.output_size
        # self.hidden_size = self.instance_emb_size+self.seg_num_emb
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=FLAGS.n_head)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=FLAGS.layer)
        self.seg_embedding = nn.Embedding(20, self.hidden_size)
        self.sep_emd = nn.Embedding(1, self.hidden_size)

        self.q_seg_idx = torch.LongTensor([19])
        self.seg_idxs = None
        self.drop = nn.Dropout(1-FLAGS.dropout_keep_prob)
        self.layNorm_hidden = nn.LayerNorm(self.hidden_size)
        self.N = 0
        self.K = 0
        # self.layNorm_out = nn.LayerNorm(self.hidden_size)

    def __dist__(self, x, y, dim):
        # return -(x*y).sum(dim)
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def encoder(self, tokens, pos1, pos2, mask):

        hidden = self.rr_model(tokens, pos1, pos2, mask)
        # hidden = self.drop(hidden)
        # hidden = self.FC(hidden)
        # hidden = torch.tanh(hidden)
        return hidden

    def forward(self, support, query, B, N, K, total_Q):
        """
        batch_samples: B*Q*N*K*seq_length
        """
        batch_support = self.encoder(*support)
        batch_query = self.encoder(*query)
        # support = self.layNorm(support)
        # query = self.layNorm(query)
        # support = self.drop(support)
        # query = self.drop(query)

        # (B*N,K,D)
        batch_support = batch_support.view(B*N, K, self.hidden_size)
        sep_emb = self.sep_emd(torch.zeros(
            B*N, 1).long().to(FLAGS.paral_cuda[0]))
        K += 1

        # (B, N, K, D)
        batch_support = torch.cat((sep_emb, batch_support), dim=1).view(
            B, N, K, self.hidden_size)
        batch_support = batch_support.view(B, N*(K), self.hidden_size)
        # (B, tot_Q, D)
        batch_query = batch_query.view(B, total_Q, self.hidden_size)

        if self.seg_idxs is None or K != self.K or N != self.N:
            seg_idxs = torch.arange(N).expand(K, N).t().reshape(-1)
            # seg_idxs = torch.cat(
            #     (seg_idxs, self.q_seg_idx))
            # self.seg_idxs = seg_idxs.expand(
            #     B, N*(K)+1).cuda(FLAGS.paral_cuda[0])
            self.seg_idxs = seg_idxs.expand(
                B, N*K).cuda(FLAGS.paral_cuda[0])
        seg_emb = self.seg_embedding(self.seg_idxs)  # (B, N*K+1, seg_emb_D)
        seg_emb = self.layNorm_hidden(seg_emb)/10

        # pos_emb = self.pos_embedding(self.pos_idxs)

        logits_list = []
        for i in range(total_Q):
            sing_query = batch_query[:, i:i+1, :]
            batch_support = batch_support + seg_emb
            instances = torch.cat(
                (batch_support, sing_query), dim=1)  # (B, N*K+1,D)
            # instances = torch.cat((instances, seg_emb), dim=-1)
            # (B,N*K+2, hidden_size)

            input_emb = instances
            input_emb = self.drop(input_emb)
            hidden = input_emb.transpose_(1, 0)  # (N*K+1, B, hidden_size)
            hidden = self.transformer(input_emb)  # (B,N*K+1, hidden_size)
            hidden = hidden.transpose_(1, 0)  # (B,N*K+1, hidden_size)
            # hidden = self.layNorm_hidden(hidden)
            # hidden =self.drop(hidden)
            support = hidden[:, :N*(K)].reshape(B, N, K, -1)  # (B, N, K, D)
            query = hidden[:, N*(K):N*(K)+1]  # (B, 1, D) the sep tensor
            support = self.layNorm_hidden(support)
            query = self.layNorm_hidden(query)
            support = torch.mean(support, 2)  # (B, N, D)

            support = self.drop(support)
            query = self.drop(query)
            logits = -self.__batch_dist__(support, query)  # (B, 1, N)
            # logits=logits.reshape(B,1,N,K)
            # logits,_=torch.max()

            logits_list.append(logits)
        logits = torch.cat(logits_list, dim=1)  # (B, total_Q, N)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred


class InstanceTransformer(framework.FewShotREModel):
    def __init__(self, tokenizer, embedder, rel_rep_model, max_length):
        super(InstanceTransformer, self).__init__(None)
        self.tokenizer = tokenizer
        self.rr_model = RRModel(embedder, rel_rep_model)
        self.seg_num_emb = FLAGS.seg_emb_size
        self.instance_emb_size = self.rr_model.output_size
        self.hidden_size = self.instance_emb_size+self.seg_num_emb
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size, nhead=4)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=FLAGS.layer)
        self.seg_embedding = nn.Embedding(20, self.seg_num_emb)

        self.q_seg_idx = torch.LongTensor([19])
        self.seg_idxs = None
        self.drop = nn.Dropout(1-FLAGS.dropout_keep_prob)
        self.layNorm = nn.LayerNorm(self.instance_emb_size)
        self.layNorm_hidden = nn.LayerNorm(self.hidden_size)
        self.seg_norm = nn.LayerNorm(self.seg_num_emb)
        self.N = 0
        self.K = 0
        # self.layNorm_out = nn.LayerNorm(self.hidden_size)

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def encoder(self, tokens, pos1, pos2, mask):

        hidden = self.rr_model(tokens, pos1, pos2, mask)
        # hidden = self.drop(hidden)
        # hidden = self.FC(hidden)
        # hidden = torch.tanh(hidden)
        return hidden

    def forward(self, support, query, B, N, K, total_Q):
        """
        batch_samples: B*Q*N*K*seq_length
        """
        support = self.encoder(*support)
        query = self.encoder(*query)
        # support = self.layNorm(support)
        # query = self.layNorm(query)
        support = self.drop(support)
        query = self.drop(query)
        # (B, N, K, D)
        batch_support = support.view(B, N, K, self.instance_emb_size)
        batch_support = batch_support.view(B, N*(K), self.instance_emb_size)
        # (B, tot_Q, D)
        batch_query = query.view(B, total_Q, self.instance_emb_size)

        if self.seg_idxs is None or K != self.K or N != self.N:
            seg_idxs = torch.arange(N).expand(K, N).t().reshape(-1)
            seg_idxs = torch.cat(
                (seg_idxs, self.q_seg_idx))
            self.seg_idxs = seg_idxs.expand(
                B, N*(K)+1).cuda(FLAGS.paral_cuda[0])
        seg_emb = self.seg_embedding(self.seg_idxs)  # (B, N*K+1, seg_emb_D)
        seg_emb = self.seg_norm(seg_emb)
        # pos_emb = self.pos_embedding(self.pos_idxs)
        logits_list = []
        for i in range(total_Q):
            sing_query = batch_query[:, i:i+1, :]
            instances = torch.cat(
                (batch_support, sing_query), dim=1)  # (B, N*K+1,D)
            instances = torch.cat((instances, seg_emb), dim=-1)
            # (B,N*K+2, hidden_size)

            input_emb = instances

            hidden = input_emb.transpose_(1, 0)  # (N*K+1, B, hidden_size)
            hidden = self.transformer(input_emb)  # (B,N*K+1, hidden_size)
            hidden = hidden.transpose_(1, 0)  # (B,N*K+1, hidden_size)
            support = hidden[:, :N*(K)].reshape(B, N, K, -1)  # (B, N, K, D)
            query = hidden[:, -1:]  # (B, 1, D) the sep tensor

            support = torch.mean(support, 2)  # (B, N, D)

            support = self.drop(support)
            query = self.drop(query)
            logits = -self.__batch_dist__(support, query)  # (B, 1, N)
            # logits=logits.reshape(B,1,N,K)
            # logits,_=torch.max()

            logits_list.append(logits)
        logits = torch.cat(logits_list, dim=1)  # (B, total_Q, N)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred
