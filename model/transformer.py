from torch.nn.modules import transformer
from torch.nn.modules.activation import MultiheadAttention
from torch import Tensor
from typing import Optional
import torch
import torch.nn as nn


# class GlobalTransformerEncoderLayer(transformer.TransformerEncoderLayer):
#     def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
#         super(GlobalTransformerEncoderLayer, self).__init__(
#             d_model, nhead, dim_feedforward, dropout, activation)
#         self.self_attn_global = MultiheadAttention(
#             d_model, nhead, dropout=dropout)
#         self.self_attn_intra = MultiheadAttention(
#             d_model, nhead, dropout=dropout)
#         self.self_attn_inter = MultiheadAttention(
#             d_model, nhead, dropout=dropout)
#         self.self_attn = self.self_attn_global

#     def __generate_intra_and_inter_mask(self, N, K, device):
#         one_vec = torch.ones(K)
#         diag_mat = torch.diag(one_vec)
#         diag_att_mat = ~(diag_mat.bool())
#         all_att_mat = torch.zeros(K, K).bool()
#         all_mask_mat = torch.ones(K, K).bool()

#         intra_mask = []
#         inter_mask = []
#         for i in range(N):
#             intra_line = []
#             inter_line = []
#             for j in range(N):
#                 if i == j:
#                     intra_line.append(all_att_mat)
#                     inter_line.append(diag_att_mat)
#                 else:
#                     intra_line.append(all_mask_mat)
#                     inter_line.append(all_att_mat)
#             intra_mask.append(torch.cat(intra_line, dim=1))
#             inter_mask.append(torch.cat(inter_line, dim=1))

#         intra_mask = torch.cat(intra_mask, dim=0).to(device)
#         inter_mask = torch.cat(inter_mask, dim=0).to(device)

#         return intra_mask, inter_mask

#     def apply_self_attn(self, self_attn, src, src_mask, src_key_padding_mask):
#         src2 = self_attn(src, src, src, attn_mask=src_mask,
#                          key_padding_mask=src_key_padding_mask)
#         att = src2[1].detach().cpu().numpy().tolist()
#         src2 = src2[0]
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         return src

#     # def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#     #     r"""Pass the input through the encoder layer.
#     #     Args:
#     #         src: the sequence to the encoder layer (required).
#     #         src_mask: the mask for the src sequence (optional).
#     #         src_key_padding_mask: the mask for the src keys per batch (optional).
#     #     Shape:
#     #         see the docs in Transformer class.
#     #     """
#     #     B, N, K, hidden_size = src.shape
#     #     src = src.view(B, N*K, hidden_size)
#     #     src = src.transpose_(1, 0)
#     #     intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
#     #         N, K, src.device)

#     #     intra_src = self.apply_self_attn(
#     #         self.self_attn_intra, src, intra_mask, src_key_padding_mask)
#     #     global_src = self.apply_self_attn(
#     #         self.self_attn, src, src_mask, src_key_padding_mask)
#     #     inter_src = self.apply_self_attn(
#     #         self.self_attn_inter, src, inter_mask, src_key_padding_mask)

#     #     src = (intra_src+global_src+inter_src)/3
#     #     src = src.transpose_(1, 0)
#     #     src = src.view(B, N, K, hidden_size)
#     #     return src

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
#         r"""Pass the input through the encoder layer.
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#         Shape:
#             see the docs in Transformer class.
#         """
#         B, N, K, hidden_size = src.shape
#         src = src.view(B, N*K, hidden_size)
#         src = src.transpose_(1, 0)
#         intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
#             N, K, src.device)

#         intra_src = self.self_attn_intra(
#             src, src, src, attn_mask=intra_mask, key_padding_mask=src_key_padding_mask)[0]
#         global_src = self.self_attn_global(
#             src, src, src, key_padding_mask=src_key_padding_mask)[0]

#         inter_src = self.self_attn_inter(
#             src, src, src, attn_mask=inter_mask, key_padding_mask=src_key_padding_mask)[0]

#         src2 = intra_src+global_src+inter_src
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#         src = src.transpose_(1, 0)
#         src = src.view(B, N, K, hidden_size)
#         return src

class GlobalTransformerEncoderLayer(transformer.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(GlobalTransformerEncoderLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_global = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.self_attn_intra = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.self_attn_inter = MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.linear3=nn.Linear(3*d_model,dim_feedforward)
        # self.norm2=nn.LayerNorm(dim_feedforward)
        self.dropout3=nn.Dropout(dropout)
        

    def __generate_intra_and_inter_mask(self, N, K, device):
        one_vec = torch.ones(K)
        diag_mat = torch.diag(one_vec)
        diag_att_mat = ~(diag_mat.bool())
        all_att_mat = torch.zeros(K, K).bool()
        all_mask_mat = torch.ones(K, K).bool()

        intra_mask = []
        inter_mask = []
        for i in range(N):
            intra_line = []
            inter_line = []
            for j in range(N):
                if i == j:
                    intra_line.append(all_att_mat)
                    inter_line.append(diag_att_mat)
                else:
                    intra_line.append(all_mask_mat)
                    inter_line.append(all_att_mat)
            intra_mask.append(torch.cat(intra_line, dim=1))
            inter_mask.append(torch.cat(inter_line, dim=1))

        intra_mask = torch.cat(intra_mask, dim=0).to(device)
        inter_mask = torch.cat(inter_mask, dim=0).to(device)

        return intra_mask, inter_mask

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        B, N, K, hidden_size = src.shape
        src = src.view(B, N*K, hidden_size)
        src = src.transpose_(1, 0)
        intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
            N, K, src.device)

        intra_src = self.self_attn_intra(
            src, src, src, attn_mask=intra_mask, key_padding_mask=src_key_padding_mask)[0]

        global_src = self.self_attn_global(
            src, src, src, key_padding_mask=src_key_padding_mask)[0]

        inter_src = self.self_attn_inter(
            src, src, src, attn_mask=inter_mask, key_padding_mask=src_key_padding_mask)[0]


        src2 = torch.cat([intra_src,global_src,inter_src],dim=-1)

        ##
        src2=self.linear2(self.dropout(self.activation(self.linear3(src2))))
        # src2=self.linear3(self.dropout3(src2))
        ##

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        ##
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        ##

        src = src.transpose_(1, 0)
        src = src.view(B, N, K, hidden_size)
        return src

class NoIntraLayer(transformer.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(NoIntraLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_global = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.self_attn_inter = MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.linear3=nn.Linear(2*d_model,dim_feedforward)
        # self.norm2=nn.LayerNorm(dim_feedforward)
        self.dropout3=nn.Dropout(dropout)
        

    def __generate_intra_and_inter_mask(self, N, K, device):
        one_vec = torch.ones(K)
        diag_mat = torch.diag(one_vec)
        diag_att_mat = ~(diag_mat.bool())
        all_att_mat = torch.zeros(K, K).bool()
        all_mask_mat = torch.ones(K, K).bool()

        intra_mask = []
        inter_mask = []
        for i in range(N):
            intra_line = []
            inter_line = []
            for j in range(N):
                if i == j:
                    intra_line.append(all_att_mat)
                    inter_line.append(diag_att_mat)
                else:
                    intra_line.append(all_mask_mat)
                    inter_line.append(all_att_mat)
            intra_mask.append(torch.cat(intra_line, dim=1))
            inter_mask.append(torch.cat(inter_line, dim=1))

        intra_mask = torch.cat(intra_mask, dim=0).to(device)
        inter_mask = torch.cat(inter_mask, dim=0).to(device)

        return intra_mask, inter_mask

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        B, N, K, hidden_size = src.shape
        src = src.view(B, N*K, hidden_size)
        src = src.transpose_(1, 0)
        intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
            N, K, src.device)

        global_src = self.self_attn_global(
            src, src, src, key_padding_mask=src_key_padding_mask)[0]

        inter_src = self.self_attn_inter(
            src, src, src, attn_mask=inter_mask, key_padding_mask=src_key_padding_mask)[0]


        src2 = torch.cat([global_src,inter_src],dim=-1)

        ##
        src2=self.linear2(self.dropout(self.activation(self.linear3(src2))))
        # src2=self.linear3(self.dropout3(src2))
        ##

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        ##
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        ##

        src = src.transpose_(1, 0)
        src = src.view(B, N, K, hidden_size)
        return src

class NoGlobalLayer(transformer.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(NoGlobalLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_intra = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.self_attn_inter = MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.linear3=nn.Linear(2*d_model,dim_feedforward)
        # self.norm2=nn.LayerNorm(dim_feedforward)
        self.dropout3=nn.Dropout(dropout)
        

    def __generate_intra_and_inter_mask(self, N, K, device):
        one_vec = torch.ones(K)
        diag_mat = torch.diag(one_vec)
        diag_att_mat = ~(diag_mat.bool())
        all_att_mat = torch.zeros(K, K).bool()
        all_mask_mat = torch.ones(K, K).bool()

        intra_mask = []
        inter_mask = []
        for i in range(N):
            intra_line = []
            inter_line = []
            for j in range(N):
                if i == j:
                    intra_line.append(all_att_mat)
                    inter_line.append(diag_att_mat)
                else:
                    intra_line.append(all_mask_mat)
                    inter_line.append(all_att_mat)
            intra_mask.append(torch.cat(intra_line, dim=1))
            inter_mask.append(torch.cat(inter_line, dim=1))

        intra_mask = torch.cat(intra_mask, dim=0).to(device)
        inter_mask = torch.cat(inter_mask, dim=0).to(device)

        return intra_mask, inter_mask

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        B, N, K, hidden_size = src.shape
        src = src.view(B, N*K, hidden_size)
        src = src.transpose_(1, 0)
        intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
            N, K, src.device)

        intra_src = self.self_attn_intra(
            src, src, src, attn_mask=intra_mask, key_padding_mask=src_key_padding_mask)[0]


        inter_src = self.self_attn_inter(
            src, src, src, attn_mask=inter_mask, key_padding_mask=src_key_padding_mask)[0]


        src2 = torch.cat([intra_src,inter_src],dim=-1)

        ##
        src2=self.linear2(self.dropout(self.activation(self.linear3(src2))))
        # src2=self.linear3(self.dropout3(src2))
        ##

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        ##
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        ##

        src = src.transpose_(1, 0)
        src = src.view(B, N, K, hidden_size)
        return src


class NoInterLayer(transformer.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(NoInterLayer, self).__init__(
            d_model, nhead, dim_feedforward, dropout, activation)
        self.self_attn_global = MultiheadAttention(
            d_model, nhead, dropout=dropout)
        self.self_attn_intra = MultiheadAttention(
            d_model, nhead, dropout=dropout)

        self.linear3=nn.Linear(2*d_model,dim_feedforward)
        # self.norm2=nn.LayerNorm(dim_feedforward)
        self.dropout3=nn.Dropout(dropout)
        

    def __generate_intra_and_inter_mask(self, N, K, device):
        one_vec = torch.ones(K)
        diag_mat = torch.diag(one_vec)
        diag_att_mat = ~(diag_mat.bool())
        all_att_mat = torch.zeros(K, K).bool()
        all_mask_mat = torch.ones(K, K).bool()

        intra_mask = []
        inter_mask = []
        for i in range(N):
            intra_line = []
            inter_line = []
            for j in range(N):
                if i == j:
                    intra_line.append(all_att_mat)
                    inter_line.append(diag_att_mat)
                else:
                    intra_line.append(all_mask_mat)
                    inter_line.append(all_att_mat)
            intra_mask.append(torch.cat(intra_line, dim=1))
            inter_mask.append(torch.cat(inter_line, dim=1))

        intra_mask = torch.cat(intra_mask, dim=0).to(device)
        inter_mask = torch.cat(inter_mask, dim=0).to(device)

        return intra_mask, inter_mask

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        B, N, K, hidden_size = src.shape
        src = src.view(B, N*K, hidden_size)
        src = src.transpose_(1, 0)
        intra_mask, inter_mask = self.__generate_intra_and_inter_mask(
            N, K, src.device)

        intra_src = self.self_attn_intra(
            src, src, src, attn_mask=intra_mask, key_padding_mask=src_key_padding_mask)[0]

        global_src = self.self_attn_global(
            src, src, src, key_padding_mask=src_key_padding_mask)[0]



        src2 = torch.cat([intra_src,global_src],dim=-1)

        ##
        src2=self.linear2(self.dropout(self.activation(self.linear3(src2))))
        # src2=self.linear3(self.dropout3(src2))
        ##

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        ##
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        ##

        src = src.transpose_(1, 0)
        src = src.view(B, N, K, hidden_size)
        return src
