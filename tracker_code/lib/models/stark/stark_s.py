# ############### step1

# """
# Basic STARK Model (Spatial-only).
# """
# import math

# import torch
# from torch import nn
# import numpy as np
# from lib.utils.misc import NestedTensor

# from .backbone import build_backbone
# from .transformer import build_transformer
# from .head import build_box_head
# import torch.nn.functional as F
# from lib.utils.box_ops import box_xyxy_to_cxcywh


# class STARKS(nn.Module):
#     """ This is the base class for Transformer Tracking """
#     # def __init__(self, backbone, transformer, box_head, dynamic, num_queries,
#     #              aux_loss=False, head_type="CORNER"):
#     def __init__(self, backbone1, backbone2, transformer1, transformer2, box_head1, box_head2, training, dynamic, num_queries,
#                  aux_loss=False, head_type="CORNER"):                                                   #20231028
#         """ Initializes the model.
#         Parameters:
#             backbone: torch module of the backbone to be used. See backbone.py
#             transformer: torch module of the transformer architecture. See transformer.py
#             num_queries: number of object queries.
#             aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
#         """
#         super().__init__()
#         # self.backbone = backbone
#         self.backbone1 = backbone1
#         self.backbone2 = backbone2
#         self.transformer1 = transformer1
#         self.transformer2 = transformer2
#         self.box_head1 = box_head1
#         self.box_head2 = box_head2     # 20231028
#         self.num_queries = num_queries
#         self.training = training          # 20231105
#         self.dynamic = dynamic             # added 20231028
#         hidden_dim1 = transformer1.d_model
#         hidden_dim2 = transformer2.d_model           # added 20231028
#         # self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
#         self.query_embed1 = nn.Embedding(num_queries, hidden_dim1)  # object queries
#         self.query_embed2 = nn.Embedding(num_queries, hidden_dim2)  # object queries        # added 20231028
#         # self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
#         self.bottleneck1 = nn.Conv2d(backbone1.num_channels, hidden_dim1, kernel_size=1)  # the bottleneck layer
#         self.bottleneck2 = nn.Conv2d(backbone2.num_channels, hidden_dim2, kernel_size=1)  # the bottleneck layer   # added 20231028
#         self.aux_loss = aux_loss
#         self.head_type = head_type
#         if head_type == "CORNER":
#             self.feat_sz_s1 = int(box_head1.feat_sz)
#             self.feat_sz_s2 = int(box_head2.feat_sz)    # added 20231028
#             self.feat_len_s1 = int(box_head1.feat_sz ** 2)
#             self.feat_len_s2 = int(box_head2.feat_sz ** 2)        # added 20231028
#         if self.dynamic:            # added 20231028
#             router_channels = 256   # resnet18
#             reduction = 4
#             self.router = AdaptiveRouter(router_channels, 1, reduction=reduction)
#             self.dy_thres = 0.5

#     def forward(self, img=None, seq_dict=None, feat_dict=None, feat_dict_list=None, mode="backbone", run_box_head=True, run_cls_head=False):
#         if mode == "backbone":
#             output_dict1 = self.forward_backbone1(img)
#             if self.dynamic:                               # the following are all added 20231105
#                 score = self.router(output_dict1['feat'])    # score denotes the (1-difficulty score)
#             need_second = self.training or (not self.dynamic) or score[:, 0] < self.dy_thres
#             # need_first_head = self.training or (self.dynamic and score[:, 0] >= self.dy_thres)
#             if need_second:
#                 output_dict2 = self.forward_backbone2(img)
#                 return output_dict1, output_dict2
#             else:
#                 return output_dict1
#             # return self.forward_backbone(img)
#             # return output_dict1, output_dict2
#         elif mode == "transformer":
#             return self.forward_transformer(seq_dict, feat_dict, feat_dict_list, run_box_head=run_box_head, run_cls_head=run_cls_head)
#         else:
#             raise ValueError

#     def forward_backbone1(self, input: NestedTensor):
#         """The input type is NestedTensor, which consists of:
#                - tensor: batched images, of shape [batch_size x 3 x H x W]
#                - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         assert isinstance(input, NestedTensor)
#         # Forward the backbone
#         output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
#         # output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search   #20231028
#         output_dic1 = self.adjust1(output_back1, pos1)
#         # output_dic2 = self.adjust2(output_back2, pos2)
#         # print(output_dic['feat'].shape)
#         # print('********************************')
#         # print(output_dic['mask'].shape)
#         # print('---------------------------------')
#         # print(output_dic['pos'].shape)
#         # print('######################################')
#         # output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
#         # output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search
#         # if self.dynamic:       # added 20231028
#         #     score = self.router(output_dic['feat'])       # what is the meaning of "output_back1"? added 20231028

#         # Adjust the shapes
#         return output_dic1

#     def forward_backbone2(self, input: NestedTensor):
#         """The input type is NestedTensor, which consists of:
#                - tensor: batched images, of shape [batch_size x 3 x H x W]
#                - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
#         """
#         assert isinstance(input, NestedTensor)
#         # Forward the backbone
#         # output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
#         output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search   #20231028
#         # output_dic1 = self.adjust1(output_back1, pos1)
#         output_dic2 = self.adjust2(output_back2, pos2)

#         # Adjust the shapes
#         return output_dic2

#     def forward_transformer(self, seq_dict, feat_dict, feat_dict_list, run_box_head=True, run_cls_head=False):
#         if self.aux_loss:
#             raise ValueError("Deep supervision is not supported.")
#         # Forward the transformer encoder and decoder
#         if len(feat_dict) == 2:
#             seq_dict1 = seq_dict[0]
#             seq_dict2 = seq_dict[1]
#             if self.dynamic:
#                 score = self.router(feat_dict_list[1]['feat'])
#             need_first_head = self.training or (self.dynamic and score[:, 0] >= self.dy_thres)
#             if need_first_head:
#                 output_embed1, enc_mem1 = self.transformer1(seq_dict1["feat"], seq_dict1["mask"], self.query_embed1.weight,
#                                                  seq_dict1["pos"], return_encoder_output=True)
#                 output_embed2, enc_mem2 = self.transformer2(seq_dict2["feat"], seq_dict2["mask"], self.query_embed2.weight,
#                                                  seq_dict2["pos"], return_encoder_output=True)
#                 # Forward the corner head
#                 out1, outputs_coord1 = self.forward_box_head1(output_embed1, enc_mem1)
#                 out2, outputs_coord2 = self.forward_box_head2(output_embed2, enc_mem2)
#                 head_out = [out1, outputs_coord1, output_embed1, out2, outputs_coord2, output_embed2]
#             else:
#                 output_embed2, enc_mem2 = self.transformer2(seq_dict2["feat"], seq_dict2["mask"],
#                                                             self.query_embed2.weight,
#                                                             seq_dict2["pos"], return_encoder_output=True)
#                 out2, outputs_coord2 = self.forward_box_head2(output_embed2, enc_mem2)
#                 head_out = [out2, outputs_coord2, output_embed2]
#         else:
#             seq_dict1 = seq_dict[0]
#             output_embed1, enc_mem1 = self.transformer1(seq_dict1["feat"], seq_dict1["mask"], self.query_embed1.weight,
#                                                         seq_dict1["pos"], return_encoder_output=True)
#             out1, outputs_coord1 = self.forward_box_head1(output_embed1, enc_mem1)
#             head_out = [out1, outputs_coord1, output_embed1]

#         return head_out

#     def forward_box_head1(self, hs, memory):
#         """
#         hs: output embeddings (1, B, N, C)
#         memory: encoder embeddings (HW1+HW2, B, C)"""
#         if self.head_type == "CORNER":
#             # adjust shape
#             enc_opt = memory[-self.feat_len_s1:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
#             dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
#             att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
#             opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
#             bs, Nq, C, HW = opt.size()
#             opt_feat = opt.view(-1, C, self.feat_sz_s1, self.feat_sz_s1)
#             # run the corner head
#             outputs_coord = box_xyxy_to_cxcywh(self.box_head1(opt_feat))
#             outputs_coord_new = outputs_coord.view(bs, Nq, 4)
#             out = {'pred_boxes': outputs_coord_new}
#             return out, outputs_coord_new
#         elif self.head_type == "MLP":
#             # Forward the class and box head
#             outputs_coord = self.box_head1(hs).sigmoid()
#             out = {'pred_boxes': outputs_coord[-1]}
#             if self.aux_loss:
#                 out['aux_outputs'] = self._set_aux_loss(outputs_coord)
#             return out, outputs_coord

#     def forward_box_head2(self, hs, memory):
#         """
#         hs: output embeddings (1, B, N, C)
#         memory: encoder embeddings (HW1+HW2, B, C)"""
#         if self.head_type == "CORNER":
#             # adjust shape
#             enc_opt = memory[-self.feat_len_s2:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
#             dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
#             att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
#             opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
#             bs, Nq, C, HW = opt.size()
#             opt_feat = opt.view(-1, C, self.feat_sz_s2, self.feat_sz_s2)
#             # run the corner head
#             outputs_coord = box_xyxy_to_cxcywh(self.box_head2(opt_feat))
#             outputs_coord_new = outputs_coord.view(bs, Nq, 4)
#             out = {'pred_boxes': outputs_coord_new}
#             return out, outputs_coord_new
#         elif self.head_type == "MLP":
#             # Forward the class and box head
#             outputs_coord = self.box_head2(hs).sigmoid()
#             out = {'pred_boxes': outputs_coord[-1]}
#             if self.aux_loss:
#                 out['aux_outputs'] = self._set_aux_loss(outputs_coord)
#             return out, outputs_coord
#     def adjust1(self, output_back: list, pos_embed: list):
#         """
#         """
#         src_feat, mask = output_back[-1].decompose()
#         assert mask is not None
#         # reduce channel
#         feat = self.bottleneck1(src_feat)  # (B, C, H, W)
#         # adjust shapes
#         feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
#         pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
#         mask_vec = mask.flatten(1)  # BxHW
#         return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

#     def adjust2(self, output_back: list, pos_embed: list):
#         """
#         """
#         src_feat, mask = output_back[-1].decompose()
#         assert mask is not None
#         # reduce channel
#         feat = self.bottleneck2(src_feat)  # (B, C, H, W)
#         # adjust shapes
#         feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
#         pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
#         mask_vec = mask.flatten(1)  # BxHW
#         return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

#     @torch.jit.unused
#     def _set_aux_loss(self, outputs_coord):
#         # this is a workaround to make torchscript happy, as torchscript
#         # doesn't support dictionary with non-homogeneous values, such
#         # as a dict having both a Tensor and a list.
#         return [{'pred_boxes': b}
#                 for b in outputs_coord[:-1]]


# def build_starks(cfg):
#     backbone1, backbone2 = build_backbone(cfg)  # backbone and positional encoding are built together  #20231028
#     transformer1 = build_transformer(cfg)
#     transformer2 = build_transformer(cfg)
#     box_head1 = build_box_head(cfg)
#     box_head2 = build_box_head(cfg)
#     model = STARKS(
#         backbone1,
#         backbone2,
#         transformer1,
#         transformer2,
#         box_head1,
#         box_head2,
#         training=cfg.TRAIN.Training,
#         dynamic=cfg.MODEL.BACKBONE.Dynamic,
#         num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
#         aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
#         head_type=cfg.MODEL.HEAD_TYPE
#     )

#     return model


# class AdaptiveRouter(nn.Module):
#     def __init__(self, features_channels, out_channels, reduction=4):
#         super(AdaptiveRouter, self).__init__()
#         self.inp = features_channels
#         self.oup = out_channels
#         self.reduction = reduction
#         self.layer1 = nn.Conv2d(self.inp, self.inp//self.reduction, kernel_size=1, bias=True)
#         self.layer2 = nn.Conv2d(self.inp//self.reduction, self.oup, kernel_size=1, bias=True)

#     def forward(self, xs, thres=0.5):
#         # print(xs.shape)
#         size, bs, num = xs.shape
#         xs = xs.contiguous().view(bs, num, -1)
#         # xs = [x.mean(dim=(1, 2), keepdim=True) for x in xs]
#         # print(xs.shape)
#         xs = xs.mean(dim=2, keepdim=True)
#         # print('############################')
#         # print(xs.shape)
#         xs = torch.unsqueeze(xs, 3)
#         # xs = torch.cat(xs, dim=1)
#         xs = self.layer1(xs)
#         xs = F.relu(xs, inplace=True)
#         xs = self.layer2(xs).flatten(1)
#         if self.training:
#             xs = sigmoid(xs, hard=False, threshold=thres)
#         else:
#             xs = xs.sigmoid()
#         # print(xs)
#         return xs


# def sigmoid(logits, hard=False, threshold=0.5):
#     y_soft = logits.sigmoid()
#     if hard:
#         indices = (y_soft < threshold).nonzero(as_tuple=True)
#         y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
#         y_hard[indices[0], indices[1]] = 1.0
#         ret = y_hard - y_soft.detach() + y_soft
#     else:
#         ret = y_soft
#     return ret


#############step2

"""
Basic STARK Model (Spatial-only).
"""
import math

import torch
from torch import nn
import numpy as np
from lib.utils.misc import NestedTensor

from .backbone import build_backbone
from .transformer import build_transformer
from .head import build_box_head
import torch.nn.functional as F
from lib.utils.box_ops import box_xyxy_to_cxcywh


class STARKS(nn.Module):
    """ This is the base class for Transformer Tracking """
    # def __init__(self, backbone, transformer, box_head, dynamic, num_queries,
    #              aux_loss=False, head_type="CORNER"):
    def __init__(self, backbone1, backbone2, transformer1, transformer2, box_head1, box_head2, training, dynamic, num_queries,
                 aux_loss=False, head_type="CORNER"):                                                   #20231028
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_queries: number of object queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        # self.backbone = backbone
        self.backbone1 = backbone1
        self.backbone2 = backbone2
        self.transformer1 = transformer1
        self.transformer2 = transformer2
        self.box_head1 = box_head1
        self.box_head2 = box_head2     # 20231028
        self.num_queries = num_queries
        self.training = training          # 20231105
        self.dynamic = dynamic             # added 20231028
        hidden_dim1 = transformer1.d_model
        hidden_dim2 = transformer2.d_model           # added 20231028
        # self.query_embed = nn.Embedding(num_queries, hidden_dim)  # object queries
        self.query_embed1 = nn.Embedding(num_queries, hidden_dim1)  # object queries
        self.query_embed2 = nn.Embedding(num_queries, hidden_dim2)  # object queries        # added 20231028
        # self.bottleneck = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)  # the bottleneck layer
        self.bottleneck1 = nn.Conv2d(backbone1.num_channels, hidden_dim1, kernel_size=1)  # the bottleneck layer
        self.bottleneck2 = nn.Conv2d(backbone2.num_channels, hidden_dim2, kernel_size=1)  # the bottleneck layer   # added 20231028
        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER":
            self.feat_sz_s1 = int(box_head1.feat_sz)
            self.feat_sz_s2 = int(box_head2.feat_sz)    # added 20231028
            self.feat_len_s1 = int(box_head1.feat_sz ** 2)
            self.feat_len_s2 = int(box_head2.feat_sz ** 2)        # added 20231028
        if self.dynamic:            # added 20231028
            router_channels = 256   # resnet18
            reduction = 4
            self.router = AdaptiveRouter(router_channels, 1, reduction=reduction)
            self.dy_thres = 0.5

    def forward(self, img=None, seq_dict=None, feat_dict=None, feat_dict_list=None, mode="backbone", run_box_head=True, run_cls_head=False):
        if mode == "backbone":
            output_dict1 = self.forward_backbone1(img)
            if self.dynamic:                               # the following are all added 20231105
                output_dict1_feat = output_dict1['feat'].requires_grad_(True)
                score = self.router(output_dict1_feat)    # score denotes the (1-difficulty score)
            need_second = self.training or (not self.dynamic) or score[:, 0] < self.dy_thres
            # need_first_head = self.training or (self.dynamic and score[:, 0] >= self.dy_thres)
            if need_second:
                output_dict2 = self.forward_backbone2(img)
                return output_dict1, output_dict2
            else:
                return output_dict1
            # return self.forward_backbone(img)
            # return output_dict1, output_dict2
        elif mode == "transformer":
            return self.forward_transformer(seq_dict, feat_dict, feat_dict_list, run_box_head=run_box_head, run_cls_head=run_cls_head)
        else:
            raise ValueError

    def forward_backbone1(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
        # output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search   #20231028
        output_dic1 = self.adjust1(output_back1, pos1)
        # output_dic2 = self.adjust2(output_back2, pos2)
        # print(output_dic['feat'].shape)
        # print('********************************')
        # print(output_dic['mask'].shape)
        # print('---------------------------------')
        # print(output_dic['pos'].shape)
        # print('######################################')
        # output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
        # output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search
        # if self.dynamic:       # added 20231028
        #     score = self.router(output_dic['feat'])       # what is the meaning of "output_back1"? added 20231028

        # Adjust the shapes
        return output_dic1

    def forward_backbone2(self, input: NestedTensor):
        """The input type is NestedTensor, which consists of:
               - tensor: batched images, of shape [batch_size x 3 x H x W]
               - mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        assert isinstance(input, NestedTensor)
        # Forward the backbone
        # output_back1, pos1 = self.backbone1(input)  # features & masks, position embedding for the search
        output_back2, pos2 = self.backbone2(input)  # features & masks, position embedding for the search   #20231028
        # output_dic1 = self.adjust1(output_back1, pos1)
        output_dic2 = self.adjust2(output_back2, pos2)

        # Adjust the shapes
        return output_dic2

    #######forward_transformer---20240513---for testing
    def forward_transformer(self, seq_dict, feat_dict_list, need_second, run_box_head=True, run_cls_head=False):
        if self.aux_loss:
            raise ValueError("Deep supervision is not supported.")
        # Forward the transformer encoder and decoder
        # if len(feat_dict) == 2:
        #     seq_dict1 = seq_dict[0]
        #     seq_dict2 = seq_dict[1]
        # if self.dynamic:
        #     # score = self.router(feat_dict_list[1]['feat'].requires_grad_(True))
        #     score = self.router(feat_dict_list[1]['feat'])
        # need_second = self.training or (not self.dynamic) or score[:, 0] < self.dy_thres
        if need_second:
            output_embed, enc_mem = self.transformer2(seq_dict["feat"], seq_dict["mask"], self.query_embed2.weight,
                                                        seq_dict["pos"], return_encoder_output=True)
            out, outputs_coord = self.forward_box_head2(output_embed, enc_mem)
        else:
            output_embed, enc_mem = self.transformer1(seq_dict["feat"], seq_dict["mask"], self.query_embed1.weight,
                                                        seq_dict["pos"], return_encoder_output=True)
            out, outputs_coord = self.forward_box_head1(output_embed, enc_mem)
        return out, outputs_coord, output_embed

#     #####forward_transformer---20240513---for training
#     def forward_transformer(self, seq_dict, feat_dict, feat_dict_list, run_box_head=True, run_cls_head=False):
#         if self.aux_loss:
#             raise ValueError("Deep supervision is not supported.")
#         # Forward the transformer encoder and decoder
#         if len(feat_dict) == 2:
#             seq_dict1 = seq_dict[0]
#             seq_dict2 = seq_dict[1]
#             if self.dynamic:
#                 score = self.router(feat_dict_list[1]['feat'].requires_grad_(True))
#             need_first_head = self.training or (self.dynamic and score[:, 0] >= self.dy_thres)
#             if need_first_head:
#                 output_embed1, enc_mem1 = self.transformer1(seq_dict1["feat"], seq_dict1["mask"], self.query_embed1.weight,
#                                                  seq_dict1["pos"], return_encoder_output=True)
#                 output_embed2, enc_mem2 = self.transformer2(seq_dict2["feat"], seq_dict2["mask"], self.query_embed2.weight,
#                                                  seq_dict2["pos"], return_encoder_output=True)
#                 # Forward the corner head
#                 out1, outputs_coord1 = self.forward_box_head1(output_embed1, enc_mem1)
#                 out2, outputs_coord2 = self.forward_box_head2(output_embed2, enc_mem2)
#                 head_out = [out1, outputs_coord1, output_embed1, out2, outputs_coord2, output_embed2, score]
#             else:
#                 output_embed2, enc_mem2 = self.transformer2(seq_dict2["feat"], seq_dict2["mask"],
#                                                             self.query_embed2.weight,
#                                                             seq_dict2["pos"], return_encoder_output=True)
#                 out2, outputs_coord2 = self.forward_box_head2(output_embed2, enc_mem2)
#                 head_out = [out2, outputs_coord2, output_embed2]
#         else:
#             seq_dict1 = seq_dict[0]
#             output_embed1, enc_mem1 = self.transformer1(seq_dict1["feat"], seq_dict1["mask"], self.query_embed1.weight,
#                                                         seq_dict1["pos"], return_encoder_output=True)
#             out1, outputs_coord1 = self.forward_box_head1(output_embed1, enc_mem1)
#             head_out = [out1, outputs_coord1, output_embed1]
    
#         return head_out

    def forward_box_head1(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s1:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s1, self.feat_sz_s1)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head1(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head1(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord

    def forward_box_head2(self, hs, memory):
        """
        hs: output embeddings (1, B, N, C)
        memory: encoder embeddings (HW1+HW2, B, C)"""
        if self.head_type == "CORNER":
            # adjust shape
            enc_opt = memory[-self.feat_len_s2:].transpose(0, 1)  # encoder output for the search region (B, HW, C)
            dec_opt = hs.squeeze(0).transpose(1, 2)  # (B, C, N)
            att = torch.matmul(enc_opt, dec_opt)  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute((0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)
            bs, Nq, C, HW = opt.size()
            opt_feat = opt.view(-1, C, self.feat_sz_s2, self.feat_sz_s2)
            # run the corner head
            outputs_coord = box_xyxy_to_cxcywh(self.box_head2(opt_feat))
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new}
            return out, outputs_coord_new
        elif self.head_type == "MLP":
            # Forward the class and box head
            outputs_coord = self.box_head2(hs).sigmoid()
            out = {'pred_boxes': outputs_coord[-1]}
            if self.aux_loss:
                out['aux_outputs'] = self._set_aux_loss(outputs_coord)
            return out, outputs_coord
    def adjust1(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck1(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    def adjust2(self, output_back: list, pos_embed: list):
        """
        """
        src_feat, mask = output_back[-1].decompose()
        assert mask is not None
        # reduce channel
        feat = self.bottleneck2(src_feat)  # (B, C, H, W)
        # adjust shapes
        feat_vec = feat.flatten(2).permute(2, 0, 1)  # HWxBxC
        pos_embed_vec = pos_embed[-1].flatten(2).permute(2, 0, 1)  # HWxBxC
        mask_vec = mask.flatten(1)  # BxHW
        return {"feat": feat_vec, "mask": mask_vec, "pos": pos_embed_vec}

    @torch.jit.unused
    def _set_aux_loss(self, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_boxes': b}
                for b in outputs_coord[:-1]]


def build_starks(cfg):
    backbone1, backbone2 = build_backbone(cfg)  # backbone and positional encoding are built together  #20231028
    transformer1 = build_transformer(cfg)
    transformer2 = build_transformer(cfg)
    box_head1 = build_box_head(cfg)
    box_head2 = build_box_head(cfg)
    model = STARKS(
        backbone1,
        backbone2,
        transformer1,
        transformer2,
        box_head1,
        box_head2,
        training=cfg.TRAIN.Training,
        dynamic=cfg.MODEL.BACKBONE.Dynamic,
        num_queries=cfg.MODEL.NUM_OBJECT_QUERIES,
        aux_loss=cfg.TRAIN.DEEP_SUPERVISION,
        head_type=cfg.MODEL.HEAD_TYPE
    )

    return model


class AdaptiveRouter(nn.Module):
    def __init__(self, features_channels, out_channels, reduction=4):
        super(AdaptiveRouter, self).__init__()
        self.inp = features_channels
        self.oup = out_channels
        self.reduction = reduction
        self.layer1 = nn.Conv2d(self.inp, self.inp//self.reduction, kernel_size=1, bias=True)
        self.layer2 = nn.Conv2d(self.inp//self.reduction, self.oup, kernel_size=1, bias=True)

    def forward(self, xs, thres=0.5):
        # print(xs.shape)
        size, bs, num = xs.shape
        # print()
        xs = xs.contiguous().view(bs, num, -1)
        # xs = [x.mean(dim=(1, 2), keepdim=True) for x in xs]
        # print(xs.shape)
        xs = xs.mean(dim=2, keepdim=True)
        # print('############################')
        # print(xs.shape)
        xs = torch.unsqueeze(xs, 3)
        # xs = torch.cat(xs, dim=1)
        xs = self.layer1(xs)
        xs = F.relu(xs, inplace=True)
        xs = self.layer2(xs).flatten(1)
        if self.training:
            xs = sigmoid(xs, hard=False, threshold=thres)
        else:
            xs = xs.sigmoid()
        # print(xs)
        return xs


def sigmoid(logits, hard=False, threshold=0.5):
    y_soft = logits.sigmoid()
    if hard:
        indices = (y_soft < threshold).nonzero(as_tuple=True)
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format)
        y_hard[indices[0], indices[1]] = 1.0
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret