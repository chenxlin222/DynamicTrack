#####################step1
from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search


class STARKSActor(BaseActor):
    """ Actor for training the STARK-S and STARK-ST(Stage1)"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)       # 20231028

        # process the groundtruth
        gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

        # compute losses
        if len(out_dict) == 6:
            out_dict1 = out_dict[0]
            out_dict2 = out_dict[3]
            loss1, status1 = self.compute_losses(out_dict1, gt_bboxes[0])
            loss2, status2 = self.compute_losses(out_dict2, gt_bboxes[0])    # 20231028

            loss = loss1 + loss2

            status = {"Loss/total": loss.item(),           # 20231028
                  "Loss/loss1": loss1.item(),
                  "Loss/loss2": loss2.item(),
                  "Loss/giou1": status1["Loss/giou"],
                  "Loss/giou2": status2["Loss/giou"],
                  "Loss/l1-1": status1["Loss/l1"],
                  "Loss/l1-2": status2["Loss/l1"],
                  "IoU1": status1["IoU"],
                  "IoU2": status2["IoU"]}
        else:
            out_dict1 = out_dict[0]
            loss, status = self.compute_losses(out_dict1, gt_bboxes[0])

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head):
        feat_dict_list1 = []
        feat_dict_list2 = []
        # process the templates
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            # print(self.settings)
            feat_dict_list_temp_out = self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone')     # 20231105
            if len(feat_dict_list_temp_out) == 2:                     # 20231105
                feat_dict_list1.append(feat_dict_list_temp_out[0])
                feat_dict_list2.append(feat_dict_list_temp_out[0])    # 20231028  template only use the backbone1 --20240421
            else:
                feat_dict_list1.append(feat_dict_list_temp_out)        # 20231105

        # process the search regions (t-th frame)
        search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
        feat_dict_list_sear_out = self.net(img=NestedTensor(search_img, search_att), mode='backbone')    # 20231105
        # Judge the length of the backbone output, and merge template and search.    -----20231105
        if len(feat_dict_list_sear_out) == 2:                         # 20231105
            feat_dict_list1.append(feat_dict_list_sear_out[0])       # 20231105
            feat_dict_list2.append(feat_dict_list_sear_out[1])     # 20231028
            seq_dict1 = merge_template_search(feat_dict_list1)        # 20231105
            seq_dict2 = merge_template_search(feat_dict_list2)           # 20231105
            seq_dict = [seq_dict1, seq_dict2]                            # 20231105
            # seq_dict_ = seq_dict[0]
            # print(seq_dict_)
            # seq_dict__ = seq_dict[1]
            # print(seq_dict__)
        else:
            feat_dict_list1.append(feat_dict_list_sear_out)
            seq_dict1 = merge_template_search(feat_dict_list1)
            seq_dict = [seq_dict1]

        # run the transformer and compute losses
        out_dict = self.net(seq_dict=seq_dict, feat_dict=feat_dict_list_sear_out, feat_dict_list=feat_dict_list1,
                                                    mode="transformer", run_box_head=run_box_head,
                                                    run_cls_head=run_cls_head)
        # if len(out_dict) == 6:
        #     return out_dict[0], out_dict[3]
        # else:
        #     return out_dict[0]
        # out_dict1, _, _, out_dict2, _, _ = self.net(seq_dict=seq_dict, feat_dict=feat_dict_list_sear_out,
        #                                             mode="transformer", run_box_head=run_box_head,
        #                                             run_cls_head=run_cls_head)                    # 20231028
        # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
        # return out_dict1, out_dict2
        return out_dict

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss


#########################step2

# from . import BaseActor
# from lib.utils.misc import NestedTensor
# from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
# import torch
# from lib.utils.merge import merge_template_search
# import numpy as np


# class STARKSActor(BaseActor):
#     """ Actor for training the STARK-S and STARK-ST(Stage1)"""
#     def __init__(self, net, objective, loss_weight, settings):
#         super().__init__(net, objective)
#         self.loss_weight = loss_weight
#         self.settings = settings
#         self.bs = self.settings.batchsize  # batch size
#         self.tracked_diff = 0         # 20231116
#         self.iter_count = 0           # 20231116
#         self.diff_list = []           # 20231116

#     def __call__(self, data):
#         """
#         args:
#             data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
#             template_images: (N_t, batch, 3, H, W)
#             search_images: (N_s, batch, 3, H, W)
#         returns:
#             loss    - the training loss
#             status  -  dict containing detailed losses
#         """
#         # forward pass
#         out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)       # 20231028

#         # process the groundtruth
#         gt_bboxes = data['search_anno']  # (Ns, batch, 4) (x1,y1,w,h)

#         # compute losses
#         if len(out_dict) == 7:        # 20231116
#             out_dict1 = out_dict[0]
#             out_dict2 = out_dict[3]
#             loss1, status1 = self.compute_losses(out_dict1, gt_bboxes[0])
#             loss2, status2 = self.compute_losses(out_dict2, gt_bboxes[0])    # 20231028
#             self.iter_count += 1            # 20231116
#             current_diff = loss1 - loss2    # 20231116
#             self.diff_list.append(current_diff.item())     # 20231116

#             # (online) calculation of the loss difference   --20231116
#             if self.iter_count < 10000:                           # 20231116
#                 self.tracked_diff = np.median(self.diff_list)    # 20231116
#             elif self.iter_count % 1000 == 0:                   # 20231116
#                 self.tracked_diff = np.median(self.diff_list)   # 20231116

#             loss1 = loss1 - self.tracked_diff / 2    # 20231116
#             loss2 = loss2 + self.tracked_diff / 2   # 20231116

#             # loss = loss1 + loss2
#             loss = out_dict[6][:, 0] * loss1 + (1-out_dict[6][:, 0]) * loss2                     # 20231116

#             status = {"Loss/total": loss.item(),           # 20231028
#                   "Loss/loss1": loss1.item(),
#                   "Loss/loss2": loss2.item(),
#                   "Loss/giou1": status1["Loss/giou"],
#                   "Loss/giou2": status2["Loss/giou"],
#                   "Loss/l1-1": status1["Loss/l1"],
#                   "Loss/l1-2": status2["Loss/l1"],
#                   "IoU1": status1["IoU"],
#                   "IoU2": status2["IoU"]}
#         else:
#             out_dict1 = out_dict[0]
#             loss, status = self.compute_losses(out_dict1, gt_bboxes[0])

#         return loss, status

#     def forward_pass(self, data, run_box_head, run_cls_head):
#         feat_dict_list1 = []
#         feat_dict_list2 = []
#         # process the templates
#         for i in range(self.settings.num_template):
#             template_img_i = data['template_images'][i].view(-1, *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
#             template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
#             # print(self.settings)
#             feat_dict_list_temp_out = self.net(img=NestedTensor(template_img_i, template_att_i), mode='backbone')     # 20231105
#             if len(feat_dict_list_temp_out) == 2:                     # 20231105
#                 feat_dict_list1.append(feat_dict_list_temp_out[0])
#                 feat_dict_list2.append(feat_dict_list_temp_out[1])    # 20231028
#             else:
#                 feat_dict_list1.append(feat_dict_list_temp_out)        # 20231105

#         # process the search regions (t-th frame)
#         search_img = data['search_images'].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
#         search_att = data['search_att'].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
#         feat_dict_list_sear_out = self.net(img=NestedTensor(search_img, search_att), mode='backbone')    # 20231105
#         # Judge the length of the backbone output, and merge template and search.    -----20231105
#         if len(feat_dict_list_sear_out) == 2:                         # 20231105
#             feat_dict_list1.append(feat_dict_list_sear_out[0])       # 20231105
#             feat_dict_list2.append(feat_dict_list_sear_out[1])     # 20231028
#             seq_dict1 = merge_template_search(feat_dict_list1)        # 20231105
#             seq_dict2 = merge_template_search(feat_dict_list2)           # 20231105
#             seq_dict = [seq_dict1, seq_dict2]                            # 20231105
#             # seq_dict_ = seq_dict[0]
#             # print(seq_dict_)
#             # seq_dict__ = seq_dict[1]
#             # print(seq_dict__)
#         else:
#             feat_dict_list1.append(feat_dict_list_sear_out)
#             seq_dict1 = merge_template_search(feat_dict_list1)
#             seq_dict = [seq_dict1]

#         # run the transformer and compute losses
#         out_dict = self.net(seq_dict=seq_dict, feat_dict=feat_dict_list_sear_out, feat_dict_list=feat_dict_list1,
#                                                     mode="transformer", run_box_head=run_box_head,
#                                                     run_cls_head=run_cls_head)
#         # if len(out_dict) == 6:
#         #     return out_dict[0], out_dict[3]
#         # else:
#         #     return out_dict[0]
#         # out_dict1, _, _, out_dict2, _, _ = self.net(seq_dict=seq_dict, feat_dict=feat_dict_list_sear_out,
#         #                                             mode="transformer", run_box_head=run_box_head,
#         #                                             run_cls_head=run_cls_head)                    # 20231028
#         # out_dict: (B, N, C), outputs_coord: (1, B, N, C), target_query: (1, B, N, C)
#         # return out_dict1, out_dict2
#         return out_dict

#     def compute_losses(self, pred_dict, gt_bbox, return_status=True):
#         # Get boxes
#         pred_boxes = pred_dict['pred_boxes']
#         if torch.isnan(pred_boxes).any():
#             raise ValueError("Network outputs is NAN! Stop Training")
#         num_queries = pred_boxes.size(1)
#         pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
#         gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
#         # compute giou and iou
#         try:
#             giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
#         except:
#             giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
#         # compute l1 loss
#         l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
#         # weighted sum
#         loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
#         if return_status:
#             # status for log
#             mean_iou = iou.detach().mean()
#             status = {"Loss/total": loss.item(),
#                       "Loss/giou": giou_loss.item(),
#                       "Loss/l1": l1_loss.item(),
#                       "IoU": mean_iou.item()}
#             return loss, status
#         else:
#             return loss