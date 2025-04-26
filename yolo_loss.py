import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def compute_iou(box1, box2):
    """Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    """
    N = box1.size(0)
    M = box2.size(0)

    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou


class YoloLoss(nn.Module):

    def __init__(self, S, B, l_coord, l_noobj):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.l_coord = l_coord
        self.l_noobj = l_noobj

    def xywh2xyxy(self, boxes):
        """
        Parameters:
        boxes: (N,4) representing by x,y,w,h

        Returns:
        boxes: (N,4) representing by x1,y1,x2,y2

        if for a Box b the coordinates are represented by [x, y, w, h] then
        x1, y1 = x/S - 0.5*w, y/S - 0.5*h ; x2,y2 = x/S + 0.5*w, y/S + 0.5*h
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        """
        ### CODE ###
        x, y = boxes[:,0], boxes[:,1]
        w, h = boxes[:,2], boxes[:,3]
        
        converted_boxes = torch.zeros_like(boxes)
        converted_boxes[:,0] = x/self.S - 0.5*w #x1
        converted_boxes[:,1] = y/self.S - 0.5*h #y1
        converted_boxes[:,2] = x/self.S + 0.5*w #x2
        converted_boxes[:,3] = y/self.S + 0.5*h #y2

        return converted_boxes

    def find_best_iou_boxes(self, pred_box_list, box_target):
        """
        Parameters:
        box_pred_list : (list) [(tensor) size (-1, 5)]  
        box_target : (tensor)  size (-1, 4)

        Returns:
        best_iou: (tensor) size (-1, 1)
        best_boxes : (tensor) size (-1, 5), containing the boxes which give the best iou among the two (self.B) predictions

        Hints:
        1) Find the iou's of each of the 2 bounding boxes of each grid cell of each image.
        2) For finding iou's use the compute_iou function
        3) Use xywh2xyxy to convert bbox format if necessary,
        Note: Over here initially x, y are the center of the box and w,h are width and height.
        We perform this transformation to convert the correct coordinates into bounding box coordinates.
        """

        ### CODE ###
        device = box_target.device
        results = [] # (N, 6) = (N, [x,y,w,h] + conf + iou)
        N = box_target.size(0)
        conv_box_target = self.xywh2xyxy(box_target).to(device) # (N, 4) = (N, [x1,y1,x2,y2])
        
        for pred_box in pred_box_list:
            pred_box = pred_box.to(device) # (N, 5) = (N, [x,y,w,h] + conf)
            
            conv_pred_box = self.xywh2xyxy(pred_box[:,:4]) # (N, 4) = (N, [x1,y1,x2,y2])
            # select sample-wise iou
            iou = compute_iou(conv_pred_box, conv_box_target).diagonal().unsqueeze(1) # (N, 1)
            results.append(torch.cat([pred_box, iou], dim=1))
        
        stacked_results = torch.stack(results, dim=1) # (N, B, 6)
        best_ious, best_indices = stacked_results[:,:,-1].max(dim=1) # select max(best) iou option out of B options
        
        best_ious = best_ious.unsqueeze(1) # (N, 1)
        best_boxes = stacked_results[torch.arange(N, device=stacked_results.device), best_indices, :5] # (N, 5)
        
        return best_ious, best_boxes # (N, 1) and (N, 5) = (N, [x,y,w,h,c]), respectively

    def get_class_prediction_loss(self, classes_pred, classes_target, has_object_map):
        """
        Parameters:
        classes_pred : (tensor) size (batch_size, S, S, 20)
        classes_target : (tensor) size (batch_size, S, S, 20)
        has_object_map: (tensor) size (batch_size, S, S)

        Returns:
        class_loss : scalar
        """
        ### CODE ###
        has_object_map =  has_object_map.unsqueeze(-1).float() # (batch_size, S, S, 1)
        pred_has_obj = classes_pred*has_object_map
        target_has_obj = classes_target*has_object_map
        
        loss = F.mse_loss(pred_has_obj, target_has_obj, reduction='sum')
        
        return loss

    def get_no_object_loss(self, pred_boxes_list, has_object_map):
        """
        Parameters:
        pred_boxes_list: (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        has_object_map: (tensor) size (N, S, S)

        Returns:
        loss : scalar

        Hints:
        1) Only compute loss for cell which doesn't contain object
        2) Compute loss for all predictions in the pred_boxes_list list
        3) You can assume the ground truth confidence of non-object cells is 0
        """
        ### CODE
        no_object_map = (~has_object_map).float().unsqueeze(-1) # revert has_object_map to get no_object_map
        loss = 0
        
        for box_pred_conf in pred_boxes_list:
            noobj_conf = box_pred_conf*no_object_map
            loss += torch.sum(noobj_conf[:,:,:,4] ** 2) # since ground truth confidence of non-object cells is 0
        
        return self.l_noobj*loss

    def get_contain_conf_loss(self, box_pred_conf, box_target_conf):
        """
        Parameters:
        box_pred_conf : (tensor) size (-1,1)
        box_target_conf: (tensor) size (-1,1)

        Returns:
        contain_loss : scalar

        Hints:
        The box_target_conf should be treated as ground truth, i.e., no gradient

        """
        ### CODE
        loss = F.mse_loss(box_pred_conf, box_target_conf, reduction='sum')
        return loss

    def get_regression_loss(self, box_pred_response, box_target_response):
        """
        Parameters:
        box_pred_response : (tensor) size (-1, 4)
        box_target_response : (tensor) size (-1, 4)
        Note : -1 corresponds to ravels the tensor into the dimension specified
        See : https://pytorch.org/docs/stable/tensors.html#torch.Tensor.view_as

        Returns:
        reg_loss : scalar

        """
        ### CODE
        xy_loss = self.l_coord * F.mse_loss(box_pred_response[:,:2], box_target_response[:,:2], reduction='sum')
        wh_loss = self.l_coord * F.mse_loss(box_pred_response[:,2:].sqrt(), box_target_response[:,2:].sqrt(), reduction='sum')
        reg_loss = xy_loss + wh_loss
        
        return reg_loss

    def forward(self, pred_tensor, target_boxes, target_cls, has_object_map):
        """
        pred_tensor: (tensor) size(N,S,S,Bx5+20=30) where:  
                            N - batch_size
                            S - width/height of network output grid%
                            B - number of bounding boxes this grid cell is a part of = 2
                            5 - number of bounding box values corresponding to [x, y, w, h, c]
                                where x - x_coord, y - y_coord, w - width, h - height, c - confidence of having an object
                            20 - number of classes

        target_boxes: (tensor) size (N, S, S, 4): the ground truth bounding boxes
        target_cls: (tensor) size (N, S, S, 20): the ground truth class
        has_object_map: (tensor, bool) size (N, S, S): the ground truth for whether each cell contains an object (True/False)

        Returns:
        loss_dict (dict): with key value stored for total_loss, reg_loss, containing_obj_loss, no_obj_loss and cls_loss
        """
        N = pred_tensor.size(0)
        has_object_map = has_object_map.to(pred_tensor.device)

        # split the pred tensor from an entity to separate tensors:
        # -- pred_boxes_list: a list containing all bbox prediction (list) [(tensor) size (N, S, S, 5)  for B pred_boxes]
        # -- pred_cls (containing all classification prediction)
        
        pred_boxes_list = []
        for b in range(self.B):
            start = b * 5
            end = (b + 1) * 5
            pred_boxes_list.append(pred_tensor[:, :, :, start:end])
        
        pred_cls = pred_tensor[:, :, :, self.B * 5:]
        
        # compute classification loss
        cls_loss = self.get_class_prediction_loss(pred_cls, target_cls, has_object_map)/N

        # compute no-object loss
        no_obj_loss = self.get_no_object_loss(pred_boxes_list, has_object_map)/N
        
        # Re-shape boxes in pred_boxes_list and target_boxes to meet the following desires
        # 1) only keep having-object cells
        # 2) vectorize all dimensions except for the last one for faster computation
        
        mask = has_object_map.view(N * self.S * self.S) # (N*S*S, )
        
        # reshape and mask target boxes
        target_boxes_flat = target_boxes.view(N * self.S * self.S, 4)
        target_boxes_masked = target_boxes_flat[mask]  # (-1, 4)
        
        # reshape and mask predicted boxes
        pred_boxes_masked = []
        for b in range(self.B):
            pred_b = pred_boxes_list[b].reshape(N * self.S * self.S, 5)
            pred_boxes_masked.append(pred_b[mask])  # (-1, 5)

        # find the best boxes among the 2 (or self.B) predicted boxes and the corresponding iou
        best_ious, best_boxes = self.find_best_iou_boxes(pred_boxes_masked, target_boxes_masked)
        
        # compute regression loss between the found best bbox and GT bbox for all the cell containing objects
        # best_boxes[:, :4] = (N, [x,y,w,h])
        # target_boxes_masked = (N, [x,y,w,h])
        reg_loss = self.get_regression_loss(best_boxes[:, :4], target_boxes_masked)/N
        
        # compute contain_object_loss
        containing_obj_loss = self.get_contain_conf_loss(best_boxes[:, 4:], best_ious.detach())/N

        # compute final loss
        total_loss = reg_loss + containing_obj_loss + no_obj_loss + cls_loss
        
        # construct return loss_dict
        loss_dict = dict(
            total_loss=total_loss,
            reg_loss=reg_loss,
            containing_obj_loss=containing_obj_loss,
            no_obj_loss=no_obj_loss,
            cls_loss=cls_loss,
        )
        
        return loss_dict
