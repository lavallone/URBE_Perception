import torch
import torch.nn as nn
import math

####################################################### UTILS ####################################################################
##################################################################################################################################
# https://github.com/aladdinpersson/Machine-Learning-Collection
# it is only needed during the targets transformation function
def iou_width_height(gt_box, anchors, strided_anchors=False, stride=[8, 16, 32]):
    """
    Parameters:
        gt_box (tensor): width and height of the ground truth box
        anchors (tensor): lists of anchors containing width and height
        strided_anchors (bool): if the anchors are divided by the stride or not
    Returns:
        tensor: Intersection over union between the gt_box and each of the n-anchors
    """
    anchors = anchors.float()
    anchors /= 640
    if strided_anchors:
        anchors = anchors.reshape(9, 2) * torch.tensor(stride).repeat(6, 1).T.reshape(9, 2)
    else:
        anchors = anchors.reshape(9, 2)
    anchors = anchors.to("cuda")
    intersection = torch.min(gt_box[..., 0], anchors[..., 0]) * torch.min(
        gt_box[..., 1], anchors[..., 1]
    )
    
    union = (
        gt_box[..., 0] * gt_box[..., 1] + anchors[..., 0] * anchors[..., 1] - intersection
    )
    
    # intersection/union shape (9,)
    return intersection / union

# added the possibility of computing also GIoU, DIoU and CIoU!
# we only use this function for the loss during training
def intersection_over_union(boxes_preds, boxes_labels, box_format="coco", GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    """
    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): coco/corners, if boxes (x1,y1,w,h) or (x1,y1,x2,y2)
        GIoU (bool): if True it computed GIoU loss (https://arxiv.org/pdf/1902.09630.pdf)
        DIoU (bool): if True it computed DIoU loss (https://arxiv.org/abs/1911.08287v1)
        CIoU (bool): if True it computed CIoU loss (https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47)
        eps (float): for numerical stability

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "coco":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2] 
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] 
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1] 
        box2_y1 = boxes_labels[..., 1:2] 
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4]

    else:
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    w1, h1, w2, h2 = box1_x2 - box1_x1, box1_y2 - box1_y1, box2_x2 - box2_x1, box2_y2 - box2_y1
    
    # Intersection area
    # clamp(0) is for the case when they do not intersect
    inter = (torch.min(box1_x2, box2_x2) - torch.max(box1_x1, box2_x1)).clamp(0) * \
            (torch.min(box1_y2, box2_y2) - torch.max(box1_y1, box2_y1)).clamp(0)

    # Union Area
    union = (w1 * h1) + (w2 * h2) - inter + eps

    iou = inter / union

    # lasciamo perdere per il momento
    if False: # GIoU or DIoU or CIoU:
        cw = torch.max(box1_x2, box2_x2) - torch.min(box1_x1, box2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(box1_y2, box2_y2) - torch.min(box1_y1, box2_y1)
        if CIoU or DIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((box2_x1 + box2_x2 - box1_x1 - box1_x2) ** 2 +
                    (box2_y1 + box2_y2 - box1_y1 - box1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        c_area = cw * ch + eps  # convex height
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    else: # simple IoU
        return iou 
##################################################################################################################################

class YOLO_Loss:
    
    # After the computation of the 'autoanchor' algorithm, we acknowledge that these are the "best" anchors (the default ones used in YOLOv5)
    # https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
    ANCHORS = torch.tensor([ [(10, 13), (16, 30), (33, 23)],  # P3/8
                             [(30, 61), (62, 45), (59, 119)],  # P4/16
                             [(116, 90), (156, 198), (373, 326)] ])  # P5/32
    
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
    # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
    # a variable available in runtime) that should not be considered a model parameter
    STRIDE = [8, 16, 32]
    
    # https://github.com/ultralytics/yolov5/issues/2026
    BALANCE = [4.0, 1.0, 0.4]
    
    @staticmethod
    # this function deal with one image at a time
    def transform_targets(input_tensor, bboxes, anchors, strides, ignore_iou_thresh, num_anchors_per_scale=3):
        targets = [torch.zeros((num_anchors_per_scale, input_tensor[i].shape[2], input_tensor[i].shape[3], 6))
                   for i in range(len(strides))]
    
        # bboxes is relative to a single batch --> (max_labels_batch, 5)
        classes = bboxes[:, 0].tolist()
        bboxes = bboxes[:, 1:]
        # filtering only the real annotations --> remember what we have done with the collate function!
        for i, e in enumerate(bboxes):
            if e.sum() == 0:
                classes = classes[:i]
                bboxes = bboxes[:i]
                break

        for idx, box in enumerate(bboxes):
            iou_anchors = iou_width_height(box[2:4], anchors) # we calculate the iou for the particular box and all the anchors
            anchor_indices = iou_anchors.argsort(descending=True, dim=0) # which anchors are the best?
            x, y, width, height = box
            has_anchor = [False] * 3 # we make sure that there is an anchor for each scale for each particular box
            for anchor_idx in anchor_indices: # we iterate starting from the "best ones" first
                scale_idx = torch.div(anchor_idx, num_anchors_per_scale, rounding_mode="floor") # in this way we know to which scale the anchor belongs to
                anchor_on_scale = anchor_idx % num_anchors_per_scale # which anchor in the particular scale
                
                scale_y = input_tensor[int(scale_idx)].shape[2]
                scale_x = input_tensor[int(scale_idx)].shape[3]
                i, j = int(scale_y * y), int(scale_x * x) # coordinates of the particular cell
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 4]
                
                # if the cell at a particular scale is not already taken (by another object) and we didn't pick it yet a cell at the particular scale
                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = 1 # we say that is "taken"
                    
                    x_cell, y_cell = scale_x * x - j, scale_y * y - i # coordinates of x and y w.r.t. the cell
                    width_cell, height_cell = (width * scale_x, height * scale_y,)
                    
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell]) # w.r.t. the cell
                    targets[scale_idx][anchor_on_scale, i, j, 0:4] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(classes[idx])
                    has_anchor[scale_idx] = True # for this scale and for this particular bbox we have the anchor
                # e come se servisse mettere il -1 agli anchor della stessa scala che non sono i migliori ma che comunque hanno un iou alto!
                elif not anchor_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 4] = -1  # ignore prediction
        return targets
    
    def __init__(self, hparams):

        #self.mse = nn.MSELoss()
        self.BCE_cls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.BCE_obj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.BCE_noobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.0))
        self.sigmoid = nn.Sigmoid()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.nc = hparams["num_classes"] # number of classes
        self.nl = len(YOLO_Loss.ANCHORS) # number of scale/layers
        self.anchors_d = YOLO_Loss.ANCHORS.clone().detach() # (3, 3, 2) --> they are exactly the anchor boxes, but modified
        self.anchors = YOLO_Loss.ANCHORS.clone().detach().to("cpu")

        self.na = self.anchors.reshape(9,2).shape[0] # number of anchors --> 9
        self.num_anchors_per_scale = self.na // 3 # number of anchors for each scale --> 3
        self.S = YOLO_Loss.STRIDE
        self.ignore_iou_thresh = 0.5
        
        # https://github.com/ultralytics/yolov5/blob/master/data/hyps/hyp.scratch-low.yaml)
        # https://github.com/ultralytics/yolov5/blob/master/utils/loss.py#L170)
        # https://github.com/ultralytics/yolov5/blob/master/train.py#L232)
        self.lambda_class = hparams["weight_class"] * (self.nc / 80 * 3 / self.nl) # scale to layers
        self.lambda_obj = hparams["weight_obj"] * ((hparams["img_size"] / 640) ** 2 * 3 / self.nl) # scale to classes and layers
        self.lambda_box = hparams["weight_box"] * (3 / self.nl) # scale to image size and layers

        self.balance = YOLO_Loss.BALANCE
        self.add_no_obj_loss = hparams["add_no_obj_loss"]

    def __call__(self, preds, targets):

        # we transform the targets in order to be able to compare them with the predictions output by the model
        targets = [YOLO_Loss.transform_targets(preds, bboxes, self.anchors, self.S, self.ignore_iou_thresh, self.num_anchors_per_scale) for bboxes in targets]

        t1 = torch.stack([target[0] for target in targets], dim=0).to(self.device, non_blocking=True)
        t2 = torch.stack([target[1] for target in targets], dim=0).to(self.device, non_blocking=True)
        t3 = torch.stack([target[2] for target in targets], dim=0).to(self.device, non_blocking=True)
        
        # we compute it layer by layer...
        loss = (
            self.compute_loss(preds[0], t1, anchors=self.anchors_d[0], balance=self.balance[0])
            + self.compute_loss(preds[1], t2, anchors=self.anchors_d[1], balance=self.balance[1])
            + self.compute_loss(preds[2], t3, anchors=self.anchors_d[2], balance=self.balance[2])
        )
        return {"loss" : loss}

    # the actual function which computes the TRAINING loss
    def compute_loss(self, preds, targets, anchors, balance):
        bs = preds.shape[0]
        # originally anchors have shape (3,2) --> 3 set of anchors of width and height
        anchors = anchors.reshape(1, 3, 1, 1, 2)
        anchors = anchors.to("cuda")
        
        obj = targets[..., 4] == 1
        if self.add_no_obj_loss:
            noobj = targets[..., 4] == 0
        
        pxy = (preds[..., 0:2].sigmoid() * 2) - 0.5
        pwh = ((preds[..., 2:4].sigmoid() * 2) ** 2) * anchors
        pbox = torch.cat((pxy[obj], pwh[obj]), dim=-1)
        tbox = targets[..., 0:4][obj]
        
        # iou computation
        iou = intersection_over_union(pbox, tbox, box_format="coco", GIoU=True).squeeze()

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        # (Aladdin does something weird with the targets for the iou computations for the gradient flow)
        lbox = (1.0 - iou).mean()  # iou loss

        # ======================= #
        #   FOR OBJECTNESS SCORE  #
        # ======================= #
        iou = iou.detach().clamp(0)
        targets[..., 4][obj] *= iou # instead of simply having objectness=1 for the targets
        lobj = self.BCE_obj(preds[..., 4], targets[..., 4]) * balance
        
        # ======================= #
        #   FOR NO_OBJECTNESS SCORE  #
        # ======================= #
        if self.add_no_obj_loss:
            lnoobj = self.BCE_noobj(preds[..., 4], targets[..., 4]) * balance
        
        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        tcls = torch.zeros_like(preds[..., 5:][obj], device=self.device)
        tcls[torch.arange(tcls.size(0)), targets[..., 5][obj].long()] = 1.0  # for torch > 1.11.0
        lcls = self.BCE_cls(preds[..., 5:][obj], tcls)

        return (self.lambda_box * lbox + self.lambda_obj * lobj + self.lambda_class * lcls) * bs # like in YOLOv5 official code