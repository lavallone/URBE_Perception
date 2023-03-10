import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import wandb
import pytorch_lightning as pl
from .loss import YOLO_Loss, intersection_over_union
import random
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.ops import batched_nms
import torchvision.transforms as T
import time

########################################## BASIC BUILDING BLOCKS ##############################################
##                                                                                                           ##
## All these blocks are entirely taken (with just few little changes) from the repo of an italian AI         ##
## enthusiast who implemented a "personal implementation of YOLOv5". That was exactly what I needed!         ##
## Thanks to https://github.com/AlessandroMondin/YOLOV5m :)                                                  ##
##                                                                                                           ##
###############################################################################################################
# performs a convolution, a batch_norm and then applies a SiLU activation function
class CBL(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(CBL, self).__init__()

        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        bn = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)

        self.cbl = nn.Sequential(
            conv,
            bn,
            # https://pytorch.org/docs/stable/generated/torch.nn.SiLU.html
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        #print(self.cbl(x).shape)
        return self.cbl(x)

# which is just a residual block
class Bottleneck(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
    """
    def __init__(self, in_channels, out_channels, width_multiple=1):
        super(Bottleneck, self).__init__()
        c_ = int(width_multiple*in_channels)
        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c2 = CBL(c_, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        return self.c2(self.c1(x)) + x

# kind of CSP backbone (https://arxiv.org/pdf/1911.11929v1.pdf)
class C3(nn.Module):
    """
    Parameters:
        in_channels (int): number of channel of the input tensor
        out_channels (int): number of channel of the output tensor
        width_multiple (float): it controls the number of channels (and weights)
                                of all the convolutions beside the
                                first and last one. If closer to 0,
                                the simpler the modelIf closer to 1,
                                the model becomes more complex
        depth (int): it controls the number of times the bottleneck (residual block)
                        is repeated within the C3 block
        backbone (bool): if True, self.seq will be composed by bottlenecks 1, if False
                            it will be composed by bottlenecks 2 (check in the image linked below)
        https://user-images.githubusercontent.com/31005897/172404576-c260dcf9-76bb-4bc8-b6a9-f2d987792583.png

    """
    def __init__(self, in_channels, out_channels, width_multiple=1, depth=1, backbone=True):
        super(C3, self).__init__()
        c_ = int(width_multiple*in_channels)

        self.c1 = CBL(in_channels, c_, kernel_size=1, stride=1, padding=0)
        self.c_skipped = CBL(in_channels,  c_, kernel_size=1, stride=1, padding=0)
        if backbone:
            self.seq = nn.Sequential(
                *[Bottleneck(c_, c_, width_multiple=1) for _ in range(depth)]
            )
        else:
            self.seq = nn.Sequential(
                *[nn.Sequential(
                    CBL(c_, c_, 1, 1, 0),
                    CBL(c_, c_, 3, 1, 1)
                ) for _ in range(depth)]
            )
        self.c_out = CBL(c_ * 2, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = torch.cat([self.seq(self.c1(x)), self.c_skipped(x)], dim=1)
        return self.c_out(x)

# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()

        c_ = int(in_channels//2)

        self.c1 = CBL(in_channels, c_, 1, 1, 0)
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.c_out = CBL(c_ * 4, out_channels, 1, 1, 0)

    def forward(self, x):
        x = self.c1(x)
        pool1 = self.pool(x)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)

        return self.c_out(torch.cat([x, pool1, pool2, pool3], dim=1))

# in the PANET the C3 block is different: no more CSP but a residual block composed
# a sequential branch of n SiLUs and a skipped branch with one SiLU
class C3_NECK(nn.Module):
    def __init__(self, in_channels, out_channels, width, depth):
        super(C3_NECK, self).__init__()
        c_ = int(in_channels*width)
        self.in_channels = in_channels
        self.c_ = c_
        self.out_channels = out_channels
        self.c_skipped = CBL(in_channels, c_, 1, 1, 0)
        self.c_out = CBL(c_*2, out_channels, 1, 1, 0)
        self.silu_block = self.make_silu_block(depth)

    def make_silu_block(self, depth):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(CBL(self.in_channels, self.c_, 1, 1, 0))
            elif i % 2 == 0:
                layers.append(CBL(self.c_, self.c_, 3, 1, 1))
            elif i % 2 != 0:
                layers.append(CBL(self.c_, self.c_, 1, 1, 0))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.c_out(torch.cat([self.silu_block(x), self.c_skipped(x)], dim=1))
##############################################################################################################################


####################################################### BACKBONE #############################################################
class Backbone(nn.Module):
    def __init__(self, first_out=48):
        super().__init__()
        self.backbone = nn.ModuleList()
        self.first_out = first_out
        self.backbone += [
            CBL(in_channels=3, out_channels=self.first_out, kernel_size=6, stride=2, padding=2),
            CBL(in_channels=self.first_out, out_channels=self.first_out*2, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*2, out_channels=self.first_out*2, width_multiple=0.5, depth=2),
            CBL(in_channels=self.first_out*2, out_channels=self.first_out*4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*4, out_channels=self.first_out*4, width_multiple=0.5, depth=4),
            CBL(in_channels=self.first_out*4, out_channels=self.first_out*8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*8, out_channels=self.first_out*8, width_multiple=0.5, depth=6),
            CBL(in_channels=self.first_out*8, out_channels=self.first_out*16, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*16, out_channels=self.first_out*16, width_multiple=0.5, depth=2),
            SPPF(in_channels=self.first_out*16, out_channels=self.first_out*16)
        ]
    
    def forward(self, x):
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0, "Width and Height aren't divisible by 32!"
        backbone_connection = []
        for idx, layer in enumerate(self.backbone):
            # takes the out of the 2nd and 3rd C3 block and stores it
            x = layer(x)
            if idx in [4, 6]:
                backbone_connection.append(x)
        return x, backbone_connection

######################################################### NECK ###############################################################
class Neck(nn.Module):
    def __init__(self, first_out=48):
        super().__init__()
        self.neck = nn.ModuleList()
        self.first_out = first_out
        self.neck += [
            CBL(in_channels=self.first_out*16, out_channels=self.first_out*8, kernel_size=1, stride=1, padding=0),
            C3(in_channels=self.first_out*16, out_channels=self.first_out*8, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=self.first_out*8, out_channels=self.first_out*4, kernel_size=1, stride=1, padding=0),
            C3(in_channels=self.first_out*8, out_channels=self.first_out*4, width_multiple=0.25, depth=2, backbone=False),
            CBL(in_channels=self.first_out*4, out_channels=self.first_out*4, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*8, out_channels=self.first_out*8, width_multiple=0.5, depth=2, backbone=False),
            CBL(in_channels=self.first_out*8, out_channels=self.first_out*8, kernel_size=3, stride=2, padding=1),
            C3(in_channels=self.first_out*16, out_channels=self.first_out*16, width_multiple=0.5, depth=2, backbone=False)
        ]
    
    def forward(self, x, backbone_connection):
        neck_connection = []
        outputs = []
        for idx, layer in enumerate(self.neck):
            if idx in [0, 2]:
                x = layer(x)
                neck_connection.append(x)
                x = Resize([x.shape[2]*2, x.shape[3]*2], interpolation=InterpolationMode.NEAREST)(x)
                x = torch.cat([x, backbone_connection.pop(-1)], dim=1)

            elif idx in [4, 6]:
                x = layer(x)
                x = torch.cat([x, neck_connection.pop(-1)], dim=1)

            elif (isinstance(layer, C3_NECK) and idx > 2) or (isinstance(layer, C3) and idx > 2):
                x = layer(x)
                outputs.append(x)

            else:
                x = layer(x)
        return outputs

######################################################### HEADs ##############################################################
class SimpleHead(nn.Module):
    def __init__(self, nc=3, ch=()):  # detection layer
        super(SimpleHead, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(URBE_Perception.ANCHORS)  # number of detection layers
        self.naxs = len(URBE_Perception.ANCHORS[0])

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
        # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
        # a variable available in runtime) that should not be considered a model parameter
        self.stride = URBE_Perception.STRIDE

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(URBE_Perception.ANCHORS).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_)  # shape(nl,na,2)

        self.out_convs = nn.ModuleList()
        for in_channels in ch:
            self.out_convs += [
                nn.Conv2d(in_channels=in_channels, out_channels=(5+self.nc) * self.naxs, kernel_size=1)
            ]

    def forward(self, x):
        for i in range(self.nl):
            # performs out_convolution and stores the result in place
            x[i] = self.out_convs[i](x[i])

            bs, _, grid_y, grid_x = x[i].shape
            # reshaping output to be (bs, n_scale_predictions, n_grid_y, n_grid_x, 5 + num_classes)
            # why .permute? Here https://github.com/ultralytics/yolov5/issues/10524#issuecomment-1356822063
            x[i] = x[i].view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()

        return x

# a BASIC convolutional block which computes also Normalization and Activation afterwards
class BaseConv(nn.Module):
	"""A Convolution2d -> Normalization -> Activation"""
	def __init__(self, in_channels, out_channels, ksize, stride, padding=None, groups=1, bias=False, norm="bn", act="silu"):
		super().__init__()
		pad = (ksize - 1) // 2 if padding is None else padding
		self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=ksize, stride=stride, padding=pad, groups=groups, bias=bias,)
		self.norm = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
		self.act = nn.SiLU(inplace=True)
	def forward(self, x):
		if self.norm is None and self.act is None:
			return self.conv(x)
		elif self.act is None:
			return self.norm(self.conv(x))
		elif self.norm is None:
			return self.act(self.conv(x))
		return self.act(self.norm(self.conv(x)))

class DecoupledHead(nn.Module):
    def __init__(self, nc=3, ch=()):  # detection layer
        super(DecoupledHead, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(URBE_Perception.ANCHORS)  # number of detection layers
        self.naxs = len(URBE_Perception.ANCHORS[0])
        self.stride = URBE_Perception.STRIDE
        anchors_ = torch.tensor(URBE_Perception.ANCHORS).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
        self.register_buffer('anchors', anchors_)  # shape(nl,na,2)
        
        self.stems = nn.ModuleList() # stem layer performs a sort of compression mechanism
        self.cls_convs = nn.ModuleList() # block to extract features for the CLASSIFICATION HEAD
        self.cls_preds = nn.ModuleList()
        self.reg_convs = nn.ModuleList() # block to extract features for the REGRESSION HEAD
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        for i in range(len(ch)): # for each layer
            self.stems.append(BaseConv(ch[i], ch[0], ksize=1, stride=1))
            self.cls_convs.append(nn.Sequential(*[BaseConv(ch[0], ch[0], ksize=3, stride=1),
                                                  BaseConv(ch[0], ch[0], ksize=3, stride=1),]
                                               )
                                 )
            self.cls_preds.append(nn.Conv2d(ch[0], self.nc * self.naxs, kernel_size=(1, 1), stride=(1, 1), padding=0))
            self.reg_convs.append(nn.Sequential(*[BaseConv(ch[0], ch[0], ksize=3, stride=1),
                                                  BaseConv(ch[0], ch[0], ksize=3, stride=1),]
                                               )
                                 )
            self.reg_preds.append(nn.Conv2d(ch[0], self.naxs * 4, kernel_size=(1, 1), stride=(1, 1), padding=0))
            self.obj_preds.append(nn.Conv2d(ch[0], self.naxs * 1, kernel_size=(1, 1), stride=(1, 1), padding=0))

    def forward(self, inputs):
        outputs = []
        for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, inputs)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)
            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)
            # the order of each "grid" output is objectness, bboxes and finally the predicted classes
            output = torch.cat([reg_output, obj_output, cls_output], 1)
            
            bs, _, grid_y, grid_x = output.shape
            output = output.view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(output)
        return outputs
##############################################################################################################################

class URBE_Perception(pl.LightningModule):
    
    # After the computation of the 'autoanchor' algorithm, we acknowledge that these are the "best" anchors (the default ones used in YOLOv5)
    # https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
    ANCHORS = [ [(10, 13), (16, 30), (33, 23)],  # P3/8
            [(30, 61), (62, 45), (59, 119)],  # P4/16
            [(116, 90), (156, 198), (373, 326)] ]  # P5/32
    
    # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
    # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
    # a variable available in runtime) that should not be considered a model parameter
    STRIDE = [8, 16, 32]
    
    """ custom YOLOv5 """
    def __init__(self, hparams):
        super(URBE_Perception, self).__init__()
        self.save_hyperparameters(hparams)
    
        self.backbone = Backbone(self.hparams.first_out)
        self.neck = Neck(self.hparams.first_out)
        if self.hparams.head == "simple":
            self.head = SimpleHead(ch=(self.hparams.first_out * 4, self.hparams.first_out * 8, self.hparams.first_out * 16))
        elif self.hparams.head == "decoupled":
            self.head = DecoupledHead(ch=(self.hparams.first_out * 4, self.hparams.first_out * 8, self.hparams.first_out * 16))
        
        # if are loaded backbone/neck pretrained weights I don't train some layers to save memory space!
        if self.hparams.load_pretrained:
            for param in self.backbone.backbone[:7].parameters(): # until the 6th layer
                param.requires_grad = False
                
        self.loss = YOLO_Loss(self.hparams)
        self.mAP = MeanAveragePrecision()

    def forward(self, x): # we expect x to be the stack of images
        x, backbone_connection = self.backbone(x)
        features = self.neck(x, backbone_connection)
        return self.head(features) # [(batch, 3, 80, 80, 8), (batch, 3, 40, 40, 8), (batch, 3, 20, 20, 8)]
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr, eps=self.hparams.adam_eps, weight_decay=self.hparams.wd)
        reduce_lr_on_plateau = ReduceLROnPlateau(optimizer, mode='min',verbose=True, min_lr=self.hparams.min_lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": reduce_lr_on_plateau,
                "monitor": 'loss',
                "frequency": 1
            },
        }

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        out = self(imgs)
        loss = self.loss(out, batch["labels"])
        # LOSS
        self.log_dict({"loss": loss})
        return {"loss": loss}

    # =======================================================================================#
    def make_grids(self, anchors, naxs, stride, nx=20, ny=20, i=0):

        x_grid = torch.arange(nx)
        x_grid = x_grid.repeat(ny).reshape(ny, nx)

        y_grid = torch.arange(ny).unsqueeze(0)
        y_grid = y_grid.T.repeat(1, nx).reshape(ny, nx)

        xy_grid = torch.stack([x_grid, y_grid], dim=-1)
        xy_grid = xy_grid.expand(1, naxs, ny, nx, 2)
        anchor_grid = (anchors[i]*stride).reshape((1, naxs, 1, 1, 2)).expand(1, naxs, ny, nx, 2)

        return xy_grid.to(self.device), anchor_grid.to(self.device)

    def cells_to_bboxes(self, predictions, anchors, strides, device, is_pred=False):
        num_out_layers = len(predictions) # num of scales 
        grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
        anchor_grid = [torch.empty(0) for _ in range(num_out_layers)]  # initialize
            
        all_bboxes = []
        for i in range(num_out_layers):
            bs, naxs, ny, nx, _ = predictions[i].shape # (bs, 3, 80/40/20, 80/40/20, _)
            stride = strides[i] # 8/16/32
            # grid rappresenta la griglia (80x80, 40x40, ...) con gli indici
            # invece anchor_grid ha lo stesso numero di celle, ma con i valori degli anchors
            grid[i], anchor_grid[i] = self.make_grids(anchors, naxs, stride, ny=ny, nx=nx, i=i) # entrambi torch.Size([1, 3, 80, 80, 2]) - 1 è per avere una dimensione inpiù epr lavoarre con i batches
            if is_pred: # if they are the predicitons made by the model
                # formula here: https://github.com/ultralytics/yolov5/issues/471
                layer_prediction = predictions[i].sigmoid()
                obj = layer_prediction[..., 4:5]
                xy = (2 * (layer_prediction[..., 0:2]) + grid[i] - 0.5) * stride
                wh = ((2*layer_prediction[..., 2:4])**2) * anchor_grid[i]
                best_class = torch.argmax(layer_prediction[..., 5:], dim=-1).unsqueeze(-1) # quindi da 8 diventa 6!

            else: # when we want to re-convert the ground_truth labels to images bboxes
                if i != num_out_layers-1:
                    continue
                predictions[i] = predictions[i].to(device, non_blocking=True)
                obj = predictions[i][..., 4:5]
                xy = (predictions[i][..., 0:2] + grid[i]) * stride
                wh = predictions[i][..., 2:4] * stride
                best_class = predictions[i][..., 5:6]

            scale_bboxes = torch.cat((best_class, obj, xy, wh), dim=-1).reshape(bs, -1, 6)
            all_bboxes.append(scale_bboxes)
        return torch.cat(all_bboxes, dim=1)

    def non_max_suppression(self, batch_bboxes, iou_threshold, threshold, max_detections=50, is_pred=False): # it can be run an analysis about which is the best choice for max_detection for image
        # for statistics purposes
        conf_thresh_ratio = 0
        nms_ratio = 0
        
        bboxes_after_nms = []
        for boxes in batch_bboxes: # we iterate over the batches
            # 'boxes' is the set of cells for one batch --> (25200, 6)
            # FIRST FILTER on the probability of objectness
            boxes = torch.masked_select(boxes, boxes[..., 1:2] > threshold).reshape(-1, 6) # if objectness is greater than the threshold, we continue...
            conf_thresh_ratio += len(boxes) / 25200
            # from (xc, yc, w, h) to (x1, y1, x2, y2) --> it is perfect for wandb bbox visualization!
            boxes[..., 2:3] = boxes[..., 2:3] - (boxes[..., 4:5]/2) # x1
            boxes[..., 3:4] = boxes[..., 3:4] - (boxes[..., 5:6]/2) # y1
            boxes[..., 4:5] = boxes[..., 2:3] + (boxes[..., 4:5]/2) # x2
            boxes[..., 5:6] = boxes[..., 3:4] + (boxes[..., 5:6]/2) # y2
            
            # we perform non maxima suppression(nms)
            if is_pred:
                indices = batched_nms(boxes[..., 2:], boxes[..., 1], boxes[..., 0].int(), iou_threshold)
                
                if indices.numel() == 0:
                    print("***NO PREDICTIONS***")
                    boxes = torch.tensor([])
                else:
                    before_nms = len(boxes)
                    boxes = boxes[indices]
                    nms_ratio += len(boxes) / before_nms
                    
                    # we set a maximum number of predictions for each image
                    if boxes.shape[0] > max_detections:
                        boxes = boxes[:max_detections, :]
            
            bboxes_after_nms.append(boxes)
        
        # if we're dealing with predictions we also want to save some statistics    
        if is_pred:
            return conf_thresh_ratio/len(batch_bboxes), nms_ratio/len(batch_bboxes), bboxes_after_nms # return a list of tensors --> len(bboxes_after_nms) == batch_size
        else:
            return bboxes_after_nms
    
    # images logging during training phase but used for validation images
    def get_images_for_log(self, imgs, bboxes, labels, scores):
        # we prepare each image
        transform = T.ToPILImage()
        images_list = [transform(img) for img in imgs]
        
        example_images = []
        for i, bbox in enumerate(bboxes): # for each image of the batch
            ris = { "predictions" : {"box_data" : [], "class_labels" : {0 : "vehicle" , 1 : "person", 2 : "motorbike"}} }
            for box, label, score in zip(bbox, labels[i], scores[i]): # for each bbox of the particular image
                position = {"minX": box[0].item(), "maxX": box[2].item(), "minY": box[1].item(), "maxY": box[3].item()}
                class_id = int(label)
                score  = float(score)
                box_caption = ris["predictions"]["class_labels"][class_id]
                x = {"position" : position, "domain" : "pixel", "class_id" : class_id, "box_caption" : box_caption, "scores" : {"score" : score}}
                ris["predictions"]["box_data"].append(x)
            example_images.append(wandb.Image(images_list[i], boxes=ris))
        return example_images
    # =======================================================================================#
    
    def predict(self, predictions, targets):
        targets = [YOLO_Loss.transform_targets(predictions, bboxes, torch.tensor(URBE_Perception.ANCHORS), URBE_Perception.STRIDE) for bboxes in targets]
        # I want targets to be the same shape as predictions --> (bs, 3 , 80/40/20, 80/40/20, 6)
        t1 = torch.stack([target[0] for target in targets], dim=0).to(self.device,non_blocking=True)
        t2 = torch.stack([target[1] for target in targets], dim=0).to(self.device,non_blocking=True)
        t3 = torch.stack([target[2] for target in targets], dim=0).to(self.device,non_blocking=True)
        targets = [t1, t2, t3] # all the batches are grouped according to the scale
        
        ## Custom "ACCURACY" for classes and objectness ##
        tot_class, correct_class = 0, 0 # total number of objects to be predicted and the  number of objects to be correctly predicted
        tot_obj, correct_obj = 0, 0
        for i in range(3): # for each layer/scale
            targets[i] = targets[i].to(self.device)
            obj = targets[i][..., 4] == 1 # mask for the target bboxes
            tot_class += torch.sum(obj)
            tot_obj += torch.sum(obj)
            
            # among the grid cells that contains an object, see if the most likely predicted class is the target one
            correct_class += torch.sum(
                torch.argmax(predictions[i][..., 5:][obj], dim=-1) == targets[i][..., 5][obj]
            )
            
            # we filter out the objects that are not "actual" predictions for the model and
            # we change the objectness score of the kept ones to 1
            obj_preds = torch.sigmoid(predictions[i][..., 4]) > self.hparams.conf_threshold
            # among the grid cells that contains an object, see if the cell is predicted as a cell which detects an object
            correct_obj += torch.sum(obj_preds[obj] == targets[i][..., 4][obj])
        
        ## mAP_50 ##
        pred_boxes = self.cells_to_bboxes(predictions, torch.tensor(URBE_Perception.ANCHORS), URBE_Perception.STRIDE, self.device,  is_pred=True) # (bs, 25200, 6) --> we use all the three layers
        true_boxes = self.cells_to_bboxes(targets, torch.tensor(URBE_Perception.ANCHORS), URBE_Perception.STRIDE, self.device, is_pred=False) # (bs, 20*20*3, 6)
        # after 'cell_to_boxes' the bboxes are set for 640x640 image size (indeed not normalized)
        conf_thresh_ratio, nms_ratio, pred_boxes = self.non_max_suppression(pred_boxes, iou_threshold=self.hparams.nms_iou_thresh, threshold=self.hparams.conf_threshold, max_detections=50, is_pred=True)
        true_boxes = self.non_max_suppression(true_boxes, iou_threshold=self.hparams.nms_iou_thresh, threshold=self.hparams.conf_threshold, max_detections=50, is_pred=False)

        pred_dict_list = []
        for b in range(len(pred_boxes)):
            if pred_boxes[b].numel() == 0: # if the model hasn't predict any bboxes
                pred_dict_list.append( dict(boxes=torch.tensor([]).to("cuda"), scores=torch.tensor([]).to("cuda"), labels=torch.tensor([]).to("cuda"),) )
            else:
                pred_dict_list.append( dict(boxes=pred_boxes[b][..., 2:], scores=pred_boxes[b][..., 1], labels=pred_boxes[b][..., 0],) )
        true_dict_list = [ dict(boxes=true_boxes[i][..., 2:], labels=true_boxes[i][..., 0],) for i in range(len(true_boxes)) ]
        
        return conf_thresh_ratio, nms_ratio, {"mAP" : (pred_dict_list, true_dict_list) , "accuracy" : (tot_class, correct_class, tot_obj, correct_obj)}

    def validation_step(self, batch, batch_idx):

        imgs = batch['img']
        out = self(imgs)
        val_loss = self.loss(out, batch["labels"])
        
        # LOSS
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, batch_size=imgs.shape[0])
        
        conf_thresh_ratio, nms_ratio, pred = self.predict(out, batch['labels'])
        # STATISTICS
        self.log("conf_thresh_ratio", conf_thresh_ratio, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("nms_ratio", nms_ratio, on_step=False, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        # METRICS
        self.log("val_accuracy_class", (pred["accuracy"][1] / (pred["accuracy"][0] + 1e-16)), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        self.log("val_accuracy_obj", (pred["accuracy"][3] / (pred["accuracy"][2] + 1e-16)), on_step=True, on_epoch=True, prog_bar=True, batch_size=self.hparams.batch_size)
        
        # good  practice for logging metrics in lightning
        # see https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html
        self.mAP.update(pred["mAP"][0], pred["mAP"][1])
                
     	# IMAGES
        if self.hparams.log_image_each_epoch != 0:
            bboxes = [e["boxes"] for e in (pred["mAP"][0][0:self.hparams.log_images]) ]
            labels = [e["labels"] for e in (pred["mAP"][0][0:self.hparams.log_images]) ]
            scores = [e["scores"] for e in (pred["mAP"][0][0:self.hparams.log_images]) ]
            images = self.get_images_for_log(imgs[0:self.hparams.log_images], bboxes, labels, scores)
            return {"images": images}
        else:
            return None
    
    # we keep it only for image logging purposes
    def validation_epoch_end(self, outputs):
        self.log('map_50', self.mAP.compute()["map_50"])
        self.mAP.reset()
        
        if self.hparams.log_image_each_epoch!=0 and self.current_epoch%self.hparams.log_image_each_epoch==0:
            # we randomly select one batch index
            bidx = random.randrange(100) % len(outputs)
            images = outputs[bidx]["images"]
            self.logger.experiment.log({f"images": images})