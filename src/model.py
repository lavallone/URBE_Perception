from collections import Counter
import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Resize
from torchvision.transforms import InterpolationMode
import torchvision.utils
import wandb
import math
import numpy as np
import pytorch_lightning as pl
from .data_module import URBE_DataModule
import random
from torchmetrics import Precision

# After the computation of the 'autoanchor' algorithm, we acknowledge that these are the "best" anchors (the default ones used in YOLOv5)
# https://github.com/ultralytics/yolov5/blob/master/models/yolov5m.yaml
ANCHORS = [ [(10, 13), (16, 30), (33, 23)],  # P3/8
            [(30, 61), (62, 45), (59, 119)],  # P4/16
            [(116, 90), (156, 198), (373, 326)] ]  # P5/32

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

# They are just a composition of the above "basic building blocks"
## BACKBONE ##
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

## NECK ##
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

################################################################################################################
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
################################################################################################################

## HEADs ##
class SimpleHead(nn.Module):
    def __init__(self, nc=3, anchors=(), ch=()):  # detection layer
        super(SimpleHead, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])

        # https://pytorch.org/docs/stable/generated/torch.nn.Module.html command+f register_buffer
        # has the same result as self.anchors = anchors but, it's a way to register a buffer (make
        # a variable available in runtime) that should not be considered a model parameter
        self.stride = [8, 16, 32]

        # anchors are divided by the stride (anchors_for_head_1/8, anchors_for_head_1/16 etc.)
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
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
    
class DecoupledHead(nn.Module):
    def __init__(self, nc=3, anchors=(), ch=()):  # detection layer
        super(DecoupledHead, self).__init__()
        self.nc = nc  # number of classes
        self.nl = len(anchors)  # number of detection layers
        self.naxs = len(anchors[0])
        self.stride = [8, 16, 32]
        anchors_ = torch.tensor(anchors).float().view(self.nl, -1, 2) / torch.tensor(self.stride).repeat(6, 1).T.reshape(3, 3, 2)
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
            # reshaping output to be (bs, n_scale_predictions, n_grid_y, n_grid_x, 5 + num_classes)
            # why .permute? Here https://github.com/ultralytics/yolov5/issues/10524#issuecomment-1356822063
            output = output.view(bs, self.naxs, (5+self.nc), grid_y, grid_x).permute(0, 1, 3, 4, 2).contiguous()
            outputs.append(output)
        return outputs

class URBE_Perception(pl.LightningModule):
    """ custom YOLOv5m (medium size) """
    def __init__(self, hparams):
        super(URBE_Perception, self).__init__()
        self.save_hyperparameters(hparams)
        self.anchors = ANCHORS
    
        self.backbone = Backbone(self.hparams.first_out)
        self.neck = Neck(self.hparams.first_out)
        if self.hparams.head == "simple":
            self.head = SimpleHead(anchors=self.anchors, ch=(self.hparams.first_out * 4, self.hparams.first_out * 8, self.hparams.first_out * 16))
        elif self.hparams.head == "decoupled":
            self.head = DecoupledHead(anchors=self.anchors, ch=(self.hparams.first_out * 4, self.hparams.first_out * 8, self.hparams.first_out * 16))
        
        # if the backbone is pretrained I don't train it at all
        if self.hparams.load_pretrained:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.val_precision = Precision(task = 'binary', num_classes = 2)

    def forward(self, x): # we expect x to be the stack of images
        x, backbone_connection = self.backbone(x)
        features = self.neck(x, backbone_connection)
        return self.head(features) # [(batch, 3, 80, 80, 8), (batch, 3, 40, 40, 8), (batch, 3, 20, 20, 8)]
    
    def predict(self, recon=None, batch = None):
        return

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

    def loss_function(self,recon_x, x):
        # the loss function is simply the main loss here
        return {"loss": self.main_loss(recon_x, x)}

    def training_step(self, batch, batch_idx):
        imgs = batch['img']
        recon = self(imgs)
        loss = self.loss_function(recon, imgs)
        # LOSS
        self.log_dict(loss)
        return {'loss': loss['loss']}

    def training_epoch_end(self, outputs):
        a = np.stack([x['anom'] for x in outputs]) 
        a_std = np.stack([x['a_std'] for x in outputs]) 
        self.avg_anomaly = a.mean()
        self.std_anomaly = a_std.mean()
        self.log_dict({"anom_avg": self.avg_anomaly, "anom_std": self.std_anomaly})
        self.log("anomaly_threshold", self.hparams.threshold, on_step=False, on_epoch=True, prog_bar=True)

    # # images logging during training phase but used for validation images
    # def get_images_for_log(self, real, reconstructed):
    # 	example_images = []
    # 	real = MVTec_DataModule.denormalize(real)
    # 	reconstructed = MVTec_DataModule.denormalize(reconstructed)
    # 	for i in range(real.shape[0]):
    # 		couple = torchvision.utils.make_grid(
    # 			[real[i], reconstructed[i]],
    # 			nrow=2,
    # 			scale_each=False,
    # 			pad_value=1,
    # 			padding=4,
    # 		)  
    # 		example_images.append(
    # 			wandb.Image(couple.permute(1, 2, 0).detach().cpu().numpy(), mode="RGB")
    # 		)
    # 	return example_images

    # def validation_step(self, batch, batch_idx):
    # 	imgs = batch['img']
    # 	recon_imgs = self(imgs)
    # 	# LOSS
    # 	self.log("val_loss", self.loss_function(recon_imgs, imgs)["loss"], on_step=False, on_epoch=True, batch_size=imgs.shape[0])
    # 	# RECALL, PRECISION, F1 SCORE
    # 	pred = self.anomaly_prediction(imgs, recon_imgs)
    # 	# good practice to follow with pytorch_lightning for logging values each iteration!
      # 	# https://github.com/Lightning-AI/lightning/issues/4396
    # 	self.val_precision.update(pred, batch['label'])
    # 	self.log("precision", self.val_precision, on_step=False, on_epoch=True, prog_bar=True, batch_size=imgs.shape[0])
    # 	# IMAGES
    # 	if self.hparams.log_image_each_epoch!=0:
    # 		images = self.get_images_for_log(imgs[0:self.hparams.log_images], recon_imgs[0:self.hparams.log_images])
    # 		return {"images": images}
    # 	else:
    # 		return None

    # def validation_epoch_end(self, outputs):
    # 	if self.hparams.log_image_each_epoch!=0 and self.global_step%self.hparams.log_image_each_epoch==0:
    # 		# we randomly select one batch index
    # 		bidx = random.randrange(100) % len(outputs)
    # 		images = outputs[bidx]["images"]
    # 		self.logger.experiment.log({f"images": images})