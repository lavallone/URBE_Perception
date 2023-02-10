import torch
from torch import nn
import math

##############################################################################################################################
def get_activation(name="silu", inplace=True):
	if name is None:
		return None
	if name == "silu":
		module = nn.SiLU(inplace=inplace)
	elif name == "relu":
		module = nn.ReLU(inplace=inplace)
	elif name == "lrelu":
		module = nn.LeakyReLU(0.1, inplace=inplace)
	elif name == "gelu":
		module = nn.GELU()
	else:
		raise AttributeError("Unsupported activation function type: {}".format(name))
	return module
	
def get_normalization(name, out_channels):
	if name is None:
		return None
	if name == "bn":
		module = nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.03)
	elif name == "ln":
		module = nn.LayerNorm(out_channels)
	else:
		raise AttributeError("Unsupported normalization function type: {}".format(name))
	return module

class BaseConv(nn.Module):
	"""A Convolution2d -> Normalization -> Activation"""
	def __init__(self, in_channels, out_channels, ksize, stride, padding=None, groups=1, bias=False, norm="bn", act="silu"):
		super().__init__()
		# same padding
		if padding is None:
			pad = (ksize - 1) // 2
		else:
			pad = padding
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=ksize,
			stride=stride,
			padding=pad,
			groups=groups,
			bias=bias,
		)
		self.norm = get_normalization(norm, out_channels)
		self.act = get_activation(act, inplace=True)
	def forward(self, x):
		if self.norm is None and self.act is None:
			return self.conv(x)
		elif self.act is None:
			return self.norm(self.conv(x))
		elif self.norm is None:
			return self.act(self.conv(x))
		return self.act(self.norm(self.conv(x)))

class Focus(nn.Module):
	"""Focus width and height information into channel space."""
	def __init__(self, in_channels, out_channels, ksize=1, stride=1, norm='bn', act="silu"):
		super().__init__()
		self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, norm=norm, act=act)
	def forward(self, x):
		# shape of x (b,c,w,h) -> y(b,4c,w/2,h/2)
		patch_top_left = x[..., ::2, ::2]
		patch_top_right = x[..., ::2, 1::2]
		patch_bot_left = x[..., 1::2, ::2]
		patch_bot_right = x[..., 1::2, 1::2]
		x = torch.cat(
			(
				patch_top_left,
				patch_bot_left,
				patch_top_right,
				patch_bot_right,
			),
			dim=1,
		)
		return self.conv(x)

class Bottleneck(nn.Module):
	# Standard bottleneck from ResNet
	def __init__(
		self,
		in_channels,
		out_channels,
		shortcut=True,
		expansion=0.5,
		norm='bn',
		act="silu",
	):
		super().__init__()
		hidden_channels = int(out_channels * expansion)
		self.bn = get_normalization(norm, out_channels)
		self.act = get_activation(act, inplace=True)
		self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
		self.conv2 = BaseConv(hidden_channels, out_channels, 3, stride=1, norm=norm, act=act)
		self.use_add = shortcut and in_channels == out_channels

	def forward(self, x):
		y = self.conv2(self.conv1(x))
		if self.use_add:
			y = y + x
		return y

class CSPLayer(nn.Module):
	def __init__(
		self,
		in_channels,
		out_channels,
		num_bottle=1,
		shortcut=True,
		expansion=0.5,
		norm='bn',
		act="silu",
	):
		"""
		Args:
			in_channels (int): input channels.
			out_channels (int): output channels.
			num_bottle (int): number of Bottlenecks. Default value: 1.
			shortcut (bool): residual operation.
			expansion (int): the number that hidden channels compared with output channels.
			norm (str): type of normalization
			act (str): type of activation
		"""
		super().__init__()
		hidden_channels = int(out_channels * expansion)  # hidden channels
		self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
		self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
		self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, norm=norm, act=act)
		module_list = [
			Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, norm=norm, act=act)
			for _ in range(num_bottle)
		]
		self.m = nn.Sequential(*module_list)

	def forward(self, x):
		x_1 = self.conv1(x)
		x_2 = self.conv2(x)
		x_1 = self.m(x_1)
		x = torch.cat((x_1, x_2), dim=1)
		return self.conv3(x)

class SPPBottleneck(nn.Module):
	"""Spatial pyramid pooling layer used in YOLOv3-SPP"""
	def __init__(
		self, in_channels, out_channels, kernel_sizes=(5, 9, 13), norm='bn', act="silu"
	):
		super().__init__()
		hidden_channels = in_channels // 2
		self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, norm=norm, act=act)
		self.m = nn.ModuleList(
			[
				nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
				for ks in kernel_sizes
			]
		)
		conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
		self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

	def forward(self, x):
		x = self.conv1(x)
		x = torch.cat([x] + [m(x) for m in self.m], dim=1)
		x = self.conv2(x)
		return x
##############################################################################################################################

## BACKBONE ##
class DarknetCSP(nn.Module):
	"""
	CSPDarkNet
	Depths and Channels
	DarkNet-tiny   (1, 3, 3, 1)     (24, 48, 96, 192, 384)
	DarkNet-small  (2, 6, 6, 2)     (32, 64, 128, 256, 512)
	DarkNet-base   (3, 9, 9, 3)     (64, 128, 256, 512, 1024)
	DarkNet-large  (4, 12, 12, 4)   (64, 128, 256, 512, 1024)
	
	CSPDarkNet consists of five block: stem, dark2, dark3, dark4 and dark5.
	"""
	def __init__(self, depths=(3, 9, 9, 3), channels=(64, 128, 256, 512, 1024), out_features=("stage2", "stage3", "stage4"), norm='bn', act="silu",):
		super().__init__()

		# parameters of the network
		assert out_features, "please provide output features of Darknet!"
		self.out_features = out_features

		# stem
		self.stem = Focus(3, channels[0], ksize=3, norm=norm, act=act)

		# stage1
		self.stage1 = nn.Sequential(
			BaseConv(channels[0], channels[1], 3, 2, norm=norm, act=act),
			CSPLayer(channels[1], channels[1], num_bottle=depths[0], norm=norm, act=act),
		)

		# stage2
		self.stage2 = nn.Sequential(
			BaseConv(channels[1], channels[2], 3, 2, norm=norm, act=act),
			CSPLayer(channels[2], channels[2], num_bottle=depths[1], norm=norm, act=act),
		)

		# stage3
		self.stage3 = nn.Sequential(
			BaseConv(channels[2], channels[3], 3, 2, norm=norm, act=act),
			CSPLayer(channels[3], channels[3], num_bottle=depths[2], norm=norm, act=act),
		)

		# stage4
		self.stage4 = nn.Sequential(
			BaseConv(channels[3], channels[4], 3, 2, norm=norm, act=act),
			SPPBottleneck(channels[4], channels[4], norm=norm, act=act),
			CSPLayer(channels[4], channels[4], num_bottle=depths[3], shortcut=False, norm=norm, act=act),
		)

	def forward(self, x):
		outputs = {}
		x = self.stem(x)
		outputs["stem"] = x
		x = self.stage1(x)
		outputs["stage1"] = x
		x = self.stage2(x)
		outputs["stage2"] = x
		x = self.stage3(x)
		outputs["stage3"] = x
		x = self.stage4(x)
		outputs["stage4"] = x
		if len(self.out_features) <= 1:
			return x
		return [v for k, v in outputs.items() if k in self.out_features]

## NECKs ##
class PA_FPN_CSP(nn.Module):
	"""
	Only proceed 3 layer input. Like stage2, stage3, stage4.
	"""
	def __init__(self, depths=(1, 1, 1, 1), in_channels=(256, 512, 1024), norm='bn', act="silu",):
		super().__init__()
		self.shrink_conv1 = BaseConv(in_channels[2], in_channels[1], 1, 1, norm=norm, act=act)
		self.shrink_conv2 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
		self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
		
		self.p5_p4 = CSPLayer(2 * in_channels[1], in_channels[1], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)
		self.p4_p3 = CSPLayer(2 * in_channels[0], in_channels[0], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)

		# bottom-up conv
		self.downsample_conv1 = BaseConv(int(in_channels[0]), int(in_channels[0]), 3, 2, norm=norm, act=act)
		self.downsample_conv2 = BaseConv(int(in_channels[1]), int(in_channels[1]), 3, 2, norm=norm, act=act)
		self.n3_n4 = CSPLayer(2 * in_channels[0], in_channels[1], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)
		self.n4_n5 = CSPLayer(2 * in_channels[1], in_channels[2], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)

	def forward(self, inputs):
		#  backbone
		[c3, c4, c5] = inputs
		# top-down
		p5 = c5
		p5_expand = self.shrink_conv1(p5)
		p5_upsample = self.upsample(p5_expand)
		p4 = torch.cat([p5_upsample, c4], 1)
		p4 = self.p5_p4(p4)

		p4_expand = self.shrink_conv2(p4)
		p4_upsample = self.upsample(p4_expand)
		p3 = torch.cat([p4_upsample, c3], 1)
		p3 = self.p4_p3(p3)

		# down-top
		n3 = p3
		n3_downsample = self.downsample_conv1(n3)
		n4 = torch.cat([n3_downsample, p4_expand], 1)
		n4 = self.n3_n4(n4)

		n4_downsample = self.downsample_conv2(n4)
		n5 = torch.cat([n4_downsample, p5_expand], 1)
		n5 = self.n4_n5(n5)

		outputs = (n3, n4, n5)
		return outputs

class PA_FPN_AL(nn.Module):
	"""
	Only proceed 3 layer input. Like stage2, stage3, stage4.
	"""
	def __init__(self, depths=(1, 1, 1, 1), in_channels=(256, 512, 1024), norm='bn', act="silu", ):
		super().__init__()
		self.shrink_conv1 = BaseConv(in_channels[2], in_channels[1], 1, 1, norm=norm, act=act)
		self.shrink_conv2 = BaseConv(in_channels[2], in_channels[1], 1, 1, norm=norm, act=act)
		self.shrink_conv3 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
		self.shrink_conv4 = BaseConv(in_channels[1], in_channels[0], 1, 1, norm=norm, act=act)
		self.upsample = nn.Upsample(scale_factor=2, mode="bicubic")
		
		self.p5_p4 = CSPLayer(in_channels[1], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)
		self.p4_p3 = CSPLayer(in_channels[0], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)

		# bottom-up conv
		self.downsample_conv1 = BaseConv(int(in_channels[0]), int(in_channels[0]), 3, 2, norm=norm, act=act)
		self.downsample_conv2 = BaseConv(int(in_channels[1]), int(in_channels[1]), 3, 2, norm=norm, act=act)

		self.n3_n4 = CSPLayer(in_channels[1], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)
		self.n4_n5 = CSPLayer( in_channels[2], num_bottle=depths[0], shortcut=False, norm=norm, act=act,)

	def forward(self, inputs):
		#  backbone
		[c3, c4, c5] = inputs
		# top-down
		p5 = c5
		p5_expand = self.shrink_conv1(p5)
		p5_upsample = self.upsample(p5_expand)
		p4 = torch.cat([p5_upsample, c4], 1)
		p4 = self.shrink_conv2(p4)
		p4 = self.p5_p4(p4)

		p4_expand = self.shrink_conv3(p4)
		p4_upsample = self.upsample(p4_expand)
		p3 = torch.cat([p4_upsample, c3], 1)
		p3 = self.shrink_conv4(p3)
		p3 = self.p4_p3(p3)

		# down-top
		n3 = p3
		n3_downsample = self.downsample_conv1(n3)
		n4 = torch.cat([n3_downsample, p4_expand], 1)
		n4 = self.n3_n4(n4)

		n4_downsample = self.downsample_conv2(n4)
		n5 = torch.cat([n4_downsample, p5_expand], 1)
		n5 = self.n4_n5(n5)

		outputs = (n3, n4, n5)
		return outputs

## HEADs ##
class SimpleHead(nn.Module):
	def __init__(self, num_classes=3, n_anchors=3, in_channels=[256, 512, 1024],):
		super().__init__()
		self.n_anchors = n_anchors
		self.num_classes = num_classes
		self.head = nn.ModuleList()
		# For each feature map we go through different convolution.
		for i in range(len(in_channels)):
			self.head.append(nn.Conv2d(in_channels[i], n_anchors * (5 + num_classes), 1))

	def forward(self, inputs):
		outputs = []
		for k, (head_conv, x) in enumerate(zip(self.head, inputs)):
			# x: [batch_size, n_ch, h, w]
			output = head_conv(x) #head_conv[k](x)
			outputs.append(output)
		return outputs

class DecoupledHead(nn.Module):
	def __init__(self, num_classes=3, n_anchors=3, in_channels=[256, 512, 1024], norm='bn', act="silu",):
		super().__init__()
		self.n_anchors = n_anchors
		self.num_classes = num_classes
		ch = self.n_anchors * self.num_classes
		conv = BaseConv
		self.stems = nn.ModuleList()
		self.cls_convs = nn.ModuleList()
		self.cls_preds = nn.ModuleList()
		self.reg_convs = nn.ModuleList()
		self.reg_preds = nn.ModuleList()
		self.obj_preds = nn.ModuleList()

		# For each feature map we go through different convolution.
		for i in range(len(in_channels)):
			self.stems.append(BaseConv(in_channels[i], in_channels[0], ksize=1, stride=1, act=act))
			self.cls_convs.append(
				nn.Sequential(
					*[
						conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
						conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
					]
				)
			)
			self.cls_preds.append(
				nn.Conv2d(in_channels[0], ch, kernel_size=(1, 1), stride=(1, 1), padding=0)
			)
			self.reg_convs.append(
				nn.Sequential(
					*[
						conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
						conv(in_channels[0], in_channels[0], ksize=3, stride=1, norm=norm, act=act),
					]
				)
			)
			self.reg_preds.append(
				nn.Conv2d(in_channels[0], self.n_anchors * 4, kernel_size=(1, 1), stride=(1, 1), padding=0)
			)
			self.obj_preds.append(
				nn.Conv2d(in_channels[0], self.n_anchors * 1, kernel_size=(1, 1), stride=(1, 1), padding=0)
			)
		self.initialize_biases(1e-2)

	def initialize_biases(self, prior_prob):
		for conv in self.cls_preds:
			b = conv.bias.view(self.n_anchors, -1)
			b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
			conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

		for conv in self.obj_preds:
			b = conv.bias.view(self.n_anchors, -1)
			b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
			conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

	def forward(self, inputs):
		outputs = []
		for k, (cls_conv, reg_conv, x) in enumerate(zip(self.cls_convs, self.reg_convs, inputs)):
			# Change all inputs to the same channel.
			x = self.stems[k](x)
			print(k)
			cls_x = x
			reg_x = x
			print(x.shape)

			cls_feat = cls_conv(cls_x)
			print(cls_feat.shape)
			cls_output = self.cls_preds[k](cls_feat)
			print(cls_output.shape)
			print("------")
			reg_feat = reg_conv(reg_x)
			print(reg_feat.shape)
			reg_output = self.reg_preds[k](reg_feat)
			obj_output = self.obj_preds[k](reg_feat)
			print(reg_output.shape)
			print(obj_output.shape)

			# output: [batch_size, n_ch, h, w]
			output = torch.cat([reg_output, obj_output, cls_output], 1)
			print(output.shape)
			outputs.append(output)
		return outputs