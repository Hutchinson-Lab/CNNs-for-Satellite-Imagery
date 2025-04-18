from torchvision import models
import torch.nn as nn
import math


class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		return x


class ResNet(nn.Module):
	"""
	Returns a ResNet18 model with the specified number of residual blocks (layers).
	"""
	# https://discuss.pytorch.org/t/layers-are-not-initialized-with-same-weights-with-manual-seed/56804/2

	def __init__(self, num_classes, in_channels, num_blocks, pretrained):
		super(ResNet, self).__init__()
		if pretrained:
			self.resnet = models.resnet18(weights='DEFAULT')
			print('Using pre-trained weights')

			self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)

			# repeat conv1 weights RGBRGB... and scale 3/C per https://arxiv.org/abs/2306.09424
			w = self.resnet.conv1.state_dict()['weight']  # [64, 3, 7, 7]
			n = math.ceil(in_channels / 3)  # how many times to repeat
			w = w.repeat((1, n, 1, 1))
			w = w[:, 0:in_channels, :, :]  # [64, in_channels, 7, 7]
			w = w * 3 / in_channels

			self.resnet.conv1.load_state_dict({'weight': w})

		else:
			self.resnet = models.resnet18()
			print('Using random weights')

			self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=2, padding=3, bias=False)

		if num_blocks == 0:  # only first convolutional layer
			self.resnet.layer1 = Identity()
			self.resnet.layer2 = Identity()
			self.resnet.layer3 = Identity()
			self.resnet.layer4 = Identity()
			self.resnet.fc = Identity()
			self.fc = nn.Linear(in_features=64, out_features=num_classes, bias=True)

		if num_blocks == 1:  # first conv layer & layer 1
			self.resnet.layer2 = Identity()
			self.resnet.layer3 = Identity()
			self.resnet.layer4 = Identity()
			self.resnet.fc = Identity()
			self.fc = nn.Linear(in_features=64, out_features=num_classes, bias=True)

		if num_blocks == 2:  # first conv layer & layers 1-2
			self.resnet.layer3 = Identity()
			self.resnet.layer4 = Identity()
			self.resnet.fc = Identity()
			self.fc = nn.Linear(in_features=128, out_features=num_classes, bias=True)

		if num_blocks == 3:  # first conv layer & layers 1-3
			self.resnet.layer4 = Identity()
			self.resnet.fc = Identity()
			self.fc = nn.Linear(in_features=256, out_features=num_classes, bias=True)

		if num_blocks == 4:  # first conv layer & layers 1-4
			self.resnet.fc = Identity()
			self.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)

	def forward(self, x):
		x = self.resnet(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)
		return x