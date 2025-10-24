import torch as t
from torch import nn
# from RGBT_dataprocessing_CNet import testData1,testData2,testData3
from train_test1.RGBT_dataprocessing_CNet import testData1
from torch.utils.data import DataLoader
import os
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch

from ABiUNet.network.ABiU_Net_b4 import VisionTransformer as ABiU_Net

import torch.nn.functional as F
import cv2
import torchvision
from SwinNet_main.models.Swin_Transformer import SwinNet
# from MSEDNet.net import Net


import numpy as np
from datetime import datetime

test_dataloader1 = DataLoader(testData1, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader2 = DataLoader(testData2, batch_size=1, shuffle=False, num_workers=4)
# test_dataloader3 = DataLoader(testData3, batch_size=1, shuffle=False, num_workers=4)
net = ABiU_Net()

net.load_state_dict(t.load('/home/yuride/Desktop/LYQ/model/train_test1/Pth/ABiU_Net_2024_10_18_21_15_best.pth'))   ######gaiyixia

a = '/home/yuride/Desktop/LYQ/model/RGBT-EvaluationTools/SalMap/'
b = 'ABiU_Net'
c = '/rail_362/'
d = '/VT1000/'
e = '/VT5000/'

aa = []

vt800 = a + b + c
vt1000 = a + b + d
vt5000 = a + b + e


path1 = vt800
isExist = os.path.exists(vt800)
if not isExist:
	os.makedirs(vt800)
else:
	print('path1 exist')

with torch.no_grad():
	net.eval()
	net.cuda()
	test_mae = 0

	for i, sample in enumerate(test_dataloader1):
		image = sample['RGB']
		depth = sample['depth']
		label = sample['label']
		name = sample['name']
		name = "".join(name)

		image = Variable(image).cuda()
		depth = Variable(depth).cuda()
		label = Variable(label).cuda()

		out1 = net(image)
		# out1 = net(image, depth)
		out1 = torch.sigmoid(out1[0])#比较结果的
		out = out1

		out_img = out.cpu().detach().numpy()
		out_img = out_img.squeeze()

		plt.imsave(path1 + name + '.png', arr=out_img, cmap='gray')
		print(path1 + name + '.png')

	# 	out = r2
	# 	print(out.shape)
	# 	out = F.interpolate(out, size=(320, 320), mode='bilinear', align_corners=False)
	# 	out_img = out.cpu().detach().numpy()
	# 	out_img = np.max(out_img, axis=1).reshape(320, 320)
	# 	out_img = (((out_img - np.min(out_img))/(np.max(out_img) - np.min(out_img)))*255).astype(np.uint8)
	# 	out_img = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
	# 	cv2.imwrite(path1 + name + '.png', out_img)
	# print("Have done!")




##########################################################################################


# path2 = vt1000
# isExist = os.path.exists(vt1000)
# if not isExist:
# 	os.makedirs(vt1000)
# else:
# 	print('path2 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader2):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5 = net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
# 		plt.imsave(path2 + name + '.png', arr=out_img, cmap='gray')
# 		print(path2 + name + '.png')
# 	cur_time = datetime.now()




#######################################################################################################
#
# path3 = vt5000
# isExist = os.path.exists(vt5000)
# if not isExist:
# 	os.makedirs(vt5000)
# else:
# 	print('path3 exist')
#
# with torch.no_grad():
# 	net.eval()
# 	net.cuda()
# 	test_mae = 0
# 	prec_time = datetime.now()
# 	for i, sample in enumerate(test_dataloader3):
# 		image = sample['RGB']
# 		depth = sample['depth']
# 		label = sample['label']
# 		name = sample['name']
# 		name = "".join(name)
#
# 		image = Variable(image).cuda()
# 		depth = Variable(depth).cuda()
# 		label = Variable(label).cuda()
#
#
# 		# out1,out2,out3,out4,out5= net(image, depth)
# 		# out1, out2 = net(image, depth)
# 		out1, out2, out3, out4, out5, out6, out7, out8 = net(image, depth)
# 		out = torch.sigmoid(out1)
#
#
#
#
#
# 		out_img = out.cpu().detach().numpy()
# 		out_img = out_img.squeeze()
#
#
# 		plt.imsave(path3 + name + '.png', arr=out_img, cmap='gray')
# 		print(path3 + name + '.png')
#
# 	cur_time = datetime.now()
#   TIANCAIDAOCIYIYOU








