# -*- coding: utf-8 -*-
"""
Created on Sun May 30 22:55:12 2021

@author: ygmr
"""
import torchvision.models.segmentation as seg
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import transforms as t
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class EthicsCaseNet(torch.nn.Module):
	def __init__(self):
		super(EthicsCaseNet, self).__init__()
		self.conv1 = torch.nn.Conv2d(3,64,3)
		self.conv2 = torch.nn.Conv2d(64,32,3)
		self.conv3 = torch.nn.Conv2d(32,32,3)
		self.pool = torch.nn.MaxPool2d(2)
		self.fc1 = torch.nn.Linear(32*30*55+1, 512)
		self.fc2 = torch.nn.Linear(512, 256)
		self.fc3 = torch.nn.Linear(256, 2)
		self.dropout = torch.nn.Dropout(0.2)
		
	def forward(self, x1,x2):
		x1 = self.pool(F.relu(self.conv1(x1)))
		x1 = self.pool(F.relu(self.conv2(x1)))
		x1 = self.pool(F.relu(self.conv3(x1)))
		#print(x1.shape)
		x1 = x1.view(-1,32*30*55)
		x2 = torch.unsqueeze(x2, dim=1)
		#print(x1.shape, x2.shape)
		x = torch.cat((x1, x2), dim=1)
		#print(x.shape)
		x = self.dropout(F.relu(self.fc1(x)))
		x = self.dropout(F.relu(self.fc2(x)))
		x = F.relu(self.fc3(x))
		
		return x
	
class EthicsEngine:
	def __init__(self, seg_pth, case_pth):
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.seg_model = seg.deeplabv3_resnet101(num_classes=6)
		self.seg_model.classifier = DeepLabHead(2048, 6)
		self.seg_model.load_state_dict(torch.load(seg_pth)["model"])
		self.seg_model.eval()
		self.case_model = EthicsCaseNet()
		self.case_model.load_state_dict(torch.load(case_pth))
		self.case_model.eval()
		self.seg_model.to(self.device)
		self.case_model.to(self.device)
		
	def process_image(self, image):
		#print(image.shape)
		shp = image.shape
		
		transform = t.Compose([t.Resize((256,128)), t.ToTensor()])
		pil = t.ToPILImage()
		image = pil(image)
		image = transform(image)
		image = image.reshape((1,3,256,128))
		return image
	
	def process_speed(self, speed):
		if speed < 10:
			speed = 0
		elif speed <20:
			speed = 1
		elif speed <30:
			speed = 2
		elif speed <40:
			speed = 3
		else:
			speed = 4
		print(speed)
		speed = np.array([speed])
		speed = torch.as_tensor(speed, dtype=torch.uint8)
		return speed
	
	def process_segmented_output(self, out):
		dims = out.shape
		#out = out.resize((dims[0], dims[1], 128))
		out = out.reshape((1,3,256,455))
		return out
		
	def process_segmentation(self, inp):
		colors = [(0,255,255), (0,255,0), (128,255,0), (255,255,0), (255,128,0), (255,0,0)]
		inp = self.process_image(inp)
		inp = inp.to(self.device)
		print(inp.shape)
		out = self.seg_model(inp)
		op = out["out"].detach().cpu().numpy()
		cop=op.argmax(1)
		cop = np.squeeze(cop).astype(np.uint8)
		
		mask = np.empty((720, 1280, 3))
		r = Image.fromarray(cop).resize((1280,720),resample=Image.NEAREST)
	
		for i in range(720):
			for j in range(1280):
				px = r.getpixel((j,i))
				mask[i,j,0] = colors[px][0]
				mask[i,j,1] = colors[px][1]
				mask[i,j,2] = colors[px][2]
		mask = mask.astype(np.uint8)
		transform = t.Compose([t.Resize((256,455)), t.ToTensor()])
		m = Image.fromarray(mask)
		pil = t.ToPILImage()
		#image = pil(m)
		image = transform(m)
		return image,mask
	
	def process_ethical_decision(self, image, speed):
		classes = ["Ethical", "Not Ethical"]
		image = image.to(self.device)
		speed = speed.to(self.device)
		output = self.case_model(image, speed)
		_, pred = torch.max(output, 1)
		print(pred)
		res = pred.detach().cpu().numpy()[0]
		print(classes[pred])
		