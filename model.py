#!/usr/bin/env python
# coding: utf-8

# In[91]:


import os
import torch
import numpy as np
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
import sys


# In[73]:


plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']


# In[90]:


class TrashClassifier:
    
    def __init__(self):
        self.CLASS_NAMES = ['banana', 'battery', 'cake', 'calendar', 'fan', 
                            'glass', 'medicine', 'metal', 'nailcolor', 'napkin', 
                            'plastic', 'quilt', 'rice', 'teamilk', 'tube']
        self.NAME_MAP = {'banana': '香蕉皮', 'battery': '电池', 'cake': '蛋糕', 'calendar': '台历', 'fan': '风扇', 'glass': '玻璃',
                        'medicine': '弃置药品', 'metal': '金属', 'nailcolor': '指甲油', 'napkin': '纸巾', 'plastic': '塑料', 'quilt': '被子', 
                        'rice': '残渣剩饭', 'teamilk': '奶茶杯', 'tube': '灯管'}
        self.TRASH_MAP = {'banana': '易腐垃圾', 'battery': '有害垃圾', 'cake': '易腐垃圾', 'calendar': '可回收垃圾', 'fan': '可回收垃圾', 
                         'glass': '可回收垃圾', 'medicine': '有害垃圾', 'metal': '可回收垃圾', 'nailcolor': '有害垃圾', 'napkin': '其他垃圾', 
                         'plastic': '可回收垃圾', 'quilt': '可回收垃圾', 'rice': '易腐垃圾', 'teamilk': '其他垃圾', 'tube': '有害垃圾'}
        self.COLOR_MAP = {'可回收垃圾': 'b', '有害垃圾': 'r', '易腐垃圾': 'g', '其他垃圾': 'gray'}
        
        self.model_ft = torchvision.models.mobilenet_v2(pretrained=False)
        self.model_ft.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.model_ft.last_channel, 15),)
        self.model_ft.load_state_dict(torch.load('trash_classifier_model.pth'))
    
    def transform(self, image):
        image_transform = transforms.Compose([
                        transforms.Resize(256), 
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
        image = image_transform(image)
        return image
    
    def predict(self, image):
        image = self.transform(image)
        image = image.view(1, image.size()[0], image.size()[1], image.size()[2])
        output = self.model_ft(image)
        _, pred = torch.max(output, 1)
        return (self.NAME_MAP[self.CLASS_NAMES[pred.item()]], self.TRASH_MAP[self.CLASS_NAMES[pred.item()]])


# In[89]:


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('请输入图片路径！')
    else:
        image_path = sys.argv[1]
        image = Image.open(image_path)
        clf = TrashClassifier()
        ans = clf.predict(image)
        print('识别结果：{0}-{1}'.format(ans[0], ans[1]))
        plt.figure(dpi=150)
        plt.title(ans[0] + '-' + ans[1], c=clf.COLOR_MAP[ans[1]])
        plt.axis('off')
        plt.imshow(image)
        plt.show()


# In[ ]:




