#!/usr/bin/env python
# coding: utf-8

import os
import torch
import numpy as np
from torchvision import transforms, datasets
import torchvision
import time
import copy

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = 'datasets'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, 
                                             num_workers=4) for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device('cuda')

model_ft = torchvision.models.mobilenet_v2(pretrained=True)
for param in model_ft.parameters():
    param.requires_grad = False
model_ft.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.2),
    torch.nn.Linear(model_ft.last_channel, 15),)

model_ft.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer_ft = torch.optim.SGD(model_ft.classifier.parameters(), lr=0.1, momentum=0.9)
exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer_ft, step_size=30, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epoches=100):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epoches):
        print('Epoch {0}/{1}'.format(epoch, num_epoches - 1))
        print('-' * 10)
        
        running_loss = 0.0
        running_corrects = 0
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
        
        epoch_loss = running_loss / dataset_sizes['train']
        epoch_acc = running_corrects / dataset_sizes['train']
        
        print('Training Loss: {0:.4f} Acc: {1:.4f}'.format(epoch_loss, epoch_acc))
        
        running_loss = 0.0
        running_corrects = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
        
        epoch_loss = running_loss / dataset_sizes['val']
        epoch_acc = running_corrects / dataset_sizes['val']
        
        print('Test Loss: {0:.4f} Acc: {1:.4f}'.format(epoch_loss, epoch_acc))
        
        if(epoch_acc > best_acc):
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        print()
    
    time_elapsed = time.time() - since
    print('Training complete in {0:.0f} min {1:.0f} seconds.'.format(time_elapsed // 60, time_elapsed % 60))
    print(best_acc)
    
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler)
torch.save(model_ft.state_dict(), 'trash_classifier_model.pth')