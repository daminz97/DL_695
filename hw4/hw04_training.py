import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
from PIL import ImageFilter
import argparse
import random
import matplotlib.pyplot as plt
import gzip
import pickle
from model import Net1, Net2, Net3
from dataloader import ImageDataloader


parser = argparse.ArgumentParser(description='HW04 Training/Validation')
parser.add_argument('--root_path', required=True, type=str)
parser.add_argument('--class_list', required=True, nargs='*', type=str)
args, args_other = parser.parse_known_args()

seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

class TrainingInstance(object):
    def load_data(self):
        transform = tvt.Compose(
            [tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        train_dataset = ImageDataloader(args.root_path, args.class_list, transform)
        training_data_loader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=10, shuffle=True, num_workers=12)
        return training_data_loader

    def run_code_for_training(self, net):
        net = net.to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)
        epochs = 10
        loss_record = []
        for epoch in range(epochs):
            running_loss = 0.0
            for i, data in enumerate(training_data_loader):
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if (i+1) % 500 == 0:
                    print("\n[epoch:%d, batch:%5d] loss: %.3f" % 
                        (epoch+1, i+1, running_loss/float(500)))
                    loss_record.append(running_loss/float(500))
                    running_loss = 0.0
        return loss_record
    
    def get_net1(self):
        return Net1()
    
    def get_net2(self):
        return Net2()
    
    def get_net3(self):
        return Net3()
    
    def plot_figure(self, loss1, loss2, loss3):
        plt.figure()
        plt.title('Training Loss')
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.plot(loss1, label='Net1 Training Loss')
        plt.plot(loss2, label='Net2 Training Loss')
        plt.plot(loss3, label='Net3 Training Loss')
        plt.legend(loc='upper right')
        plt.savefig("train_loss.jpg")
        plt.show()
    
    
if __name__ == '__main__':
    instance = TrainingInstance()
    training_data_loader = instance.load_data()

    model1 = instance.get_net1()
    net1_loss = instance.run_code_for_training(model1)
    model_save_name = 'net1.pth'
    path = "{model_save_name}" 
    torch.save(model1.state_dict(), path)
    
    model2 = instance.get_net2()
    net2_loss = instance.run_code_for_training(model2)
    model_save_name = 'net2.pth'
    path = "{model_save_name}" 
    torch.save(model2.state_dict(), path)
    
    model3 = instance.get_net3()
    net3_loss = instance.run_code_for_training(model3)
    model_save_name = 'net3.pth'
    path = "{model_save_name}" 
    torch.save(model3.state_dict(), path)
    
    instance.plot_figure(net1_loss, net2_loss, net3_loss)