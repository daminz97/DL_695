import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
from torchsummary import summary           
import numpy as np
import pandas as pd
from PIL import ImageFilter
import argparse
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
from sklearn.metrics import confusion_matrix
import seaborn as sns
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

device = torch.device('cpu')


class ValidationExample(object):
    def __init__(self):
      self.label_dict = {0:'airplane',
                  1:'boat',
                  2:'cat',
                  3:'dog',
                  4:'elephant',
                  5:'giraffe',
                  6:'horse',
                  7:'refrigerator',
                  8:'train',
                  9:'truck'}

    def load_data(self):
        transform = tvt.Compose(
            [tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        val_dataset = ImageDataloader(args.root_path, args.class_list, transform)
        val_data_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=10, shuffle=True, num_workers=12)
        return val_data_loader
    
    def get_net1(self):
        net = Net1()
        net.load_state_dict(torch.load('net1.pth'))
        net.eval()
        return net
    
    def get_net2(self):
        net = Net2()
        net.load_state_dict(torch.load('net2.pth'))
        net.eval()
        return net
    
    def get_net3(self):
        net = Net3()
        net.load_state_dict(torch.load('net3.pth'))
        net.eval()
        return net
    
    def run_code_for_val(self, net, val_data_loader):
        net = net.to(device)
        pred_labels = []
        true_labels = []
        with torch.no_grad():
          for i, data in enumerate(val_data_loader):
              inputs, labels = data
              inputs = inputs.to(device)
              labels = labels.to(device)
              outputs = net(inputs)
              _, preds = torch.max(outputs.data, 1)
              pred_labels.extend(preds.numpy())
              true_labels.extend(labels.numpy())
        return pred_labels, true_labels
    
    def plot_matrix(self, pred, true):
        acc = np.sum(np.array(pred)==np.array(true)) / len(pred)
        arr = confusion_matrix(true, pred)
        df_cm = pd.DataFrame(arr, index = [v for k, v in self.label_dict.items()],
                  columns = [v for k, v in self.label_dict.items()])
        ax = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.set_title('Accuracy = %.3f' % acc)
        return ax
    
if __name__ == '__main__':
    instance = ValidationExample()
    val_data_loader = instance.load_data()
    
    model1 = instance.get_net1()
    model1_pred, model1_true = instance.run_code_for_val(model1, val_data_loader)
    figure1 = instance.plot_matrix(model1_pred, model1_true)
    figure1.figure.savefig('net1_confusion_matrix.jpg')
    plt.close(figure1.figure)
    
    model2 = instance.get_net2()
    model2_pred, model2_true = instance.run_code_for_val(model2, val_data_loader)
    figure2 = instance.plot_matrix(model2_pred, model2_true)
    figure2.figure.savefig('net2_confusion_matrix.jpg')
    plt.close(figure2.figure)
    
    model3 = instance.get_net3()
    model3_pred, model3_true = instance.run_code_for_val(model3, val_data_loader)
    figure3 = instance.plot_matrix(model3_pred, model3_true)
    figure3.figure.savefig('net3_confusion_matrix.jpg')
    plt.close(figure3.figure)