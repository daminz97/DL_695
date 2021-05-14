import argparse
import torch
import os
import numpy as np
import glob
import random
import torchvision.transforms as tvt

from PIL import Image
from torch.utils.data import DataLoader, Dataset


parser = argparse.ArgumentParser(description='HW02 Task2')
parser.add_argument('--imagenet_root', type=str, required=True)
parser.add_argument('--class_list', nargs='*', type=str, required=True)
args, args_other = parser.parse_known_args()

# subtask 1


class ImageDataloader(Dataset):
    def __init__(self, data_path, transforms=None):
        # use arguments from argparse initialize program-defined variables
        # e.g. image path lists for cat and dog classes
        # you could also maintain label_array
        # 0 -- cat, 1 -- dog index
        # initialize the required transform
        # self.image_classes = [args.imagenet_root+class_path for class_path in args.class_list]
        # self.image_paths = [glob.glob(class_path, recursive=True) for class_path in image_classes]

        # self.folder_path = [os.path.join(data_path, img_path+'/*') for img_path in args.class_list]
        self.img_path = []
        self.labels = []
        for path in args.class_list:
            folder_path = os.path.join(data_path, path+'/*')
            for name in glob.glob(folder_path):
                self.img_path.append(name)
                if path == 'cat':
                    self.labels.append([1, 0])
                else:
                    self.labels.append([0, 1])
        self.transforms = transforms

    def __len__(self):
        # return the total number of images
        # refer pytorch documentation for more details
        return len(self.img_path)

    def __getitem__(self, index):
        # load color image(s), apply necessary data conversion and transformation
        # e.g. if an image is loaded in H*W*C format
        # rearrange it in C*H*W format, normalize values from 0-255 to 0-1
        # and apply the necessary transformation

        # convert the corresponding label in 1-hot encoding
        # return the processed image and labels in 1-hot encoded format

        img = Image.open(self.img_path[index])
        if self.transforms is not None:
            img_norm = self.transforms(img)
        label = torch.from_numpy(np.asarray(
            self.labels[index]))

        return img_norm, label


transform = tvt.Compose(
    [tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_dataset = ImageDataloader(os.path.join(
    args.imagenet_root, 'Train'), transform)
train_data_loader = torch.utils.data.DataLoader(
    dataset=train_dataset, batch_size=10, shuffle=True, num_workers=0)
val_dataset = ImageDataloader(os.path.join(
    args.imagenet_root, 'Val'), transform)
val_data_loader = torch.utils.data.DataLoader(
    dataset=val_dataset, batch_size=10, shuffle=True, num_workers=0)


# subtask 2
seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)

f = open("output.txt", "a")

dtype = torch.float64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

epochs = 40
D_in, H1, H2, D_out = 3*64*64, 1000, 256, 2
w1 = torch.randn(D_in, H1, device=device, dtype=dtype)
w2 = torch.randn(H1, H2, device=device, dtype=dtype)
w3 = torch.randn(H2, D_out, device=device, dtype=dtype)
learning_rate = 1e-9

for t in range(epochs):
    epoch_loss = 0.0
    for i, data in enumerate(train_data_loader):
        inputs, labels = data
        inputs = inputs.to(device, dtype=dtype)
        labels = labels.to(device, dtype=dtype)
        x = inputs.view(inputs.size(0), -1)
        # in numpy, h1 - x.dot(w1)
        h1 = x.mm(w1)
        h1_relu = h1.clamp(min=0)
        h2 = h1_relu.mm(w2)
        h2_relu = h2.clamp(min=0)
        y_pred = h2_relu.mm(w3)
        # Compute and print loss
        loss = (y_pred - labels).pow(2).sum().item()
        epoch_loss += loss
        y_error = y_pred - labels

        # TODO: accumulate loss for printing per epoch
        # gradient of loss w.r.t w3
        grad_w3 = h2_relu.t().mm(2 * y_error)
        # backpropagated error to the h2 hidden layer
        h2_error = 2.0 * y_error.mm(w3.t())
        # we set those elements of the backpropagated error
        h2_error[h2 < 0] = 0
        # gradient of loss w.r.t w2
        grad_w2 = h1_relu.t().mm(2 * h2_error)
        # backpropagated error to the h1 hidden layer
        h1_error = 2.0 * h2_error.mm(w2.t())
        # we set those elements of the backpropagated error
        h1_error[h1 < 0] = 0
        # gradient of loss w.r.t w2
        grad_w1 = x.t().mm(2 * h1_error)

        # update weights using gradient descent
        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2
        w3 -= learning_rate * grad_w3

    # print loss per epoch
    print('Epoch %d:\t %0.4f' % (t, epoch_loss), file=f)

print(file=f)
# store layer weights in pickle file format
torch.save({'w1': w1, 'w2': w2, 'w3': w3}, './wts.pkl')



# val set
para = torch.load('./wts.pkl')
w1 = para['w1']
w2 = para['w2']
w3 = para['w3']
correct = 0
total_guess = 0
total_loss = 0.0
for i, data in enumerate(val_data_loader):
    inputs, labels = data
    inputs = inputs.to(device, dtype=dtype)
    labels = labels.to(device, dtype=dtype)
    x = inputs.view(inputs.size(0), -1)
    h1 = x.mm(w1)
    h1_relu = h1.clamp(min=0)
    h2 = h1_relu.mm(w2)
    h2_relu = h2.clamp(min=0)
    y_pred = h2_relu.mm(w3)

    loss = (y_pred - labels).pow(2).sum().item()
    total_loss += loss

    predictions = torch.argmax(y_pred, dim=1)
    truth = torch.argmax(labels, dim=1)
    for j in range(len(predictions)):
        if predictions[j] == truth[j]:
            correct += 1
        total_guess += 1
print('Val Loss:\t %0.4f' % (total_loss), file=f)
print('Val Accuracy:\t %0.4f' % (correct*100/total_guess) + "%", file=f)
f.close()