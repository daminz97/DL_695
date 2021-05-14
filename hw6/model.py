import os, sys
import numpy as np
import random
import math
import copy
import gzip
import pickle
from skimage import io
import cv2
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tvt
import torch.optim as optim
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from dataloader import CocoLoader
from pycocotools.coco import COCO


seed = 0
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmarks = False
os.environ['PYTHONHASHSEED'] = str(seed)


class YoloNaive(nn.Module):
    def __init__(self, json_path, root_train, root_test, img_size, interval, path_saved_model, momentum, learning_rate, epochs, batch_size, class_list, use_gpu=True):
        super(YoloNaive, self).__init__()
        self.json_path = json_path
        self.root_train = root_train
        self.root_test = root_test
        self.img_size = img_size
        self.interval = interval
        self.path_saved_model = path_saved_model
        self.momentum = momentum
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_list = class_list
        self.use_gpu = use_gpu
        self.train_dataloader = None
        self.test_dataloader = None
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')

    def set_dataloaders(self, train=False, test=False):
        if train:
            train_dataset = CocoLoader(dataset_size=1000,
                                        train_or_test='train',
                                       json_path=self.json_path,
                                       class_list=self.class_list,
                                       root_train=self.root_train,
                                       root_test=None,
                                       transforms=None)
            print('\nCreating train dataloader...')
            self.train_dataloader = torch.utils.data.DataLoader(
                train_dataset, self.batch_size, shuffle=True, num_workers=2)
        if test:
            test_dataset = CocoLoader(dataset_size=300,
                                      train_or_test='test',
                                      json_path=self.json_path,
                                      class_list=self.class_list,
                                      root_train=None,
                                      root_test=self.root_test,
                                      transforms=None)
            print('\nCreating test dataloader...')
            self.test_dataloader = torch.utils.data.DataLoader(
                test_dataset, self.batch_size, shuffle=False, num_workers=2)

    class SkipBlock(nn.Module):
        def __init__(self, in_ch, out_ch, downsample=False, skip_connections=True):
            super(YoloNaive.SkipBlock, self).__init__()
            self.in_ch = in_ch
            self.out_ch = out_ch
            self.downsample = downsample
            self.skip_connections = skip_connections
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1)
            norm1 = nn.BatchNorm2d
            norm2 = nn.BatchNorm2d
            self.bn1 = norm1(out_ch)
            self.bn2 = norm2(out_ch)
            if downsample:
                self.downsampler = nn.Conv2d(in_ch, out_ch, 1, stride=2)

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = torch.nn.functional.relu(out)
            if self.in_ch == self.out_ch:
                out = self.conv2(out)
                out = self.bn2(out)
                out = torch.nn.functional.relu(out)
            if self.downsample:
                out = self.downsampler(out)
                identity = self.downsampler(identity)
            if self.skip_connections:
                if self.in_ch == self.out_ch:
                    out += identity
                else:
                    out[:, :self.in_ch, :, :] += identity
                    out[:, self.in_ch:, :, :] += identity
            return out

    class Net(nn.Module):
        def __init__(self, skip_connections=True, depth=8):
            super(YoloNaive.Net, self).__init__()
            if depth not in [8, 10, 12, 14, 16]:
                sys.exit(
                    "Net has only been tested for 'depth' values 8, 10, 12, 14, and 16")
            self.depth = depth // 2
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
            self.pool = nn.MaxPool2d(2, 2)
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(256)
            self.skip64_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip64_arr.append(
                    YoloNaive.SkipBlock(64, 64, skip_connections=skip_connections))
            self.skip64ds = YoloNaive.SkipBlock(
                64, 64, downsample=True, skip_connections=skip_connections)
            self.skip64to128 = YoloNaive.SkipBlock(
                64, 128, skip_connections=skip_connections)
            self.skip128_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip128_arr.append(
                    YoloNaive.SkipBlock(128, 128, skip_connections=skip_connections))
            self.skip128ds = YoloNaive.SkipBlock(
                128, 128, downsample=True, skip_connections=skip_connections)
            self.skip128to256 = YoloNaive.SkipBlock(
                128, 256, skip_connections=skip_connections)
            self.skip256_arr = nn.ModuleList()
            for i in range(self.depth):
                self.skip256_arr.append(YoloNaive.SkipBlock(
                    256, 256, skip_connections=skip_connections))
            self.skip256ds = YoloNaive.SkipBlock(
                256, 256, downsample=True, skip_connections=skip_connections)
            self.fc_seq = nn.Sequential(
                nn.Linear(4096, 2048),
                nn.Linear(2048, 1440)
            )

        def forward(self, x):
            x = self.pool(torch.nn.functional.relu(self.conv1(x)))
            x = nn.MaxPool2d(2, 2)(torch.nn.functional.relu(self.conv2(x)))
            for i, skip64 in enumerate(self.skip64_arr[:self.depth//4]):
                x = skip64(x)
            x = self.skip64ds(x)
            for i, skip64 in enumerate(self.skip64_arr[self.depth//4:]):
                x = skip64(x)
            x = self.bn1(x)
            x = self.skip64to128(x)
            for i, skip128 in enumerate(self.skip128_arr[:self.depth//4]):
                x = skip128(x)
            x = self.bn2(x)
            x = self.skip128ds(x)
            x = self.skip128to256(x)
            for i, skip256 in enumerate(self.skip256_arr[:self.depth//4]):
              x = skip256(x)
            x = self.skip256ds(x)
            x = x.view(-1, 4096)
            x = self.fc_seq(x)
            return x

    def training_part(self, net, display_img=False):
        yolo_debug = False
        filename_for_out1 = 'performance_numbers_'+str(self.epochs)+'label.txt'
        filename_for_out2 = 'performance_numbers_' +str(self.epochs)+'regres.txt'
        file1 = open(filename_for_out1, 'w')
        file2 = open(filename_for_out2, 'w')
        net = copy.deepcopy(net)
        net = net.to(self.device)
        criterion1 = nn.CrossEntropyLoss()
        criterion2 = nn.MSELoss()
        # criterion3 = YoloNaive.YoloLoss(self.batch_size)
        optimizer = optim.SGD(net.parameters(), lr=self.lr,
                              momentum=self.momentum)
        print('\nStart training...')
        loss_arr = []
        yolo_interval = self.interval
        num_yolo_cells = (self.img_size//yolo_interval) * \
            (self.img_size//yolo_interval)
        num_anchor_boxes = 5
        yolo_tensor = torch.zeros(
            self.batch_size, num_yolo_cells, num_anchor_boxes, 8)
        
        def IoU(self, inp, tar):
          t_x, t_y, t_w, t_h = inp[0], inp[1], inp[2], inp[3]
          p_x, p_y, p_w, p_h = tar[0], tar[1], tar[2], tar[3]
          min_x = min(t_x, p_x, t_x+t_w, p_x+p_w)
          min_y = min(t_y, p_y, t_y+t_h, p_y+p_h)
          max_x = max(t_x, p_x, t_x+t_w, p_x+p_w)
          max_y = max(t_y, p_y, t_y+t_h, p_y+p_h)
          composite_loss = []
          union = intersection = 0.0
          for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                if i >= t_x and i <= (t_x+t_w) and j>= t_y and j<=(t_y+t_h):
                  if i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                    intersection += 1
                    union += 1
                  elif i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                    union += 1
          if union == 0.0:
            raise Exception("\n\nSomething wrong")
          sample_iou = intersection / float(union)
          return 1 - torch.tensor([sample_iou])   

        class AnchorBox:
            def __init__(self, ratio, tlc, ab_h, ab_w, adx):
                self.ar = ratio
                self.tlc = tlc
                self.ab_h = ab_h
                self.ab_w = ab_w
                self.adx = adx

            def __str__(self):
                return 'AnchorBox ratio: %s tlc for yolo cell: %s anchor-box height: %s anchor-box width: %s adx: %d' % (self.ar, str(self.tlc), self.ab_h, self.ab_w, self.adx)
        
        for epoch in range(self.epochs):
            running_loss = 0.0
            for key, data in enumerate(self.train_dataloader):
                if key == 2000:
                  break
                imgId, im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
                im_tensor = im_tensor.to(self.device)
                bbox_tensor = bbox_tensor.to(self.device)
                bbox_label_tensor = bbox_label_tensor.to(self.device)
                yolo_tensor = yolo_tensor.to(self.device)
                cell_h = yolo_interval
                cell_w = yolo_interval
                obj_centers = {ibx: {idx: None for idx in range(num_objects_in_image[ibx])} for ibx in range(im_tensor.shape[0])}
                num_cells_image_height = self.img_size // yolo_interval
                num_cells_image_width = self.img_size // yolo_interval
                
                ab_1_1 = [[AnchorBox('1/1', (i*yolo_interval, j*yolo_interval), yolo_interval, yolo_interval, 0) for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
                ab_1_3 = [[AnchorBox('1/3', (i*yolo_interval, j*yolo_interval), yolo_interval, 3*yolo_interval, 1) for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
                ab_1_5 = [[AnchorBox('1/5', (i*yolo_interval, j*yolo_interval), yolo_interval, 5*yolo_interval, 2) for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
                ab_3_1 = [[AnchorBox('3/1', (i*yolo_interval, j*yolo_interval), 3*yolo_interval, yolo_interval, 3) for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
                ab_5_1 = [[AnchorBox('5/1', (i*yolo_interval, j*yolo_interval), 5*yolo_interval, yolo_interval, 4) for i in range(0, num_cells_image_height)] for j in range(0, num_cells_image_width)]
                
                for ibx in range(im_tensor.shape[0]):
                    for idx in range(num_objects_in_image[ibx]):
                        bb_center_h = (bbox_tensor[ibx][idx][1].item() + bbox_tensor[ibx][idx][3].item() // 2)
                        bb_center_w = (bbox_tensor[ibx][idx][0].item() + bbox_tensor[ibx][idx][2].item() // 2)
                        obj_bb_h = bbox_tensor[ibx][idx][3].item()
                        obj_bb_w = bbox_tensor[ibx][idx][2].item()
                        # label = self.class_list[bbox_label_tensor[ibx][idx].item()]
                        if (obj_bb_h < 4) or (obj_bb_w < 4):
                            continue
                        ar = float(obj_bb_h) / float(obj_bb_w)
                        
                        cell_row_idx = bb_center_h // yolo_interval
                        cell_col_idx = bb_center_w // yolo_interval
                        cell_row_idx = 5 if cell_row_idx > 5 else cell_row_idx
                        cell_col_idx = 5 if cell_col_idx > 5 else cell_col_idx
                        
                        if ar <= 0.2:
                            ab = ab_1_5[cell_row_idx][cell_col_idx]
                        elif ar <= 0.5:
                            ab = ab_1_3[cell_row_idx][cell_col_idx]
                        elif ar <= 1.5:
                            ab = ab_1_1[cell_row_idx][cell_col_idx]
                        elif ar <= 4:
                            ab = ab_3_1[cell_row_idx][cell_col_idx]
                        elif ar > 4:
                            ab = ab_5_1[cell_row_idx][cell_col_idx]
                        
                        bh = float(obj_bb_h) / float(yolo_interval)
                        bw = float(obj_bb_w) / float(yolo_interval)
                        obj_center_x = float(bbox_tensor[ibx][idx][0].item() + bbox_tensor[ibx][idx][2].item() / 2.0)
                        obj_center_y = float(bbox_tensor[ibx][idx][1].item() + bbox_tensor[ibx][idx][3].item() / 2.0)
                        
                        yolocell_center_i = cell_row_idx * yolo_interval + float(yolo_interval) / 2.0
                        yolocell_center_j = cell_col_idx * yolo_interval + float(yolo_interval) / 2.0
                        
                        del_x = float(obj_center_x - yolocell_center_i) / yolo_interval
                        del_y = float(obj_center_y - yolocell_center_j) / yolo_interval
                        iou = IoU([ab.tlc[0], ab.tlc[1], ab.ab_w, ab.ab_h], [bbox_tensor[ibx][idx][0].item(),bbox_tensor[ibx][idx][1].item(),bbox_tensor[ibx][idx][2].item(),bbox_tensor[ibx][idx][3].item()])
                        yolo_vector = [iou, del_x, del_y, bh, bw, 0, 0, 0]
                        if bbox_label_tensor[ibx][idx].item() != 13:
                          yolo_vector[5+bbox_label_tensor[ibx][idx].item()] = 1
                        # yolo_vector[5 + bbox_label_tensor[ibx][idx].item()] = 1
                        yolo_cell_index = cell_row_idx * num_cells_image_width + cell_col_idx
                        yolo_tensor[ibx, yolo_cell_index, ab.adx] = torch.FloatTensor(yolo_vector)
                
                yolo_tensor_flattened = yolo_tensor.view(im_tensor.shape[0], -1)
                optimizer.zero_grad()
                output = net(im_tensor)
                loss = criterion2(output, yolo_tensor_flattened)
                # loss = criterion3.forward(bbox_tensor, bbox_label_tensor, output.view(2, -1, 8))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                if key % 100 == 99:
                    avg_loss = running_loss / float(100)
                    print('\n\nepoch:%d/%d, iter:%4d, mean MSE loss:%7.4f\n' % (epoch+1, self.epochs, key+1, avg_loss))
                    loss_arr.append(running_loss)
                    file1.write('%.3f\n' % avg_loss)
                    file1.flush()
                    running_loss = 0.0
                    
                    if display_img:
                        predictions = output.view(4, 36, 5, 8)
                        for ibx in range(predictions.shape[0]):
                            icx_2_best_ab = {ic: None for ic in range(36)}
                            for icx in range(predictions.shape[1]):
                                cell_predi = predictions[ibx, icx]
                                prev_best = 0
                                for abdx in range(cell_predi.shape[0]):
                                    if cell_predi[abdx][0] > cell_predi[prev_best][0]:
                                        prev_best = abdx
                                best_ab_icx = prev_best
                                icx_2_best_ab[icx] = best_ab_icx
                            sorted_icx_to_box = sorted(icx_2_best_ab, key=lambda x: predictions[ibx, x, icx_2_best_ab[x]][0].item(), reverse=True)
                            retained_cells = sorted_icx_to_box[:5]
                            objects_detected = []
                            for icx in retained_cells:
                                prev_vec = predictions[ibx, icx, icx_2_best_ab[icx]]
                                class_labels_predi = prev_vec[-3:]
                                if torch.all(class_labels_predi < 0.2):
                                    predicted_class_label = None
                                else:
                                    best_predicted_class_index = (class_labels_predi == class_labels_predi.max()).nonzero().squeeze().item()
                                    predicted_class_label = self.class_list[best_predicted_class_index]
                                    objects_detected.append(predicted_class_label)
                            print('[batch image=%d] objects found in descending probability order: ' % ibx, objects_detected)
                    if display_img:
                        logger = logging.getLogger()
                        old_level = logger.level
                        logger.setLevel(100)
                        plt.figure(figsize=[15,4])
                        plt.imshow(np.transpose(torchvision.utils.make_grid(im_tensor, normalize=True, padding=3, pad_value=255).cpu(), (1,2,0)))
                        plt.show()
                        logger.setLevel(old_level)
        print('\nFinished training.')
        plt.figure(figsize=(10,5))
        plt.title('Loss vs. Iterations')
        plt.plot(loss_arr)
        plt.xlabel('iterations')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig('results_data/training_loss.png')
        plt.show()
        torch.save(net.state_dict(), self.path_saved_model)
        return net
    
    class YoloLoss(nn.Module):
      def __init__(self, batch_size):
        super(YoloNaive.YoloLoss, self).__init__()
        self.batch_size = batch_size
      
      def forward(self, true, true_labels, pred):
        for idx in range(self.batch_size):
                true_x, true_y, true_w, true_h = true[idx][0], true[idx][1], true[idx][2], true[idx][3]
                true_label = true_labels[idx]
                coor_loss_sum = 0.0
                w_h_loss_sum = 0.0
                label_loss_sum = 0.0
                has_obj_loss_sum = 0.0
                no_obj_loss_sum = 0.0
                prob_label_loss_sum = 0.0
                for i in range(36):
                    max_conf = 0.0
                    max_conf_idx = 0
                    for j in range(5):
                      if pred[idx][j+i*5][0] > max_conf:
                        max_conf = pred[idx][j+i*5][0]
                        max_conf_idx = j+i*5
                    for j in range(5):
                        conf = pred[idx][j+i*5][0]
                        offset = pred[idx][j+i*5][1]^2 + pred[idx][j+i*5][2]^2
                        w_h = (math.sqrt(true_w) - math.sqrt(pred[idx][j+i*5][4]*20))^2 + (math.sqrt(true_h) - math.sqrt(pred[idx][j+i*5][3]*20))^2

                        pred_x = true_x + pred[idx][j+i*5][1]
                        pred_y = true_y + pred[idx][j+i*5][2]
                        pred_h = pred[idx][j+i*5][3]
                        pred_w = pred[idx][j+i*5][4]
                        obj_err = IoU([true_x, true_y, true_w, true_h], [pred_x, pred_y, pred_w, pred_h])
                        no_obj_err = IoU([true_x, true_y, true_w, true_h], [pred_x, pred_y, pred_w, pred_h])
                        if j+i*5 != max_conf_idx:
                          offset = 0*offset
                          w_h = 0*w_h
                          obj_err = 0*obj_err
                        if j+i*5 == max_conf_idx:
                          no_obj_err = 0*no_obj_err
                        coor_loss_sum += offset
                        w_h_loss_sum += w_h
                        has_obj_loss_sum += obj_err^2
                        no_obj_loss_sum += no_obj_err^2
                    for c in range(3):
                      label_loss = (true_labels[idx][i] - pred[idx][max_conf_idx][c+5])^2
                      prob_label_loss_sum += label_loss
                    if pred[idx][max_conf_idx][0] < 0.5:
                      prob_label_loss_sum *= 0
                total_loss = 5*coor_loss_sum + 5*w_h_loss_sum + has_obj_loss_sum + 0.5*no_obj_loss_sum + prob_label_loss_sum
        return total_loss
      
      def IoU(self, inp, tar):
          t_x, t_y, t_w, t_h = inp[0], inp[1], inp[2], inp[3]
          p_x, p_y, p_w, p_h = tar[0], tar[1], tar[2], tar[3]
          min_x = min(t_x, p_x, t_x+t_w, p_x+p_w)
          min_y = min(t_y, p_y, t_y+t_h, p_y+p_h)
          max_x = max(t_x, p_x, t_x+t_w, p_x+p_w)
          max_y = max(t_y, p_y, t_y+t_h, p_y+p_h)
          composite_loss = []
          union = intersection = 0.0
          for i in range(int(min_x), int(max_x)):
            for j in range(int(min_y), int(max_y)):
                if i >= t_x and i <= (t_x+t_w) and j>= t_y and j<=(t_y+t_h):
                  if i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                    intersection += 1
                    union += 1
                  elif i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                    union += 1
          if union == 0.0:
            raise Exception("\n\nSomething wrong")
          sample_iou = intersection / float(union)
          return 1 - torch.tensor([sample_iou])      
    

    def testing_part(self, net, display_img=False):
        def IoU(inp, tar):
            t_x, t_y, t_w, t_h = inp[0], inp[1], inp[2], inp[3]
            p_x, p_y, p_w, p_h = tar[0], tar[1], tar[2], tar[3]
            min_x = min(t_x, p_x, t_x+t_w, p_x+p_w)
            min_y = min(t_y, p_y, t_y+t_h, p_y+p_h)
            max_x = max(t_x, p_x, t_x+t_w, p_x+p_w)
            max_y = max(t_y, p_y, t_y+t_h, p_y+p_h)
            composite_loss = []
            union = intersection = 0.0
            for i in range(int(min_x), int(max_x)):
              for j in range(int(min_y), int(max_y)):
                  if i >= t_x and i <= (t_x+t_w) and j>= t_y and j<=(t_y+t_h):
                    if i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                      intersection += 1
                      union += 1
                    elif i>=p_x and i<=(p_x+p_w) and j>=p_y and j<=(p_y+p_h):
                      union += 1
            if union == 0.0:
              return 1.0
            sample_iou = intersection / float(union)
            return sample_iou
        net.load_state_dict(torch.load(self.path_saved_model))
        net.eval()
        net = net.to(self.device)
        yolo_interval = self.interval
        num_yolo_cells = (self.img_size // yolo_interval) * (self.img_size // yolo_interval)
        num_anchor_boxes = 5
        yolo_tensor = torch.zeros(self.batch_size, num_yolo_cells, num_anchor_boxes, 8)
        true_label = []
        pred_label = []
        label_dict = {i: label for i, label in enumerate(sorted(self.class_list))}
        label_dict[13] = 'FA'
        i=0
        with torch.no_grad():
          for key, data in enumerate(self.test_dataloader):
            imgId, im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_image = data
          # if key % 50 == 49:
            print('\n\n\nShowing output for test batch %d: ' % (key+1))
            im_tensor = im_tensor.to(self.device)
            bbox_tensor = bbox_tensor.to(self.device)
            bbox_label_tensor = bbox_label_tensor.to(self.device)
            yolo_tensor = yolo_tensor.to(self.device)
            output = net(im_tensor)
            predictions = output.view(2, 36, 5, 8)
            for ibx in range(predictions.shape[0]):
              icx_2_best_ab = {ic: None for ic in range(36)}
              for icx in range(predictions.shape[1]):
                cell_predi = predictions[ibx, icx]
                prev_best = 0
                for abdx in range(cell_predi.shape[0]):
                  if cell_predi[abdx][0] > cell_predi[prev_best][0]:
                    prev_best = abdx
                best_ab_icx = prev_best
                icx_2_best_ab[icx] = best_ab_icx
              sorted_icx_to_box = sorted(icx_2_best_ab, key=lambda x: predictions[ibx, x, icx_2_best_ab[x]][0].item(), reverse=True)
              retained_cells = sorted_icx_to_box[:5]
              objects_detected = []
              coco = COCO(self.json_path)
              img = coco.loadImgs(imgId[0].item())[0]
              I = io.imread(img['coco_url'])
              if len(I.shape) == 2:
                I = skimage.color.gray2rgb(I)
              fig, ax = plt.subplots(1,1)
              image = np.uint8(I)
              m = 0
              for icx in retained_cells:
                pred_vec = predictions[ibx, icx, icx_2_best_ab[icx]]
                print(icx)
                print(retained_cells)
                tx, ty, tw, th = bbox_tensor[ibx][m][0].item(), bbox_tensor[ibx][m][1].item(), bbox_tensor[ibx][m][2].item(),bbox_tensor[ibx][m][3].item()
                px, py, pw, ph = tx+pred_vec[1]*20, ty+pred_vec[2]*20, tw+pred_vec[4]*20, th+pred_vec[3]*20
                iou_err = IoU([tx,ty,tw,th], [px,py,pw,ph])
                print('\nIoU error for current image: ', iou_err)
                class_labels_predi = pred_vec[-3:]
                if torch.all(class_labels_predi < 0.05):
                  predicted_class_label = None
                else:
                  best_predicted_class_index = (class_labels_predi == class_labels_predi.max()).nonzero().squeeze().item()
                  predicted_class_label = sorted(self.class_list)[best_predicted_class_index]
                  objects_detected.append(predicted_class_label)
                if bbox_label_tensor[ibx][m] != 13:
                  t_label = sorted(self.class_list)[bbox_label_tensor[ibx][m]]
                else:
                  t_label = 'Wrong'
                
                image = cv2.rectangle(image, (int(tx), int(ty)), (int(tx+tw), int(ty+th)), (36, 255, 12), 1)
                image = cv2.rectangle(image, (int(px), int(py)), (int(px+pw), int(py+ph)), (0,38,255), 1)
                image = cv2.putText(image, t_label, (int(tx+tw), int(ty-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (36,255,12),1)
                image = cv2.putText(image, predicted_class_label, (int(px+pw), int(py-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,38,255),1)
                # print(bbox_label_tensor)
                # print(bbox_label_tensor[ibx][icx])
                true_label.append(bbox_label_tensor[ibx][m].item())
                pred_label.append(best_predicted_class_index)
                m += 1
              ax.imshow(image)
              ax.set_axis_off()
              plt.axis('tight')
              ax.figure.savefig('results_data/example%d.jpg' % i)
              i += 1
              plt.close(ax.figure)
              plt.show()
              print('\n\n[batch image=%d objects found in descending probability order: '% ibx, objects_detected)
        acc = np.sum(np.array(pred_label)==np.array(true_label)) / len(pred_label)
        false_alarm = 0
        for elem in pred_label:
          if elem == 13:
            false_alarm += 1
        false_alarm_rate = false_alarm / len(true_label)
        arr = confusion_matrix(true_label, pred_label)
        print(acc, arr, false_alarm_rate)
        df_cm = pd.DataFrame(arr, index=[v for k, v in label_dict.items()], columns=[v for k, v in label_dict.items()])
        ax = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Accuracy = %.3f' % acc)
        ax.figure.savefig('results_data/heatmap.jpg')
        plt.close(ax.figure)

        

