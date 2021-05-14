import torch
import os
import numpy as np
import glob
import random
import torchvision
import torchvision.transforms as tvt
import requests
from skimage import io
from pycocotools.coco import COCO
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class CocoLoader(Dataset):
    def __init__(self, dataset_size, train_or_test, json_path, class_list, root_train=None, root_test=None, transforms=None):
        super(CocoLoader, self).__init__()
        self.size = dataset_size
        self.train_or_test = train_or_test
        self.root_train = root_train
        self.root_test = root_test
        self.json_path = json_path
        self.class_list = class_list
        self.train_dataset = {}
        self.test_dataset = {}
        self.train_set_size = None
        self.test_set_size = None
        self.labels = None
        self.target_size = 128
        if self.train_or_test == 'train':
            self.train_d = self.create_set()
        if self.train_or_test == 'test':
            self.test_d = self.create_set()

    def create_set(self):
        if self.train_or_test == 'train':
            dataroot = self.root_train
        elif self.train_or_test == 'test':
            dataroot = self.root_test
        if self.train_or_test == 'train' and dataroot == self.root_train:
            if os.path.exists('results_data/saved_train_coco_set.pt'):
                print('\nLoading training set from local path...')
                self.train_dataset = torch.load(
                    'results_data/saved_train_coco_set.pt')
                self.train_set_size = len(self.train_dataset)
            else:
                print('\nDo not find training set in local path\n'
                      'Creating training set for the first time...\n')
                self.download_from_json()
                print('\nTraining set created\n')
        elif self.train_or_test == 'test' and dataroot == self.root_test:
            self.download_from_json()

    def download_from_json(self):
        coco = COCO(self.json_path)
        cat1, cat2, cat3 = self.class_list[0], self.class_list[1], self.class_list[2]
        all_training_images = list()
        all_testing_images = list()
        # get imgs with all three categoties
        catIds = coco.getCatIds(catNms=self.class_list)
        subset_imgIds123 = coco.getImgIds(catIds=catIds)
        print('\nAll img ids: ', len(subset_imgIds123))
        # get imgs with at at least two categories
        catIds12 = coco.getCatIds(catNms=[cat1, cat2])
        catIds23 = coco.getCatIds(catNms=[cat2, cat3])
        catIds13 = coco.getCatIds(catNms=[cat1, cat3])
        imgIds12 = coco.getImgIds(catIds=catIds12)
        imgIds23 = coco.getImgIds(catIds=catIds23)
        imgIds13 = coco.getImgIds(catIds=catIds13)
        subset_imgIds12 = list(set(imgIds12) - set(subset_imgIds123))
        subset_imgIds23 = list(set(imgIds23) - set(subset_imgIds123))
        subset_imgIds13 = list(set(imgIds13) - set(subset_imgIds123))
        print('Cat12 has:', len(subset_imgIds12))
        print('Cat23 has:', len(subset_imgIds23))
        print('Cat13 has:', len(subset_imgIds13))
        # get imgs within each category with at least two instances
        catId1 = coco.getCatIds(catNms=cat1)
        catId2 = coco.getCatIds(catNms=cat2)
        catId3 = coco.getCatIds(catNms=cat3)
        imgIds1 = coco.getImgIds(catIds=catId1)
        imgIds2 = coco.getImgIds(catIds=catId2)
        imgIds3 = coco.getImgIds(catIds=catId3)

        def check_num_ann(imgIds_gp, catId):
            subset = list()
            for imgId in imgIds_gp:
                img = coco.loadImgs(imgId)[0]
                annIds_w_cat = coco.getAnnIds(
                    imgIds=img['id'], catIds=catId, iscrowd=False)
                annIds = coco.getAnnIds(imgIds=img['id'], iscrowd=False)
                if len(annIds_w_cat) >= 2 and len(annIds) <= 5:
                    subset.append(imgId)
            return subset

        subset_imgIds1 = list(set(check_num_ann(
            imgIds1, catId1))-set(subset_imgIds12)-set(subset_imgIds13)-set(subset_imgIds123))
        subset_imgIds2 = list(set(check_num_ann(
            imgIds2, catId2))-set(subset_imgIds12)-set(subset_imgIds23)-set(subset_imgIds123))
        subset_imgIds3 = list(set(check_num_ann(
            imgIds3, catId3))-set(subset_imgIds23)-set(subset_imgIds13)-set(subset_imgIds123))
        print('Cat1 has:', len(subset_imgIds1))
        print('Cat2 has:', len(subset_imgIds2))
        print('Cat3 has:', len(subset_imgIds3))

        total_subset = list(set(subset_imgIds1+subset_imgIds2+subset_imgIds3 +
                                subset_imgIds12+subset_imgIds23+subset_imgIds13+subset_imgIds123))
        print('Subset: ', len(total_subset))

        if self.train_or_test == 'train':
            total_subset = total_subset[:self.size]
            coco.download(self.root_train, total_subset)
            for imgId in total_subset:
                img = coco.loadImgs(imgId)[0]
                img_path = os.path.join(self.root_train, img['file_name'])
                all_training_images.append([imgId, img_path])
            for img_id, path in all_training_images:
                im = Image.open(path)
                if im.mode != 'RGB':
                  im = im.convert(mode='RGB')
                im = im.resize((self.target_size, self.target_size), Image.BOX)
                im.save(path)
            random.shuffle(all_training_images)
            self.train_dataset = {
                i: all_training_images[i] for i in range(len(all_training_images))}
            torch.save(
                self.train_dataset, 'results_data/saved_train_coco_set.pt')
            self.train_set_size = len(all_training_images)
        elif self.train_or_test == 'test':
            total_subset = total_subset[:self.size]
            coco.download(self.root_test, total_subset)
            for imgId in total_subset:
                img = coco.loadImgs(imgId)[0]
                img_path = os.path.join(self.root_test, img['file_name'])
                all_testing_images.append([imgId, img_path])
            for img_id, path in all_testing_images:
                im = Image.open(path)
                if im.mode != 'RGB':
                  im = im.convert(mode='RGB')
                im = im.resize((self.target_size, self.target_size), Image.BOX)
                im.save(path)
            random.shuffle(all_testing_images)
            self.test_dataset = {
                i: all_testing_images[i] for i in range(len(all_testing_images))}
            self.test_set_size = len(all_testing_images)

    def __len__(self):
        if self.train_or_test == 'train':
            return self.train_set_size
        elif self.train_or_test == 'test':
            return self.test_set_size

    def __getitem__(self, idx):
        if self.train_or_test == 'train':
            imgId, img_path = self.train_dataset[idx]
        elif self.train_or_test == 'test':
            imgId, img_path = self.test_dataset[idx]
        coco = COCO(self.json_path)
        im = Image.open(img_path)
        im_tensor = tvt.ToTensor()(im)
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgId, iscrowd=False)
        anns = coco.loadAnns(annIds)
        bbox_tensor = torch.zeros(5, 4, dtype=torch.uint8)
        bbox_label_tensor = torch.zeros(5, dtype=torch.uint8) + 13
        num_objects_in_img = len(anns[:5])
        obj_class_labels = sorted(self.class_list)
        obj_class_label_dict = {
            obj_class_labels[i]: i for i in range(len(obj_class_labels))}
        for i in range(num_objects_in_img):
            bbox = anns[i]['bbox']
            label_id = anns[i]['category_id']
            cat = coco.loadCats(label_id)
            label = cat[0]['name']
            if label in obj_class_label_dict:
              bbox_label_tensor[i] = obj_class_label_dict[label]
            bbox_tensor[i] = torch.LongTensor(bbox)
        return imgId, im_tensor, bbox_tensor, bbox_label_tensor, num_objects_in_img
