import sys,os,os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision                  
import torchvision.transforms as tvt
import torch.optim as optim
import numpy as np
from PIL import ImageFilter
import numbers
import re
import math
import random
import copy
import matplotlib.pyplot as plt
import gzip
import pickle
import time
import logging


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from DLStudio import *

class TextClassification_extend(DLStudio.TextClassification):
    def __init__(self, dl_studio, dataserver_train=None, dataserver_test=None, dataset_file_train=None, dataset_file_test=None):
        super(TextClassification_extend, self).__init__(dl_studio, dataserver_train, dataserver_test, dataset_file_train, dataset_file_test)
    
    class SentimentAnalysisDataset_extend(DLStudio.TextClassification.SentimentAnalysisDataset):   
        def __init__(self, dl_studio, train_or_test, dataset_file):
            super(TextClassification_extend.SentimentAnalysisDataset_extend, self).__init__(dl_studio, train_or_test, dataset_file)
            self.train_max_len = 0
            self.test_max_len = 0
            self.train_or_test = train_or_test
            root_dir = dl_studio.dataroot
            f = gzip.open(root_dir + dataset_file, 'rb')
            dataset = f.read()
            if train_or_test == 'train':
                if sys.version_info[0] == 3:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_train, self.negative_reviews_train, self.vocab = pickle.loads(dataset)
                self.categories = sorted(list(self.positive_reviews_train.keys()))
                self.category_sizes_train_pos = {category : len(self.positive_reviews_train[category]) for category in self.categories}
                self.category_sizes_train_neg = {category : len(self.negative_reviews_train[category]) for category in self.categories}
                self.indexed_dataset_train = []
                for category in self.positive_reviews_train:
                    for review in self.positive_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 1])
                for category in self.negative_reviews_train:
                    for review in self.negative_reviews_train[category]:
                        self.indexed_dataset_train.append([review, category, 0])
                random.shuffle(self.indexed_dataset_train)
                for review, cat, sent in self.indexed_dataset_train:
                    if len(review) > self.train_max_len:
                        self.train_max_len = len(review)
            elif train_or_test == 'test':
                if sys.version_info[0] == 3:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset, encoding='latin1')
                else:
                    self.positive_reviews_test, self.negative_reviews_test, self.vocab = pickle.loads(dataset)
                self.vocab = sorted(self.vocab)
                self.categories = sorted(list(self.positive_reviews_test.keys()))
                self.category_sizes_test_pos = {category : len(self.positive_reviews_test[category]) for category in self.categories}
                self.category_sizes_test_neg = {category : len(self.negative_reviews_test[category]) for category in self.categories}
                self.indexed_dataset_test = []
                for category in self.positive_reviews_test:
                    for review in self.positive_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 1])
                for category in self.negative_reviews_test:
                    for review in self.negative_reviews_test[category]:
                        self.indexed_dataset_test.append([review, category, 0])
                random.shuffle(self.indexed_dataset_test)
                for review, cat, sent in self.indexed_dataset_test:
                    if len(review) > self.test_max_len:
                        self.test_max_len = len(review)
        
        def one_hotvec_for_word(self, word):
            word_index = self.vocab.index(word)
            hotvec = bin(word_index)[2:].zfill(16)
            hotvec_tensor = torch.zeros(16)
            for i in range(len(hotvec)):
                hotvec_tensor[i] = int(hotvec[i])
            return hotvec_tensor
                
        def review_to_tensor(self, review, max_len):
            review_tensor = torch.zeros(max_len, 16)
            for i,word in enumerate(review):
                review_tensor[i,:] = self.one_hotvec_for_word(word)
            return review_tensor
        
        def __getitem__(self, idx):
            max_len = self.train_max_len if self.train_or_test == 'train' else self.test_max_len
            sample = self.indexed_dataset_train[idx] if self.train_or_test == 'train' else self.indexed_dataset_test[idx]
            review = sample[0]
            review_category = sample[1]
            review_sentiment = sample[2]
            review_sentiment = self.sentiment_to_tensor(review_sentiment)
            review_tensor = self.review_to_tensor(review, max_len)
            category_index = self.categories.index(review_category)
            sample = {'review'       : review_tensor, 
                        'category'     : category_index, # should be converted to tensor, but not yet used
                        'sentiment'    : review_sentiment }
            return sample

    class TEXTnetOrder2_extend(DLStudio.TextClassification.TEXTnetOrder2):
      def __init__(self, input_size, hidden_size, output_size):
        super(TextClassification_extend.TEXTnetOrder2_extend, self).__init__(input_size, hidden_size, output_size)
          # self.input_size = input_size
          # self.hidden_size = hidden_size
          # self.output_size = output_size
          # self.combined_to_hidden = nn.Linear(input_size + 2*hidden_size, hidden_size)
          # self.combined_to_middle = nn.Linear(input_size + 2*hidden_size, 100)
          # self.middle_to_out = nn.Linear(100, output_size)     
          # self.logsoftmax = nn.LogSoftmax(dim=1)
          # self.dropout = nn.Dropout(p=0.1)
          # # for the cell
          # self.linear_for_cell = nn.Linear(hidden_size, hidden_size)
          
      def forward(self, input, hidden, cell):
          combined = torch.cat((input, hidden, cell), 1)
          hidden = self.combined_to_hidden(combined)
          new_hidden = torch.sigmoid(self.linear_for_cell(hidden)) 
          new_hidden = torch.tanh(new_hidden)
          # additional gate for hidden                   
          out = self.combined_to_middle(combined)
          out = torch.nn.functional.relu(out)
          out = self.dropout(out)
          out = self.middle_to_out(out)
          out = self.logsoftmax(out)
          hidden_clone = hidden.clone()
          # cell = torch.tanh(self.linear_for_cell(hidden_clone))
          cell = torch.sigmoid(self.linear_for_cell(hidden_clone))
          return out,new_hidden,cell
      
      def initialize_cell(self, batch_size):
          weight = next(self.linear_for_cell.parameters()).data
          cell = weight.new(1, self.hidden_size).zero_()
          return cell

dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/sentiment_dataset/",
                #   dataroot = "/data/TextDatasets/sentiment_dataset/",
                  dataroot = "./Examples/data/",
                  path_saved_model = "./Examples/result/task3_textnet2_be_gating_model.pt",
                  momentum = 0.9,
                  learning_rate =  1e-5,  
                  epochs = 8,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )

text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = TextClassification_extend.SentimentAnalysisDataset_extend(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_train_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
                   )
dataserver_test = TextClassification_extend.SentimentAnalysisDataset_extend(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_test_3.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
                  )

text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()

model = TextClassification_extend.TEXTnetOrder2_extend(input_size=16, hidden_size=512, output_size=2)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

loss_arr = []
acc_arr = []
for i in range(1, 6):
  print('\nBatch Size: ', i)
  loss = text_cl.run_code_for_training_with_TEXTnetOrder2(model, hidden_size=512, batch_size=i)
  loss_arr.append(loss)

  acc = text_cl.run_code_for_testing_with_TEXTnetOrder2(model, hidden_size=512, batch_size=i)
  acc_arr.append(acc)

import matplotlib.pyplot as plt

print(loss_arr)
plt.figure(figsize=(10,5))
plt.title("Training Loss vs. Iterations")
for i in range(5):
  plt.plot(loss_arr[i], label=str('batch_size ' + str(i)))
plt.xlabel("iterations")
plt.ylabel("training loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.savefig("./Examples/result/task3_textnet2_be_gating_bs_loss.png")
plt.show()

plt.figure(figsize=(10,5))
plt.title("Testing Accuracy vs. Batch_size")
plt.plot(acc_arr)
plt.xlabel("iterations")
plt.ylabel("training loss")
plt.legend()
plt.savefig("./Examples/result/task3_textnet2_be_gating_bs_acc.png")
plt.show()

