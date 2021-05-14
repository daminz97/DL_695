import random
import numpy
import torch
import os, sys


seed = 0           
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
numpy.random.seed(seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmarks=False
os.environ['PYTHONHASHSEED'] = str(seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

from DLStudio import *

class SentimentAnalysisDataset_extend(DLStudio.TextClassification.SentimentAnalysisDataset):         
  def one_hotvec_for_word(self, word):
    word_index = self.vocab.index(word)
    hotvec = bin(word_index)[2:].zfill(16)
    hotvec_tensor = torch.zeros(16)
    for i in range(len(hotvec)):
      hotvec_tensor[i] = int(hotvec[i])
    return hotvec_tensor
                
  def review_to_tensor(self, review):
    review_tensor = torch.zeros(len(review), 16)
    for i,word in enumerate(review):
      review_tensor[i,:] = self.one_hotvec_for_word(word)
    return review_tensor

dls = DLStudio(
#                  dataroot = "/home/kak/TextDatasets/sentiment_dataset/",
                #   dataroot = "/data/TextDatasets/sentiment_dataset/",
                  dataroot = "./Examples/data/",
                  path_saved_model = "./Examples/result/saved_model.pt",
                  momentum = 0.9,
                  learning_rate =  1e-4,  
                  epochs = 1,
                  batch_size = 1,
                  classes = ('negative','positive'),
                  use_gpu = True,
              )

text_cl = DLStudio.TextClassification( dl_studio = dls )
dataserver_train = SentimentAnalysisDataset_extend(
                                 train_or_test = 'train',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_train_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_train_200.tar.gz",
                   )
dataserver_test = SentimentAnalysisDataset_extend(
                                 train_or_test = 'test',
                                 dl_studio = dls,
                                 dataset_file = "sentiment_dataset_test_40.tar.gz",
#                                 dataset_file = "sentiment_dataset_test_200.tar.gz",
                  )

text_cl.dataserver_train = dataserver_train
text_cl.dataserver_test = dataserver_test

text_cl.load_SentimentAnalysisDataset(dataserver_train, dataserver_test)

vocab_size = dataserver_train.get_vocab_size()

model = text_cl.TEXTnet(input_size=16, hidden_size=512, output_size=2)

number_of_learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
num_layers = len(list(model.parameters()))

print("\n\nThe number of layers in the model: %d" % num_layers)
print("\nThe number of learnable parameters in the model: %d" % number_of_learnable_params)
print("\nThe size of the vocabulary (which is also the size of the one-hot vecs for words): %d\n\n" % vocab_size)

text_cl.run_code_for_training_with_TEXTnet(model, hidden_size=512)

text_cl.run_code_for_testing_with_TEXTnet(model, hidden_size=512)