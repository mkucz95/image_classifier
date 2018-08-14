import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import network
import utility

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store', help='directory containing images')
parser.add_argument('--save_dir', action='store', nargs=1, help='save trained checkpoint to this directory' )
parser.add_argument('--arch', action='store', nargs=1, help='what kind of pretrained architecture to use', default='vgg19')
parser.add_argument('--gpu', action='store_true', help='use gpu to train model')
parser.add_argument('--epochs', action='store', help='# of epochs to train', type=int, default=4)
parser.add_argument('--learning_rate', action='store', help='which learning rate to start with', type=float, default=0.05)
parser.add_argument('--hidden_units', action='store', help='# of hidden units to add to model', type=int, default=4096)
parser.add_argument('--output_size', action='store', help='# of classes to output', type=int, default=102)

args=parser.parse_args()

#correctly sort the data for training, validation, testing
data_dir = args.data_dir
train_data, valid_data, test_data, trainloader, validloader, testloader = utility.preprocess_img(data_dir)

#create model using --arch argument with vgg19 as default
model = utility.torch_model(args.arch)
for param in model.parameters():
    param.requires_grad = False
#params are now frozen so that we do not backprop thru them again

#calculate input size into the network classifier
input_size = model.classifier[0].in_features

model.classifier = network.Network(input_size, args.output_size, args.hidden_units, drop_p=0.35)

#define the loss function and the optimization parameters
criterion = nn.NLLLoss() #want nllloss because we do the logsoftmax as our output activation
optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

#train model
network.train_network(model, trainloader,validloader,args.epochs, 32, criterion,optimizer,scheduler,args.gpu)

#test model
test_accuracy, test_loss = network.check_accuracy_loss(model, testloader, criterion, args.gpu)
print("\n ---\n Test Accuracy: {:.2f} %".format(test_accuracy*100), "Test Loss: {}".format(test_loss))

#save network to checkpoint
utility.save_checkpoint(model, train_data, optimizer, args.save_dir, args.arch)