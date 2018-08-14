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
parser.add_argument('input', action='store', help='path to image to be classified')
parser.add_argument('checkpoint', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store',nargs=1, type=int, default=5, help='how many most probable classes to print out')
parser.add_argument('--category_names', action='store', help='file which maps classes to names')
parser.add_argument('--gpu', action='store_true', help='use gpu to infer classes')
args=parser.parse_args()

if args.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else: device = "cpu"

model = utility.load_model(args.checkpoint, args.gpu).eval()
img = utility.process_image(args.input).to(device) #process image to pytensor using device
outputs = model(img) #see how our network classifies
prob = torch.exp(outputs) #get the exponents back #get prediction of model, take exponent to undo log_softmax
result = torch.topk(prob, args.top_k) #top 5 probabilities    

top_probs = result[0][0].cpu().detach().numpy() #get top5 from pytroch tensor to numpy
classes = result[1][0].cpu().numpy() #index of top5 probabilities

if(args.category_names != None): 
    classes = utility.get_class(classes, args.checkpoint, args.category_names)

utility.imshow(img.to('cpu'), ax=None)
utility.show_classes(top_probs, classes, args.top_k)
    
