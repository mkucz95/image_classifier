import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import network
import json


def preprocess_img(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms) #we want to use the train_dir data set to train, and apply the above transformations to all the data
    valid_data = datasets.ImageFolder(valid_dir, transform=test_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data)
    testloader = torch.utils.data.DataLoader(test_data)

    return train_data, valid_data, test_data, trainloader, validloader, testloader

def save_checkpoint(model, train_data, optimizer, save_dir, arch):
    model_checkpoint = {
            'input_size': model.classifier.hidden_layers[0].in_features,
            'output_size': model.classifier.output.out_features,
            'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
            'state_dict': model.classifier.state_dict(),
            'class_idx': train_data.class_to_idx,
            'optim_idx': optimizer.state_dict,
            'dropout': model.classifier.dropout.p,
            'arch':arch
                   }
    if(save_dir == None): torch.save(model_checkpoint, 'checkpoint.pth')
    else:torch.save(model_checkpoint, save_dir+'checkpoint.pth')

def load_model(filepath, gpu):
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else: device = "cpu"
    
    checkpoint = torch.load(filepath)
    model = torch_model(checkpoint['arch'])
    newModel = network.Network(checkpoint['input_size'],
                     checkpoint['output_size'],
                     checkpoint['hidden_layers'], 
                     checkpoint['dropout'])
    newModel.load_state_dict(checkpoint['state_dict'])
    model.classifier = newModel
    model = model.float().to(device)
    return model

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    img = Image.open(image)
    img = img.resize((256,256))
    #img dimensions
    width, height = img.size
    #img boundaries for a 224 crop
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    coordinates = (left,top,right,bottom)
    #crop image to 224 pixels
    img = img.crop(coordinates)
    
    #get 3d np array from img
    np_img = np.array(img)
    
    #Divide by 255 to get values between 0 and 1, then normalize by subtracting mean and dividing by std dev
    np_img = ((np_img / 255)- [0.485, 0.456, 0.406])/([0.229, 0.224, 0.225])
    
    #Set the color to the first channel
    np_img = np.ndarray.transpose(np_img, (2,0,1)) #change the channel order (shift by 1)
    img_tensor = torch.from_numpy(np.expand_dims(np_img, axis=0)).float() ##similar to .unsqueeze(1), found on slack channel

    return img_tensor

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = np.squeeze(image.numpy(), axis=0).transpose((1, 2, 0))

    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    ax.imshow(image)
    
    return ax

def torch_model(arch):
     try: model = getattr(models, arch)(pretrained=True)
     except:
         print("%r is not valid torchvision model" % arch)
         raise
     return model

def get_class(classes, checkpoint, category_names):
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)

    class_to_idx = torch.load(checkpoint)['class_idx'] 
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()} #geta dict with mapping (class index, class 'name')
    names = []
    for i in classes:
        category = idx_to_class[i] #convert index of top5 to class number
        name = cat_to_name[category] #convert category/class number to flower name
        names.append(name)
    return names

def show_classes(probabilities, classes, top_k):
    fig, ax = plt.subplots()
    ax.barh(np.arange(top_k), probabilities)
    ax.set_aspect(0.1)
    ax.set_yticks(np.arange(top_k))
    ax.set_yticklabels(classes, size='small')
    ax.set_title('Class Probability')
    ax.set_xlim(0,max(probabilities)+0.1)
    plt.tight_layout()