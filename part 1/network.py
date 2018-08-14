import torch
from torch import nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p):
        super().__init__()
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        #create hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:]) #gives input/output sizes for each layer
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
    
    def forward(self, x):
        for each in self.hidden_layers:
            x = F.relu(each(x)) #apply relu to each hidden node
            x = self.dropout(x) #apply dropout
        x = self.output(x) #apply output weights
        return F.log_softmax(x, dim=1) #apply activation log softmax
    
def check_accuracy_loss(model, loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    accuracy = 0
    loss=0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images) #see how our network classifies
            prob = torch.exp(outputs) #get the exponents back
            results = (labels.data == prob.max(1)[1]) #which labels == our predictions (highest probability gives prediction)
           # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy+=results.type_as(torch.FloatTensor()).mean() 
            loss+=criterion(outputs,labels)     
    return accuracy/len(loader), loss/len(loader) #since acc and loss are sums, we need to get the avg over all the input images
    
    #NETWORK TRAINING
def train_network(model, trainloader, validloader, epochs, print_every, criterion, optimizer, scheduler):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps=0
    # change to cuda
    model.to(device)
    model.train() #training mode

    for e in range(epochs):
        scheduler.step() #we use scheduler
        running_loss=0
        for ii, (inputs,labels) in enumerate(trainloader):
            steps+=1
            inputs, labels = inputs.to(device), labels.to(device) #move data to gpu
            optimizer.zero_grad()#zero out gradients so that one forward pass doesnt pick up previous forward's gradients
            outputs = model.forward(inputs)
            loss = criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            #code below courtesy of udacity
            if steps % print_every == 0:
                accuracy,valid_loss = check_accuracy_loss(model,validloader,criterion)
                           
                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Training Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Loss: {:.4f}".format(valid_loss),
                      "Validation Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
                
            model.train()

def predict(image_path, model, topk=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model = model.eval()
    img = process_image(image_path).to(device) #process image to pytensor using device
    outputs = model(img) #see how our network classifies
    prob = torch.exp(outputs) #get the exponents back #get prediction of model, take exponent to undo log_softmax
    result = torch.topk(prob, topk) #top 5 probabilities    
    
    top5 = result[0][0].cpu() #get top5 from pytroch tensor to numpy
    classes = result[1][0].cpu() #index of top5 probabilities
    
    ## TODO: each class has probability in softmax; we have to pick top 5 classes,
        #therefore we must find, ie find the index that has each respective probability of top 5 === classes

    return top5.detach().numpy(), classes.numpy()

def get_class(classes):
    class_to_idx = torch.load('checkpoint.pth')['class_idx'] 
    idx_to_class = {idx: pic for pic, idx in class_to_idx.items()} #geta dict with mapping (class index, class 'name')
    names = []
    for i in classes:
        category = idx_to_class[i] #convert index of top5 to class number
        name = cat_to_name[category] #convert category/class number to flower name
        names.append(name)
    return names

def show_classes(probabilities, classes):
        fig, ax = plt.subplots()
        ax.barh(np.arange(5), probabilities)
        ax.set_aspect(0.1)
        ax.set_yticks(np.arange(5))
        ax.set_yticklabels(classes, size='small')
        ax.set_title('Class Probability')
        ax.set_xlim(0,max(probabilities)+0.1)
        plt.tight_layout()
        
def sanity_check(filepath, model):
    probs, classes = predict(filepath, model, topk=5)
    names=get_class(classes)
    imshow(process_image(filepath), ax=None, title='Flower')
    show_classes(probs, names)