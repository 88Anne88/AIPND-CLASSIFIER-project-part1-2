# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 02:23:59 2018

@author: Anne Gallon
"""

#CLASSIFIER PROJECT PART 2

#Import Packages and Modules
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import json
from PIL import Image,ImageOps
import numpy as np
import argparse


   

def main():
    
    #Print some informations for the commmand lines, show the default values.     
    print("Command line arguments are: \n-s --save_dir (set to default path)"
          "\n-a --arch  You can choose between: vgg13/vgg16/vgg19,default=vgg16"
          "\n-lr --learning_rate (default=0.001)"
          "\n-hi --hidden_units (default=1000)"
          "\n-e --epochs (default=7)"
          "\n-g --gpu (default=cuda)")
    
    #Creates and retrieves 6 Command Line Arguments        
    args= get_input_args()
           
    #Check the Command line arguments, retrieves the args choosen
    check_command_line(args)

    #Data paths
    data_dir = "/home/workspace/aipnd-project/flowers"
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    # Define your transforms for the training, validation, and testing sets
    #data_transforms:
    train_transforms= transforms.Compose([transforms.Resize(255),
                                         transforms.RandomRotation(30),
                                         transforms.RandomResizedCrop(224),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                        ])
    valid_transforms= transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                        ])
    test_transforms=  transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                                        ])


    #Load the datasets with ImageFolder
    #image_datasets:
    train_data= datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data= datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data= datasets.ImageFolder(test_dir, transform=test_transforms)
    
    

    #Using the image datasets and the transforms, define the
    #dataloaders:
    trainloader= torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validloader= torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader= torch.utils.data.DataLoader(test_data, batch_size=32)
    #import the dictionary with flowers names
    import json

    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)


    #Build and train your network
    #Load a pretrained network: 3 choices between VGG13/VGG16/VGG19
    #retrieve the model entered in the command line with the argument -a --arch
   
    if args.arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif args.arch == "vgg19":
        model = models.vgg19(pretrained=True)
    else:
        model = models.vgg16(pretrained=True)
    model

    #Freeze the parameters (so we don't backprop through them)
    for param in model.parameters():
        param.requires_grad = False


    #Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout 
    # Nb of hidden layers comes from args, default is 1000
  
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(25088, args.hidden_units)), 
                              ('relu', nn.ReLU()),
                              ('dropout1',nn.Dropout(0.5)),
                              ('fc2', nn.Linear(args.hidden_units, 102)), 
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    
    model.classifier = classifier
    model.class_to_idx = train_data.class_to_idx


    #TRAINING=> DEFINE OUR HYPERPARAMETERS AND CALL THE TRAINING FUNCTION 
    #the epochs, device and learning rate comes from the input args in command line
   
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    epochs =args.epochs
    steps = 0
    model.to(args.gpu)
    print_every = 40

    train_model(model,trainloader,validloader, criterion, optimizer, epochs,
          print_every) 

    print('Training is finished')



   #CALLING THE TESTING FUNCTION TO RETURN OUR NETWORK PERFORMANCE (Accuracy and Loss)   
   
    with torch.no_grad():
        test_loss, accuracy = testing_accuracy(model,testloader,criterion)
                
    print("The Network's Accuracy is:%0.2f %%" %(100*(accuracy/len(testloader))))
    print("The Network's Loss is: {:.3f}".format(test_loss/len(testloader)))             




    # CREATE AND SAVE A CHECKPOINT WITH OUR PARAMETERS 
    checkpoint={'input_size': 25088,
                'output_size': 102,
                'hidden_layers':[args.hidden_units],
                'classifier':model.classifier,
                'state_dict': model.state_dict(),
                'optimizer_dict': optimizer.state_dict(),
                'epochs': epochs,
                'criterion': criterion,
                'print_every':print_every,
                'learnrate':args.learning_rate,
                'class_to_idx': train_data.class_to_idx,
                'arch': model
            
               }

    torch.save(checkpoint,'checkpoint.pth')
    #i was not sure how to indicate the path previously saved in args --save_dir to save this chechpoint





# Function to Create 6 command line arguments: 
    
##Set a default directory to save a checkpoint: /python train.py data_dir --save_dir save_directory  
##Choose architecture: python train.py data_dir --arch "vgg16"  
##Set hyperparameters: python train.py data_dir --learning_rate 0.001 --hidden_units 1000 --epochs 7  
##Use GPU for training: python train.py data_dir --gpu 
    
def get_input_args():
    parser= argparse.ArgumentParser(description="Get inputs from user")

    parser.add_argument("-s","--save_dir",type=str,default="/home/workspace/aipnd-project",help="path to save the checkpoint")
     
    parser.add_argument("-a","--arch",dest="arch",default="vgg16",type=str,help="Select a pretrained arch")
   
    parser.add_argument("-lr","--learning_rate",dest="learning_rate",default=0.001,metavar="", help="Select learning rate ,default= 0.001")
    
    parser.add_argument("-hi","--hidden_units",dest="hidden_units",default=1000,metavar="", type=int, help="Select nb hidden layers ,default= 1000")
    
    parser.add_argument("-e","--epochs",dest="epochs",default=7,metavar="",type=int, help="Select nb epochs ,default= 7")

    parser.add_argument("-g","--gpu",type=str,dest="gpu",default='cuda',metavar="", help="set device, default=cuda(GPU)")
    
    return parser.parse_args()    
 

def check_command_line(args):
    # prints command line args.
    print("\nHere the arguments choosen:\nSave directory: -s=>",args.save_dir,
          "\nArch:           -a=>",args.arch,
          "\nLearn rate:     -lr=>",args.learning_rate,
          "\nHidden Layers:  -hi=>",args.hidden_units,
          "\nEpochs:         -e=>",args.epochs,
          "\nDevice=:        -g=>",args.gpu)
         
 

#Create a Validation function: to check the accuracy and test loss during training.
def validation(model,validloader,criterion):

    accuracy = 0
    test_loss = 0
    count=0
    
    model.to('cuda')  
    
    for images,labels in validloader:
        
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ##Calculating the accuracy 
        # Model's output is a log-softmax, so we take the exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        count+=1
    return test_loss, accuracy



#Create a TRAINING function with a validation step
def train_model(model, trainloader, validloader, criterion, optimizer, epochs,
          print_every):
    

    epochs=epochs
    print_every = print_every
    steps = 0
    
    model.to('cuda')
    
    for e in range(epochs):   
        # Model in training mode (dropout is on)
        model.train()
        running_loss = 0
    
        for ii,(inputs,labels) in enumerate(trainloader):
            steps += 1
            
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
            

            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            #VALIDATION STEPS
            if steps % print_every == 0:
                # Model in inference mode, dropout is off
                model.eval()
                
                # Turn off gradients and printing the Training loss and Accuracy.
                with torch.no_grad():
                    test_loss, accuracy = validation(model,validloader, criterion)
                
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)),
                      "Validation Loss: {:.3f}.. ".format(test_loss/len(validloader)))
                
                running_loss = 0
                
                # Make sure dropout and grads are on for training
                model.train()  


#DEFINE A TESTING FUNCTION

def testing_accuracy(model,testloader,criterion):
    
    accuracy = 0
    test_loss = 0
    count=0
    
    model.to('cuda')
    
    for images,labels in testloader:
        
        images, labels = images.to('cuda'), labels.to('cuda')
        output = model.forward(images)
        test_loss += criterion(output, labels).item()

        ##Calculating the accuracy 
        # Model's output is a log-softmax, so we take the exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        # Accuracy is number of correct predictions divided by all predictions, just take the mean
        accuracy += equality.type_as(torch.FloatTensor()).mean()
        count+=1
    return test_loss, accuracy




                 
                 
# Call to main function to run the program
if __name__ == "__main__":
    main()
    
       

    