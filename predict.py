# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 02:23:59 2018

@author: Anne Gallon
"""

#CLASSIFIER PROJECT PART 2/ PREDICTION 

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


	#Creates and retrieves 3 Command Line Arguments        
    args= get_input_args()
           
    #Check the Command line arguments, retrieves the args choosen
    check_command_line(args)

    #Load our checkpoint and use it to create a new model
    model = load_checkpoint('checkpoint.pth') 
    model#to check 


    #CALLING THE IMAGE PROCESS FUNCTION 
    #on a testing image "im"
    image_test='/home/workspace/aipnd-project/flowers/test/10/image_07090.jpg'

    processed_image=process_image(image_test)
    processed_image#check


    #CALLING THE FUNCTION TO PREDICT THE TOP 3 CLASSES AND PROBABILITIES OF AN IMAGE FROM THE TEST SET
    
    probs , classes, names  =  predict(image_test,model,topk=args.top_k)


    print ("The image is a: {}, class:{}, with a probability of {}% ".format(names[0],classes[0],probs[0]*100))

    print ("The 3 top classes are:",classes)
    print("The 3 top probabilities:",probs)




  
def get_input_args():
    parser= argparse.ArgumentParser(description="Get inputs from user")

    parser.add_argument("-t","--top_k",type=int,default=3,help="choose top K nb")
     
    parser.add_argument("cn","--category_names ",dest="cat_to_name",default="cat_to_name.json",type=str,help="cat_to_name json file")
   
    parser.add_argument("g","--gpu",dest="gpu",default="cuda",type=str,help="choose device cuda or cpu")

    return parser.parse_args()    
 

def check_command_line(args):
    # prints command line args.
    print("\nHere the arguments choosen:\nTop K: -t=>",args.top_k,
          "\nCategory to name:                   -cn=>",args.cat_to_name,
          "\nDevice=:                             -g=>",args.gpu)





#DEFINE A FUNCTION TO LOAD THE CHECKPOINT AND CREATE A NEW MODEL 
def load_checkpoint(filepath):

    checkpoint = torch.load(filepath)#to load on cpu add arg:(map_location='cpu' or lambda storage,loc: storage)
    epochs = checkpoint['epochs']
    optimizer = checkpoint['optimizer_dict']
    model = checkpoint['arch']
    model.class_to_idx =  checkpoint['class_to_idx']
    model.classifier =  checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])

    for param in model.parameters():
        param.requires_grad = False

    return model


 

#Define a function to Process a PIL image, to use in a PyTorch 
def process_image(image_test):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array'''
    
    #Resizing proportionally an image (smallest side=256px)
    #Source :i found this resizing code on a forum ,and changed the resize argument and adapted the code to my need.
    #Rodrigo Laguna: https://stackoverflow.com/questions/273946/how-do-i-resize-an-image-using-pil-and-maintain-its-aspect-ratio
    basewidth =256 
    baseheight=256
    img = Image.open(image)
    w_ratio = (basewidth/float(img.size[0]))
    height_size = int((float(img.size[1])*float(w_ratio)))
    img_resized = img.resize((basewidth,height_size),Image.LANCZOS)
    if img_resized.size[1] < 256:
        height_size=baseheight
        h_ratio = (baseheight/float(img.size[1]))
        width_size=int((float(img.size[0])*float(h_ratio)))
        img_sized = img.resize((width_size,baseheight),Image.LANCZOS)
    else:
        img_sized = img.resize((basewidth,height_size),Image.LANCZOS)
  
    
    #Center Crop the resized image to 224x224px:
    
    #i found a PIL method in the documentation, but i was not sure if i could use it,
    #as it is mentioned to be an experimental function.
    #pil_image= ImageOps.fit(img_sized,(224,224),Image.LANCZOS,0,(0.5,0.5))
    
    
    #Other method to Center Crop:
    height = img_sized.size[1] 
    width= img_sized.size[0]
    new_width, new_height= 224, 224
    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2
    pil_image= img_sized.crop((left, top, right, bottom))


    #Normalize the image
    np_image = np.array(pil_image)
    mean= [0.485, 0.456, 0.406]
    std= [0.229, 0.224, 0.225]
    img_norm= ((np_image/255)-mean)/std
    img_norm_t= img_norm.transpose(1,2,0)
    img_norm_t= img_norm_t.transpose(1,2,0) #i needed to do 2 transposes to get shape (3,224,224)

    return img_norm_t
   



#FUNCTION PREDICTION
def predict(image_test, model, topk=args.top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    # Implement the code to predict the class from an image file
    #source: JK2390-https://github.com/JK2390/AIPND-Image-Classifier/blob/master/Part%201/Image%20Classifier%20Project.ipynb
    #i modified the code to reach the project result
     
        
    with torch.no_grad():
        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        np_image = process_image(image_test)
        #pass the array into a tensor and convert to right Pytorch required size (1,3,224,224)
        image_tensor = torch.from_numpy(np_image)
        image_tensor =torch.unsqueeze(image_tensor,0)
        
        if args.gpu == 'cuda':
            model.to('cuda')
            inputs_var = image_tensor.float().cuda() 
        else:
            model.to('cpu')
            inputs_var = image_tensor.float() 
            
             
        output = model.forward(inputs_var)  
        ps = torch.exp(output).topk(args.top_k,largest=True,sorted=True)
        probs = ps[0].cpu()#moves the tensor to cpu to host memory
        classes = ps[1].cpu() #returns the indices of the k best top classes
        

        # Convert indices to classes
        class2idx= model.class_to_idx
        idx_to_class = {val:key for key, val in class2idx.items()}
        top_classes = [idx_to_class[each] for each in classes.numpy()[0]] #create a list of best classes
        top_classes = str(top_classes) #transforms the result into a string list
        
         
        #Create a list with the 5 best classes names, and define the top one.
        name_list = []
        with open(args.cat_to_name) as f:
            data = json.load(f)
            for flower_id in top_classes:
                name_list.append(data[str(flower_id)])
            flower_id= name_list[0]

   
    return probs.numpy()[0].tolist(), top_classes , name_list
    







# Call to main function to run the program
if __name__ == "__main__":
    main()
    