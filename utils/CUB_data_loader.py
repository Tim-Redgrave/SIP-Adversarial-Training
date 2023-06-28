import os
import numpy as np
import torch
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
import sys

class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, random=True, map_location='',im_size=224):
        self.map_location = map_location
        self.image_size = im_size

        # Resize our images and make sure they are tensors
        self.data = []
        self.transform = transforms.Compose([
            transforms.Resize([self.image_size, self.image_size]),
            transforms.ToTensor()
        ])

        # Reading data from CSV file
        SegInfo=[]
        with open(split_file, 'r') as f:
            for l in f.readlines():
                v= l.strip().split(',')
                if train_test == int(v[0]):
                    image_name = v[2]
                    imagePath = os.path.join(root,image_name)
                    img = Image.open(imagePath).convert('RGB')
                    tranform_img = self.transform(img)
                    img.close()
                    annotation_map_path = os.path.join(self.map_location,image_name.split("/")[-2],image_name.split("/")[-1][:-3]+"png")
                    if train_test == 1 and os.path.exists(annotation_map_path):
                        annotations = Image.open(annotation_map_path).convert('L')
                        annotations_tensor = self.transform(annotations).type(torch.float)
                        annotations.close()
                    else:
                        annotations_tensor = 0
                    class_num = int(v[1])-1

                    # Storing data with imagepath and class
                    self.data.append([imagePath,class_num,tranform_img,annotations_tensor])

        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls, img, hmap = self.data[index]
        imageName = imagePath.split('/')[-1]
        return img, cls, imageName, hmap

    def __len__(self):
        return len(self.data)

def create_data_loader(data_split_path,data_root_path,map_location,network,batch_size):
    im_size = 299 if network == "inception" else 224
    training_loader = datasetLoader(data_split_path,data_root_path,train_test=1,map_location=map_location,im_size=im_size)
    training_data_loader = torch.utils.data.DataLoader(training_loader,batch_size,shuffle=True,num_workers=0,pin_memory=True)
    testing_loader = datasetLoader(data_split_path,data_root_path,train_test=0,map_location=map_location,im_size=im_size)
    testing_data_loader = torch.utils.data.DataLoader(testing_loader,batch_size,shuffle=True,num_workers=0,pin_memory=True)
    train_and_test_loader = {'train':training_data_loader,'test':testing_data_loader}
    return train_and_test_loader

def create_test_set_loader(data_split_path,data_root_path,map_location,network,batch_size):
    im_size = 299 if network == "inception" else 224
    testing_loader = datasetLoader(data_split_path,data_root_path,train_test=0,map_location=map_location,im_size=im_size)
    testing_data_loader = torch.utils.data.DataLoader(testing_loader,batch_size,shuffle=True,num_workers=0,pin_memory=True)
    return testing_data_loader
