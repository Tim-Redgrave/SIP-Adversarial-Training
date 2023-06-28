import os
import sys
import csv
import time
import glob
import numpy as np
import argparse
import json
import matplotlib.pyplot as plt
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from utils import CUB_data_loader, model_loader, PGD_attack_generator
sys.path.append("../")

# Description of all argument
parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=1)
parser.add_argument('-data_split_path', required=False, default= '../data/formatted_train_test_split.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= '../data/images/',type=str)
parser.add_argument('-heatmaps', required=False, default= '../data/segmentations/',type=str)
parser.add_argument('-model_type_path',type=str)
parser.add_argument('-network', default= 'densenet',type=str,choices=["resnet","densenet","inception"])
parser.add_argument('-nClasses', default= 200,type=int)
parser.add_argument('-pretrained', action='store',nargs='?',const="IMAGENET1K_V1",default=None)
parser.add_argument('-epsilon', default=8,type=float)
parser.add_argument('-num_attack_iters',default=32,type=int)
parser.add_argument('-attack_step_size',default=1./255,type=float)
parser.add_argument('-attack_norm',default="L2",type=str)
parser.add_argument('-test_run_num', default=1,type=int)

args = parser.parse_args()
device = torch.device('cuda')

# Load the model
models_dict = {}
for i,path in enumerate(glob.glob(args.model_type_path+"/*")):
    if os.path.isdir(path):
        for sub_path in glob.glob(path+"/*"):
            if "best_model.pth" in sub_path:
                models_dict[i] = {}
                models_dict[i]["model"] = model_loader.load_model(model_architecture=args.network, pretraining=args.pretrained, num_classes=args.nClasses,weights_path=sub_path)
                #models_dict[i]["num_correct"] = 0
                models_dict[i]["top-1"] = 0
                models_dict[i]["top-3"] = 0
                models_dict[i]["top-5"] = 0
                models_dict[i]["base_save_path"] = path + f"/epsilon-{args.epsilon}_steps-{args.num_attack_iters}_run-{args.test_run_num}"
for key in models_dict.keys():
    models_dict[key]["model"] = models_dict[key]["model"].to(device)
    for param in models_dict[key]["model"].parameters():
        param.requires_grad = False
    models_dict[key]["model"].eval()
    models_dict[key]["attack_generator"] = PGD_attack_generator.PGD_Attack_Generator(models_dict[key]["model"],args.epsilon/255.,args.num_attack_iters,args.attack_step_size,args.attack_norm)

# Set up our data loader
test_set_loader = CUB_data_loader.create_test_set_loader(data_split_path=args.data_split_path, data_root_path=args.datasetPath, map_location=args.heatmaps,network=args.network,batch_size=args.batchSize)

# Create destination folder
criterion = nn.CrossEntropyLoss()
batch_corrected_CE = nn.CrossEntropyLoss(reduction="sum")

for batch_idx, (data, cls, imageName, hmap) in enumerate(test_set_loader):
    data = data.to(device)
    cls = cls.to(device)
    for key in models_dict.keys():
        adv_x = models_dict[key]["attack_generator"](data,cls)
        with torch.no_grad():
            outputs = models_dict[key]["model"](adv_x)
            top_k_predictions = torch.topk(outputs,k=5,dim=-1)[1]
            for ground_truth_label,top_predicted_labels in zip(cls,top_k_predictions):
                if ground_truth_label in top_predicted_labels[0]:
                    models_dict[key]["top-1"] = models_dict[key]["top-1"] + 1
                if ground_truth_label in top_predicted_labels[:3]:
                    models_dict[key]["top-3"] = models_dict[key]["top-3"] + 1
                if ground_truth_label in top_predicted_labels[:5]:
                    models_dict[key]["top-5"] = models_dict[key]["top-5"] + 1

        del adv_x,outputs,top_k_predictions


for key in models_dict.keys():
    print(models_dict[key]["base_save_path"])
    print(f"Top-1 Accuracy: {models_dict[key]['top-1']} | Top-3 Accuracy: {models_dict[key]['top-3']} | Top-5 Accuracy: {models_dict[key]['top-5']}")
    with open ("_".join([models_dict[key]["base_save_path"],"top-1.txt"]),"w") as f:
        f.write("%d" % models_dict[key]["top-1"])
    with open ("_".join([models_dict[key]["base_save_path"],"top-3.txt"]),"w") as f:
        f.write("%d" % models_dict[key]["top-3"])
    with open ("_".join([models_dict[key]["base_save_path"],"top-5.txt"]),"w") as f:
        f.write("%d" % models_dict[key]["top-5"])
    print()
