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
parser.add_argument('-attacker_model_type',type=str)
parser.add_argument('-attacker_model_type_path',type=str)
parser.add_argument('-defender_model_type_path',type=str)
parser.add_argument('-network', default= 'densenet',type=str,choices=["resnet","densenet","inception"])
parser.add_argument('-nClasses', default= 200,type=int)
parser.add_argument('-pretrained', action='store',nargs='?',const="IMAGENET1K_V1",default=None)
parser.add_argument('-epsilon', default=8,type=float)
parser.add_argument('-num_attack_iters',default=32,type=int)
parser.add_argument('-attack_step_size',default=1./255,type=float)
parser.add_argument('-attack_norm',default="L-Inf",type=str)
parser.add_argument('-test_run_num', default=1,type=int)

args = parser.parse_args()
device = torch.device('cuda')

# Load the models that are to be used to generate attacks
attacker_models = {}
for i,path in enumerate(glob.glob(args.attacker_model_type_path+"/*")):
    if os.path.isdir(path):
        for sub_path in glob.glob(path+"/*"):
            if "best_model.pth" in sub_path:
                attacker_models[i] = {}
                attacker_models[i]["model"] = model_loader.load_model(model_architecture=args.network, pretraining=args.pretrained, num_classes=args.nClasses,weights_path=sub_path)
for key in attacker_models.keys():
    attacker_models[key]["model"] = attacker_models[key]["model"].to(device)
    for param in attacker_models[key]["model"].parameters():
        param.requires_grad = False
    attacker_models[key]["model"].eval()
    attacker_models[key]["attack_generator"] = PGD_attack_generator.PGD_Attack_Generator(attacker_models[key]["model"],args.epsilon/255.,args.num_attack_iters,args.attack_step_size,args.attack_norm)


# Load the models that the attacks are going to be performed against
# Paths may need to be updated
defender_models = {}
for i,path in enumerate(glob.glob(args.defender_model_type_path+"/*")):
    if os.path.isdir(path):
        for sub_path in glob.glob(path+"/*"):
            if "best_model.pth" in sub_path:
                defender_models[i] = {}
                defender_models[i]["model"] = model_loader.load_model(model_architecture=args.network, pretraining=args.pretrained, num_classes=args.nClasses,weights_path=sub_path)
                defender_models[i]["model"] = defender_models[i]["model"].to(device)
                for param in defender_models[i]["model"].parameters():
                    param.requires_grad = False
                defender_models[i]["model"].eval()
                defender_models[i]["top-1"] = [0 for _ in range(len(list(attacker_models.keys())))]
                defender_models[i]["top-3"] = [0 for _ in range(len(list(attacker_models.keys())))]
                defender_models[i]["top-5"] = [0 for _ in range(len(list(attacker_models.keys())))]
                model_save_path = path.replace("pretrained",args.attacker_model_type)
                os.makedirs(model_save_path,exist_ok=True)
                defender_models[i]["base_save_path"] = model_save_path + f"/epsilon-{args.epsilon}_steps-{args.num_attack_iters}_run-{args.test_run_num}"


# Set up our data loader
test_set_loader = CUB_data_loader.create_test_set_loader(data_split_path=args.data_split_path, data_root_path=args.datasetPath, map_location=args.heatmaps,network=args.network,batch_size=args.batchSize)

# Set up our loss function
criterion = nn.CrossEntropyLoss()
batch_corrected_CE = nn.CrossEntropyLoss(reduction="sum")

for batch_idx, (data, cls, imageName, hmap) in enumerate(test_set_loader):
    data = data.to(device)
    cls = cls.to(device)
    for attacker_num,key in enumerate(attacker_models.keys()):
        adv_x = attacker_models[key]["attack_generator"](data,cls)
        for defender in defender_models.keys():
            with torch.no_grad():
                outputs = defender_models[defender]["model"](adv_x)
                top_k_predictions = torch.topk(outputs,k=5,dim=-1)[1]
                for ground_truth_label,top_predicted_labels in zip(cls,top_k_predictions):
                    if ground_truth_label in top_predicted_labels[0]:
                        defender_models[defender]["top-1"][attacker_num] = defender_models[defender]["top-1"][attacker_num] + 1
                    if ground_truth_label in top_predicted_labels[:3]:
                        defender_models[defender]["top-3"][attacker_num] = defender_models[defender]["top-3"][attacker_num] + 1
                    if ground_truth_label in top_predicted_labels[:5]:
                        defender_models[defender]["top-5"][attacker_num] = defender_models[defender]["top-5"][attacker_num] + 1
                del outputs,top_k_predictions
        del adv_x



for key in defender_models.keys():
    print(defender_models[key]["base_save_path"])
    print(f"Top-1 Accuracy: {defender_models[key]['top-1']} | Top-3 Accuracy: {defender_models[key]['top-3']} | Top-5 Accuracy: {defender_models[key]['top-5']}")
    with open ("_".join([defender_models[key]["base_save_path"],"top-1.txt"]),"w") as f:
        results = ",".join([str(i) for i in defender_models[key]["top-1"]])
        f.write(results)
    with open ("_".join([defender_models[key]["base_save_path"],"top-3.txt"]),"w") as f:
        results = ",".join([str(i) for i in defender_models[key]["top-3"]])
        f.write(results)
    with open ("_".join([defender_models[key]["base_save_path"],"top-5.txt"]),"w") as f:
        results = ",".join([str(i) for i in defender_models[key]["top-5"]])
        f.write(results)
    print()