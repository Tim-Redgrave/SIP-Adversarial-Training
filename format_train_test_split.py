# Connect segmentation masks to images in a convenient way
import os
import sys
import csv
sys.path.append("../")

with open("../data/train_test_split.txt", "r") as f:
    unformatted_split = f.readlines()

with open("../data/images.txt", "r") as f:
    image_num_to_image_name_map = f.readlines()

with open("../data/image_class_labels.txt", "r") as f:
    image_num_to_image_label_map = f.readlines()

formatted_split = []
for i in range(len(unformatted_split)):
    train_test = unformatted_split[i].split()[1]
    image_name = image_num_to_image_name_map[i].split()[1]
    image_label = image_num_to_image_label_map[i].split()[1]
    formatted_split.append([train_test,image_label,image_name])

with open("../data/formatted_train_test_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(formatted_split)
