## Tool recommendation model for ICSHM2021
## Developed By Kareem Eltouny - University at Buffalo
## Part of: 
## Zhang, X., Eltouny, K., Liang, X., & Behdad, S. (2023)
## Automatic Screw Detection and Tool Recommendation System for Robotic Disassembly. 
## Journal of Manufacturing Science and Engineering, 145(3), 031008.
## 
## 11/15/2021

import numpy as np
import OsUtils as OsU
import os
from shutil import copy2

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--dir', type=str, default='screws_extracts/', help='Saving directory for screw crops')
parser.add_argument('--allDir', type=str, default='all', help='Directory containing all available data in --dir')
parser.add_argument('--classes', nargs='+', type=str, default=["Torx security", "Phillips", "Pentalobe", "none"], help='Available classes')
parser.add_argument('--valSplit', type=float, default=0.1, help='Ratio of validation data to split from all available data')


args = parser.parse_args()


path_to_screws = args.dir
class_type = args.classes
val_split = args.valSplit
all_dir = args.allDir

np.random.seed(0)


for i_class in class_type:
    all_dir_class = path_to_screws+all_dir+"/"+i_class+"/"
    train_dir_class = path_to_screws+"train/"+i_class+"/"
    val_dir_class = path_to_screws+"validation/"+i_class+"/"
    
    os.makedirs(train_dir_class[:-1], exist_ok=True)
    OsU.wipe_dir(train_dir_class[:-1])
    os.makedirs(val_dir_class[:-1], exist_ok=True)
    OsU.wipe_dir(val_dir_class[:-1])
    
    files = os.listdir(all_dir_class)
    indices = [int(os.path.splitext(x)[0]) for x in files]
    class_size = len(files)

    np.random.shuffle(indices)

    train_indices = indices[:int(class_size*(1-val_split))]
    val_indices = indices[int(class_size*(1-val_split)):]

    csv_train_name = path_to_screws + i_class + '_train_dict.csv'
    csv_val_name = path_to_screws + i_class + '_val_dict.csv'
    
    np.savetxt(csv_train_name, np.array(train_indices), fmt='%i')
    np.savetxt(csv_val_name, np.array(val_indices), fmt='%i')
    
    for i in train_indices:
        copy2(all_dir_class + str(i) + ".JPG", train_dir_class)
    for i in val_indices:
        copy2(all_dir_class + str(i) + ".JPG", val_dir_class)