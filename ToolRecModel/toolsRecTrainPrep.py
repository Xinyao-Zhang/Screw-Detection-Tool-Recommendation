## Tool recommendation model for ICSHM2021
## Developed By Kareem Eltouny - University at Buffalo
## Part of: 
## Zhang, X., Eltouny, K., Liang, X., & Behdad, S. (2023)
## Automatic Screw Detection and Tool Recommendation System for Robotic Disassembly. 
## Journal of Manufacturing Science and Engineering, 145(3), 031008.
## 
## 11/15/2021

import OsUtils as OsU
import os
import csv
from PIL import Image

import argparse


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('-d', '--dir', type=str, default='train/', help='Source image directory')
parser.add_argument('--cropSize', nargs=2, type=int, default=[128,128], help='Cropped screw image size [width, height]')
parser.add_argument('--saveScrew', type=str, default='screws_extracts/all/', help='Saving directory for screw crops')
#parser.add_argument('--saveCrops', action='store_true', help='Whether to save extracted screws crops')
parser.add_argument('-v', '--verbose', action='store_true', help='Verbose for screws extraction')
parser.add_argument('--absolute', action='store_true', help='Flag if the cropping coordinates are absolute (default is relative)')
parser.add_argument('--datasize', type=int, default=80, help='Maximum number of images per class')
parser.add_argument('--classes', nargs='+', type=str, default=["Torx security", "Phillips", "Pentalobe"], help='Available classes')


args = parser.parse_args()

new_size = (args.cropSize[1], args.cropSize[0])
verbose = args.verbose

#save_screw_crops = args.saveCrops


path_to_screws = args.saveScrew

path_to_origin = args.dir

relative = not args.absolute
data_size = args.datasize
class_type = args.classes


os.makedirs(path_to_screws[:-1], exist_ok=True)
OsU.wipe_dir(path_to_screws[:-1])


from screwCropper import screwCropper

Cropper = screwCropper(new_size=new_size, relative=relative)

# Crop screw images for training

if os.path.exists(path_to_screws + 'screws_dict.csv'):
  os.remove(path_to_screws + 'screws_dict.csv')

with open(path_to_screws + 'screws_dict.csv', 'w', newline='') as csvfile:
    fieldnames = ['Class', 'Image', 'Screws']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    
    k = 1
        
    for i_class in class_type:
        origin_class = path_to_origin+i_class+"/"
        screws_class = path_to_screws+i_class+"/"
        os.makedirs(screws_class[:-1], exist_ok=True)
        OsU.wipe_dir(screws_class[:-1])
        
        all_files = []
        all_files_raw = os.listdir(origin_class)
        for file in all_files_raw:
            if file.endswith(".JPG"):
                file = file[:-4]
                if int(file)<=data_size:
                    all_files.append(file)
                    data = Cropper.crop_screw(origin_class+file, verbose=verbose)
                    screws_list = []
                    for i in range(len(data)):
                        img = Image.fromarray(data[i])
                        img.save(screws_class+str(k)+".JPG")
                        screws_list.append(k)
                        k += 1
                    writer.writerow({'Class': i_class, 'Image': file, 'Screws': screws_list})
                        

