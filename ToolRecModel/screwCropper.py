## Tool recommendation model for ICSHM2021
## Developed By Kareem Eltouny - University at Buffalo
## Part of: 
## Zhang, X., Eltouny, K., Liang, X., & Behdad, S. (2023)
## Automatic Screw Detection and Tool Recommendation System for Robotic Disassembly. 
## Journal of Manufacturing Science and Engineering, 145(3), 031008.
## 
## 11/15/2021

import numpy as np
from PIL import Image, ExifTags
import PIL
import copy


class screwCropper:
    def __init__(self,
                 new_size=(224,224),
                 relative=True,
                 center_label=True,
                 first_col_label=True,
                 return_label=False):
                 
        self.new_size = new_size
        self.relative = relative
        self.center_label = center_label
        
        if first_col_label:
            self.l_col = 0
            self.return_label = return_label
        else:
            self.l_col = -1
            self.return_label = False
                 


    def crop_screw(self, path, verbose=False):
    
    
        xc = self.l_col+1
        yc = self.l_col+2
        wc = self.l_col+3
        hc = self.l_col+4

        path_to_jpg = path + '.jpg'
        path_to_txt = path + '.txt'

        img = Image.open(path_to_jpg) # 0 to 255
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = img._getexif()
        if exif[orientation] == 3:
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img=img.rotate(90, expand=True)
            
        img_label = np.loadtxt(path_to_txt)

        w_img, h_img = img.size

        if self.relative:
            #print(f'relative: {self.relative}')
            img_label[:,(yc,hc)] = img_label[:,(yc,hc)]*h_img
            img_label[:,(xc,wc)] = img_label[:,(xc,wc)]*w_img
            
        img_label = img_label.astype('int')

        data = np.zeros((len(img_label),self.new_size[0],self.new_size[1],3),dtype='uint8')
        label = np.zeros((len(img_label)),dtype='int')

        if self.center_label:
            #print(f'center_label: {self.center_label}')
            for i in range(len(img_label)):
                area = 1.2*img_label[i,wc]*img_label[i,hc]
                w_new = int(np.sqrt(area))
                x1 = img_label[i,xc]-w_new//2
                y1 = img_label[i,yc]-w_new//2
                x2 = x1 + w_new
                y2 = y1 + w_new
                #print(f'image: {path}, screw {i}, x1 {x1}, y1 {y1}, x2 {x2}, y2 {y2}')
                
                screw_img = img.crop((x1,y1,x2,y2))
                screw_img = np.array(screw_img.resize(self.new_size,resample=PIL.Image.LANCZOS))
                data[i,:,:,:] = copy.deepcopy(screw_img)

                if self.return_label:
                    label[i] = img_label[i,0]

                if verbose:
                    print(f'Class {img_label[i,0]}, Image {path[-2:]}, cropped screw {i} with {w_new} square ')

        else:
            for i in range(len(img_label)):
                area = 1.2*img_label[i,wc]*img_label[i,hc]
                w_new = int(np.sqrt(area))
                x1 = img_label[i,xc]
                y1 = img_label[i,yc]
                x2 = x1 + w_new
                y2 = y1 + w_new

                screw_img = img.crop((x1,y1,x2,y2))
                screw_img = np.array(screw_img.resize(self.new_size,resample=PIL.Image.LANCZOS))
                data[i,:,:,:] = copy.deepcopy(screw_img)

                if self.return_label:
                    label[i] = img_label[i,0]

                if verbose:
                    if self.l_col == 0:
                        print(f'Class {img_label[i,0]}, Image {path[-2:]}, cropped screw {i} with {w_new} square ')
                    else:
                        print(f'Image {path[-2:]}, cropped screw {i} with {w_new} square ')

        if self.return_label:
            return data, label
        else:
            return data
                
    
    
class screwCropperEval:
    def __init__(self,
                 new_size=(224,224),
                 relative=True,
                 center_label=True,
                 first_col_label=True,
                 return_label=False):
                 
        self.new_size = new_size
        self.relative = relative
        self.center_label = center_label

        if first_col_label:
            self.l_col = 0
            self.return_label = return_label
        else:
            self.l_col = -1
            self.return_label = False
            
            
    def crop_screw(self, path, verbose=False):
    
        xc = self.l_col+1
        yc = self.l_col+2
        wc = self.l_col+3
        hc = self.l_col+4

        path_to_jpg = path + '.jpg'
        path_to_txt = path + '.txt'

        img = Image.open(path_to_jpg) # 0 to 255
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation]=='Orientation':
                break
        exif = img._getexif()
        if exif[orientation] == 3:
            img=img.rotate(180, expand=True)
        elif exif[orientation] == 6:
            img=img.rotate(270, expand=True)
        elif exif[orientation] == 8:
            img=img.rotate(90, expand=True)
            
        img_label = np.loadtxt(path_to_txt)

        w_img, h_img = img.size
        
        if self.relative:
            img_label[:,(yc,hc)] = img_label[:,(yc,hc)]*h_img
            img_label[:,(xc,wc)] = img_label[:,(xc,wc)]*w_img
            
        img_label = img_label.astype('int')

        data = np.zeros((len(img_label),self.new_size[0],self.new_size[1],3),dtype='uint8')
        label = np.zeros((len(img_label)),dtype='int')

        if self.center_label:
            for i in range(len(img_label)):
                area = 1.2*img_label[i,wc]*img_label[i,hc]
                w_new = int(np.sqrt(area))
                x1 = img_label[i,xc]-w_new//2
                y1 = img_label[i,yc]-w_new//2
                x2 = x1 + w_new
                y2 = y1 + w_new
        else:
            for i in range(len(img_label)):
                area = 1.2*img_label[i,wc]*img_label[i,hc]
                w_new = int(np.sqrt(area))
                x1 = img_label[i,xc]
                y1 = img_label[i,yc]
                x2 = x1 + w_new
                y2 = y1 + w_new

                screw_img = img.crop((x1,y1,x2,y2))
                screw_img = np.array(screw_img.resize(self.new_size,resample=PIL.Image.LANCZOS))
                data[i,:,:,:] = copy.deepcopy(screw_img)


                if verbose:
                    print(f'File {path[-2:]}, cropped screw {i} with {w_new} square ')

        return data
                
    
    
