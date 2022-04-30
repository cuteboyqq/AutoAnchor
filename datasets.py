#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 14:09:15 2022

@author: ali
"""
import glob
from pathlib import Path
import os
from tqdm import tqdm
import numpy as np
from PIL import Image
from colorstr import colorstr


class AnnotParser(object):
    
    def __init__(self, file_type, stop_num=200000, img_size=320):
        
        assert file_type in ['txt'], "Unsupported file type."
        
        self.type = file_type
        self.stop_num = stop_num
        self.img_size = img_size
        
        
    def parse(self, img_dir):
        
        '''
        parse annotation file, the file type must be txt
        
        input param : image dir
        return : boxes, 2-d array, shape (n,2), 
                 each row represent box,
                 and each column represent width and height
        '''
        if self.type == 'txt':
            return self.parse_txt(img_dir)
        
    def parse_txt(self,img_dir):
        IMG_FORMATS = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # acceptable image suffixes
        f = [] #img file
        prefix=''
        for p in img_dir if isinstance(img_dir,list) else [img_dir]:
            p = Path(img_dir)
            
            if p.is_dir():
                f+=glob.glob( str( p / '**' / '*.*' ),recursive=True)
            elif p.is_file():
                with open(p) as t:
                    t = t.read().strip().splitlines()
                    parent = str(p.parent) + os.sep
                    f += [x.replace('./', parent) if x.startswith('./') else x for x in t]  # local to global path
                    # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
            
            else:
                raise Exception(f'{prefix}{p} does not exist')
                
        
        self.img_files = sorted(x.replace('/', os.sep) for x in f if x.split('.')[-1].lower() in IMG_FORMATS)
        
        self.label_files = img2label_paths(self.img_files)
        
        label_list = []
        PREFIX = colorstr('Generate labels List: ')
        rimg = 0
        c = 1
        for label_file in tqdm(self.label_files,desc=f'{PREFIX}Start Analysis labels ri={rimg}:'):
            c+=1 
            if c==self.stop_num:
                break
            if os.path.isfile(label_file):
                with open(label_file) as f:
                    y = f.read().strip().splitlines()
                    l = [x.split() for x in y if len(y)]
                    l = np.array(l, dtype=np.float16())
                    _, i = np.unique(l, axis=0, return_index=True)
                    #print("_ = ",_)
                    if not len(_)==0:
                        label_list.append(_)   
            else:
                img_path = label2img_path(label_file)
                rimg+=1
                #print("remove img from list :", img_path)
                while(img_path in self.img_files):
                    self.img_files.remove(img_path)
        #==============================================================================================
        e=1
        shapes = []
        #print("Start Analysis Train image shapes...")
        PREFIX = colorstr('Generate Image shapes List:')
        for im_file in tqdm(self.img_files,desc =f'{PREFIX}Start Analysis Train image shapes:'):
            im = Image.open(im_file)
            im.verify()  # PIL verify
            shape = im.size  # image size
            shapes.append(shape)
            e+=1
            if e==self.stop_num:
                break
        #for shape in shapes:
            #print(shape)
            
        shapes = np.array(shapes, dtype=np.float64)
        shapes = self.img_size * shapes / shapes.max(1, keepdims=True)
        #for shape in shapes:
            #print(shape)
        #=========================================================================================================== 
        print("shapes size =",len(shapes))
        print("label_list size = ",len(label_list))  
        print("Start filter labels list : wh  And multiple s , which s = shapes = img_size * shapes / shapes.max(1, keepdims=True) ")    
        boxes = np.concatenate([l[:, 3:5]*s for s,l in zip(shapes, label_list)])  # wh
        print("filter < 2 pixels")
        boxes = boxes[(boxes >= 2.0).any(1)]  # filter > 2 pixels
        return boxes    

def img2label_paths(img_paths):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return [sb.join(x.rsplit(sa, 1)).rsplit('.', 1)[0] + '.txt' for x in img_paths]


def label2img_path(label_path):
    # Define label paths as a function of image paths
    sa, sb = os.sep + 'images' + os.sep, os.sep + 'labels' + os.sep  # /images/, /labels/ substrings
    return sa.join(label_path.rsplit(sb, 1)).rsplit('.', 1)[0] + '.jpg'