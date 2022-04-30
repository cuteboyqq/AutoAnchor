#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:38:11 2022

@author: ali
"""

from Kmeans import KmeansAnchor
from datasets import AnnotParser
from colorstr import colorstr

def get_args():
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-imgdir','--img-dir',help='images dir',default=r'C:\bdd100k_supersmall_WPI\images\train')
    parser.add_argument('-numanchors','--num-anchors',type=int,default=9)
    parser.add_argument('-iter','--iter',type=int,default=400,help='kmeans max iters')
    parser.add_argument('-imgsize','--img-size',type=int,default=448,help='train img input size')
    parser.add_argument('-stopnum','--stop-num',type=int,default=200000,help='analysis max num of images')
    parser.add_argument('-gennum','--gen-num',type=int,default=1500,help='num iter of generic algorithm (GA)')
    
    return parser.parse_args()
    


if __name__=="__main__":
    
    ''' get parameters from console'''
    args = get_args()
    ''' assign parameter'''
    img_dir = args.img_dir
    stop_num = args.stop_num
    anchor_num = args.num_anchors
    K_means_max_iter = args.iter
    img_size = args.img_size
    gen = args.gen_num
     
    print("image dir :",img_dir)
    print("anchor_num :",anchor_num)
    print("K_means_max_iter :",K_means_max_iter)
    print("img_size :",img_size)
    print("stop_num :",stop_num)
    print("gen num:",gen)
    file_type = 'txt'
   
    data = AnnotParser(file_type,stop_num,img_size)
    boxes = data.parse(img_dir)
        
    model = KmeansAnchor(anchor_num,K_means_max_iter,gen)
    
    model.fit(boxes)
    
    anchors = model.anchors
    print("\n")
    for i in range(anchor_num):
        print(anchors[i])