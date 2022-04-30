#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 11:40:40 2022

@author: ali
"""
import numpy as np
import random
from tqdm import tqdm
from datasets import AnnotParser
from colorstr import colorstr
import torch

class KmeansAnchor(object):
    '''
    K-means clustering on bounding boxes to generate anchors
    
    '''
    def __init__(self, k, max_iter=100,gen=2000, random_seed=None):
        self.k = k
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.anchors = None
        self.labels = None
        self.ious = None
        self.gen = gen

    @staticmethod
    def iou(boxes,anchors):
        '''
        Calculate the IOU between boxes and anchors
        
        input param : boxes, 2-d array, shape (n,2)
        
                      anchors, 2-d array, shape (k,2)
        
        return : iou, 2-d array, shape (n,k)
        
        '''
        #calculate inersection of boxes and anchors
        #expend the boxes_w and boxes_h dimension from shape(n) to shape(n,1)
        #expend the anchors_w and anchors_h dimension from shape(k) to shape(1,k)
        #so w_min & h_min shape is (n,k)
        #inter = w_min * h_min , shape is (n,k)
        w_min = np.minimum(boxes[:,0,np.newaxis],anchors[np.newaxis,:,0])
        h_min = np.minimum(boxes[:,1,np.newaxis],anchors[np.newaxis,:,1])
        inter = w_min * h_min
        #calculate the union
        box_area = boxes[:,0] * boxes[:,1]
        anchor_area = anchors[:,0] * anchors[:,1]
        union = box_area[:,np.newaxis] + anchor_area[np.newaxis]
        
        return inter / ( union - inter )
        
    def fit(self,boxes):
        
        '''
        Run Kmean clustering on input boxes
        
        param : input boxes shape (n,2) , after iou funciton, return shpae is (n,k)
        
        param : output anchors shape (k,2)
        
        '''
        np.random.seed(self.random_seed)
        boxes = np.array(boxes)        
        n = len(boxes)
        #initialize K cluster centers (i.e., K canchors)
        self.anchors = boxes[np.random.choice(n,self.k,replace=True)]
        self.labels = np.zeros((n,))
        
        PREFIX = colorstr('Start Kmeans Anchor: ')
        for i in tqdm(range(self.max_iter),desc=f'{PREFIX}'):
            self.ious = self.iou(boxes, self.anchors)
            distances = 1 - self.ious
            cur_labels = np.argmin(distances, axis=1)
            #if anchors not changed anymore, then break
            if (cur_labels == self.labels).all():
                print("all anchor is not changed, stopped finding anchors")
                break
            # Updata K anchors
            for j in range(self.k):
                self.anchors[j] = np.mean(boxes[cur_labels == j], axis=0)
            self.labels = cur_labels
        
        sort_k = self.anchors[np.argsort(self.anchors.prod(1))]  # sort small to large
        num_iou = 1
        for ks in sort_k:
            print(num_iou,": ",round(ks[0])," ," ,round(ks[1]))
            num_iou = num_iou + 1
            
        print("Start Generic Algorithm...")
        # Evolve
        npr = np.random
        
        PREFIX = colorstr('AutoAnchor: ')
        boxes = torch.tensor(boxes, dtype=torch.float32)  # filtered
        f, sh, mp, s = self.anchor_fitness(self.anchors,boxes), self.anchors.shape, 0.9, 0.1  # fitness, generations, mutation prob, sigma
        pbar = tqdm(range(self.gen), desc=f'{PREFIX}Evolving anchors with Genetic Algorithm:')  # progress bar
        for _ in pbar:
            v = np.ones(sh)
            while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
                #v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
                v = ((npr.random(sh) < mp) * npr.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
            kg = (self.anchors.copy() * v).clip(min=2.0)
            fg = self.anchor_fitness(kg,boxes)
            if fg < f:
                f, k = fg, kg.copy()
                ff = 1 - f
                pbar.desc = f'{PREFIX}Evolving anchors with Genetic Algorithm: fitness = {ff:.4f}'
        print("")
        print("After Generic Algorithm...Anchor Values")
        sort_k = k[np.argsort(k.prod(1))]  # sort small to large
        num_ga = 1
        for ks in sort_k:
            print(num_ga,": ", round(ks[0])," ," ,round(ks[1]))
            num_ga=num_ga+1
            
        for i in range(len(sort_k)):
            self.anchors[i] = sort_k[i]

    def metric(self,k, wh):  # compute metrics
        #r = wh[:, None] / k[None]
        #x = torch.min(r, 1 / r).min(2)[0]  # ratio metric
        x = 1-self.iou(wh, k)  # iou metric
        return x, x.min(1)[0]  # x, best_x

    def anchor_fitness(self,k,wh):  # mutation fitness
        thr = 0.25 #0.25
        _, best = self.metric(torch.tensor(k, dtype=torch.float32), wh)
        return (best * (best < thr).float()).mean()  # fitness
'''
def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)
'''
            
            

        