
import numpy as np
from math import sqrt
import os
import random
import pickle

def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
A class that handles the training set
'''
class TrainingDataSet(object):
    def __init__(self,feat_dir,clip_gt_path,background_path,batch_size,movie_length_info,ctx_num,unit_feature_size,unit_size):
        #it_path: image_token_file path
        self.ctx_num=ctx_num
        self.unit_feature_size=unit_feature_size
        self.unit_size=unit_size
        self.batch_size=batch_size
        self.movie_length_info=movie_length_info
        self.visual_feature_dim=self.unit_feature_size*3
        self.feat_dir=feat_dir
        self.training_samples=[]

        print "Reading training data list from "+clip_gt_path+" and "+background_path
        with open(clip_gt_path) as f:
            for l in f:
                movie_name=l.rstrip().split(" ")[0]
                clip_start=float(l.rstrip().split(" ")[1])
                clip_end=float(l.rstrip().split(" ")[2])
                gt_start=float(l.rstrip().split(" ")[3])
                gt_end=float(l.rstrip().split(" ")[4])
                round_gt_start=np.round(gt_start/unit_size)*self.unit_size+1
                round_gt_end=np.round(gt_end/unit_size)*self.unit_size+1
                self.training_samples.append((movie_name,clip_start,clip_end,gt_start,gt_end,round_gt_start,round_gt_end,1))
        print str(len(self.training_samples))+" training samples are read"
        positive_num=len(self.training_samples)*1.0
        with open(background_path) as f:
            for l in f:
                # control the ratio between  background samples and positive samples to be 10:1
                if random.random()>10.0*positive_num/270000: continue
                movie_name=l.rstrip().split(" ")[0]
                clip_start=float(l.rstrip().split(" ")[1])
                clip_end=float(l.rstrip().split(" ")[2])
                self.training_samples.append((movie_name,clip_start,clip_end,0,0,0,0,0))
        self.num_samples=len(self.training_samples)
        print str(len(self.training_samples))+" training samples are read"

    def calculate_regoffset(self,clip_start,clip_end,round_gt_start,round_gt_end):
        start_offset=(round_gt_start-clip_start)/self.unit_size
        end_offset=(round_gt_end-clip_end)/self.unit_size
        return start_offset, end_offset

    '''
    Get the central features
    '''    
    def get_pooling_feature(self,feat_dir,movie_name,start,end):
        swin_step=self.unit_size
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        current_pos=start
        while current_pos<end:
            swin_start=current_pos
            swin_end=swin_start+swin_step
            feat=np.load(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
            all_feat=np.vstack((all_feat,feat))
            current_pos+=swin_step
        pool_feat=np.mean(all_feat,axis=0)
        return pool_feat


    '''
    Get the past (on the left of the central unit) context features
    '''
    def get_left_context_feature(self,feat_dir,movie_name,start,end):
        swin_step=self.unit_size
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        count=0
        current_pos=start
        context_ext=False
        while  count<self.ctx_num:
            swin_start=current_pos-swin_step
            swin_end=current_pos
            if os.path.exists(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
                feat=np.load(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                all_feat=np.vstack((all_feat,feat))
                context_ext=True
            current_pos-=swin_step
            count+=1
        if context_ext:
            pool_feat=np.mean(all_feat,axis=0)
        else:
        #    print "no left "+str(start)
            pool_feat=np.zeros([self.unit_feature_size],dtype=np.float32)
        #print pool_feat.shape
        return pool_feat


    '''
    Get the future (on the right of the central unit) context features
    ''' 
    def get_right_context_feature(self,feat_dir,movie_name,start,end):
        swin_step=self.unit_size
        all_feat=np.zeros([0,self.unit_feature_size],dtype=np.float32)
        count=0
        current_pos=end
        context_ext=False
        while  count<self.ctx_num:
            swin_start=current_pos
            swin_end=current_pos+swin_step
            if os.path.exists(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy"):
                feat=np.load(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
                all_feat=np.vstack((all_feat,feat))
                context_ext=True
            current_pos+=swin_step
            count+=1
        if context_ext:
            pool_feat=np.mean(all_feat,axis=0)
        else:
        #    print "no right "+str(end)
            pool_feat=np.zeros([self.unit_feature_size],dtype=np.float32)
        #print pool_feat.shape
        return pool_feat

    def next_batch(self):

        random_batch_index=random.sample(range(self.num_samples),self.batch_size)
        image_batch=np.zeros([self.batch_size,self.visual_feature_dim])
        label_batch=np.zeros([self.batch_size],dtype=np.int32)
        offset_batch=np.zeros([self.batch_size,2],dtype=np.float32)
        index=0
        while index < self.batch_size:
            k=random_batch_index[index]
            movie_name=self.training_samples[k][0]
            if self.training_samples[k][7]==1:
                clip_start=self.training_samples[k][1]
                clip_end=self.training_samples[k][2]
                round_gt_start=self.training_samples[k][5]
                round_gt_end=self.training_samples[k][6]
                start_offset,end_offset=self.calculate_regoffset(clip_start,clip_end,round_gt_start,round_gt_end) 
                featmap=self.get_pooling_feature(self.feat_dir,movie_name,clip_start,clip_end)
                left_feat=self.get_left_context_feature(self.feat_dir,movie_name,clip_start,clip_end)
                right_feat=self.get_right_context_feature(self.feat_dir,movie_name,clip_start,clip_end)
                image_batch[index,:]=np.hstack((left_feat,featmap,right_feat))
                label_batch[index]=1
                offset_batch[index,0]=start_offset
                offset_batch[index,1]=end_offset
                #print str(clip_start)+" "+str(clip_end)+" "+str(round_gt_start)+" "+str(round_gt_end)+" "+str(start_offset)+" "+str(end_offset)
                index+=1
            else:
                clip_start=self.training_samples[k][1]
                clip_end=self.training_samples[k][2]
                left_feat=self.get_left_context_feature(self.feat_dir,movie_name,clip_start,clip_end)
                right_feat=self.get_right_context_feature(self.feat_dir,movie_name,clip_start,clip_end)
                featmap=self.get_pooling_feature(self.feat_dir,movie_name,clip_start,clip_end)
                image_batch[index,:]=np.hstack((left_feat,featmap,right_feat))
                label_batch[index]=0
                offset_batch[index,0]=0
                offset_batch[index,1]=0
                index+=1  
        
        return image_batch, label_batch,offset_batch


'''
A class that handles the test set
'''
class TestingDataSet(object):
    def __init__(self,feat_dir,test_clip_path,batch_size,ctx_num):
        self.ctx_num=ctx_num
        #il_path: image_label_file path
        self.batch_size=batch_size
        self.feat_dir=feat_dir
        print "Reading testing data list from "+test_clip_path
        self.test_samples=[]
        with open(test_clip_path) as f:
            for l in f:
                movie_name=l.rstrip().split(" ")[0]
                clip_start=float(l.rstrip().split(" ")[1])
                clip_end=float(l.rstrip().split(" ")[2])
                self.test_samples.append((movie_name,clip_start,clip_end))
        self.num_samples=len(self.test_samples)
        print "test clips number: "+str(len(self.test_samples))
        



