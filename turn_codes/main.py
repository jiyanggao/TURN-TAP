import tensorflow as tf
import numpy as np
import turn_model
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import vs_multilayer
import operator
import os 

ctx_num=4
unit_size=16.0
unit_feature_size=2048
lr=0.005
lambda_reg=2.0
batch_size=128
test_steps=4000

def get_pooling_feature(feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    current_pos=start
    while current_pos<end:
        swin_start=current_pos
        swin_end=swin_start+swin_step
        feat=np.load(feat_dir+movie_name+".mp4"+"_"+str(swin_start)+"_"+str(swin_end)+".npy")
        all_feat=np.vstack((all_feat,feat))
        current_pos+=swin_step
    pool_feat=np.mean(all_feat,axis=0)
    return pool_feat


def get_left_context_feature(feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    count=0
    current_pos=start
    context_ext=False
    while  count<ctx_num:
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
        pool_feat=np.zeros([unit_feature_size],dtype=np.float32)
    return np.reshape(pool_feat,[unit_feature_size])


def get_right_context_feature(feat_dir,movie_name,start,end):
    swin_step=unit_size
    all_feat=np.zeros([0,unit_feature_size],dtype=np.float32)
    count=0
    current_pos=end
    context_ext=False
    while  count<ctx_num:
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
        pool_feat=np.zeros([unit_feature_size],dtype=np.float32)
    return np.reshape(pool_feat,[unit_feature_size])


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

# test 
def do_eval_slidingclips(sess,vs_eval_op,model,movie_length_info,iter_step):
    results_lst=[]
    for k,test_sample in enumerate(model.test_set.test_samples):
        if k%1000==0:
            print str(k)+"/"+str(len(model.test_set.test_samples))
        movie_name=test_sample[0]
        movie_length=movie_length_info[movie_name]
        clip_start=test_sample[1]
        clip_end=test_sample[2]
        featmap=get_pooling_feature(model.test_set.feat_dir,movie_name,clip_start,clip_end)
        left_feat=get_left_context_feature(model.test_set.feat_dir,movie_name,clip_start,clip_end)
        right_feat=get_right_context_feature(model.test_set.feat_dir,movie_name,clip_start,clip_end)
        feat=np.hstack((left_feat,featmap,right_feat))
        feat=np.reshape(feat,[1,unit_feature_size*3])
        
        feed_dict = {
            model.visual_featmap_ph_test: feat
            }
        
        outputs=sess.run(vs_eval_op,feed_dict=feed_dict) 
        reg_end=clip_end+outputs[3]*unit_size
        reg_start=clip_start+outputs[2]*unit_size
        round_reg_end=clip_end+np.round(outputs[3])*unit_size
        round_reg_start=clip_start+np.round(outputs[2])*unit_size
        softmax_score=softmax(outputs[0:2])
        action_score=softmax_score[1] 
        results_lst.append((movie_name,round_reg_start,round_reg_end,reg_start,reg_end,action_score,outputs[0],outputs[1]))
    pickle.dump(results_lst,open("./test_results/results_TURN_flow_iter"+str(iter_step)+".pkl","w")) 


def run_training():
    initial_steps=0
    max_steps=30000
    train_clip_path="./val_training_samples.txt"
    background_path="./background_samples.txt"
    train_featmap_dir="./path_to_features_val/"
    test_featmap_dir="./path_to_features_test/"
    test_clip_path="./test_swin.txt"
    test_video_length_info={}
    with open("./thumos14_video_length_test.txt") as f:
        for l in f:
            test_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])
    train_video_length_info={}
    with open("./thumos14_video_length_val.txt") as f:
        for l in f:
            train_video_length_info[l.rstrip().split(" ")[0]]=int(l.rstrip().split(" ")[2])

    model=turn_model.TURN_Model(batch_size,train_video_length_info,ctx_num,unit_feature_size,unit_size,
        lambda_reg,lr,train_clip_path,background_path,test_clip_path,train_featmap_dir,test_featmap_dir)

    with tf.Graph().as_default():
		
        loss_cls_reg,vs_train_op,vs_eval_op, loss_reg=model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)
        sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train_reg()

            _, loss_v, loss_reg_v = sess.run([vs_train_op,loss_cls_reg, loss_reg], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: total loss = %.2f, regression loss = %.2f(%.3f sec)' % (step, loss_v, loss_reg_v, duration)) 

            if (step+1) % test_steps == 0:
                print "Start to test:-----------------\n"
                do_eval_slidingclips(sess,vs_eval_op,model,test_video_length_info,step+1)

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



