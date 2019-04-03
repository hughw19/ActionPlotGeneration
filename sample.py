from __future__ import print_function
import numpy as np
import tensorflow as tf
import argparse
import time, cv2, copy
import os, string,json

from six.moves import cPickle
from utils import EventtoReadable
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='ckpts/',
                       help='model directory to store checkpointed models')
    parser.add_argument('--prime_length', type=int, default=5,
                       help='how many frames to use for prime')
    parser.add_argument('--sample_length', type=int, default=100,
                       help='number of characters to sample')
    parser.add_argument('--gpuid', type=int, default=0,
                       help='which gpu to use')
    parser.add_argument('--sample', type=int, default=1,
                       help='0 to use max at each timestep, 1 to sample at each timestep, 2 to sample on spaces')
    parser.add_argument('--n_sample', type=int, default=100,
                       help='how many different sequences to sample?')
    parser.add_argument('--obj_list', type=str, default='book phone bowl bottle cup orange',
                       help='list objects in the scene')
    

    args = parser.parse_args()
    args.obj_list = args.obj_list.split(' ')
    sample(args)

def sample(args):
    foldername = args.save_dir.split('/')[1].strip()
    sample_dir = os.path.join('samples', foldername)
    outputpath = os.path.join(sample_dir, 'output.npy')
    output_txt_dir = os.path.join('outputs', foldername)
    if not os.path.exists(output_txt_dir):
        os.makedirs(output_txt_dir)
    
    
    sample_txt_path = os.path.join(sample_dir, 'sampled_result_0.txt'.format(foldername))
    if not os.path.exists(sample_dir):
        os.mkdir(sample_dir)
    
    if os.path.exists(sample_txt_path):
        index = sample_txt_path[-5]
        index = int(sample_txt_path[-5]) + 1
        sample_txt_path = '{}{}{}'.format(sample_txt_path[:-5],index, sample_txt_path[-4:])
        
    sample_json_path = sample_txt_path[:-4] + '.json'
    
    with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
        saved_args = cPickle.load(f)

    model = Model(saved_args, True)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            #saver.restore(sess, os.path.join(args.save_dir, 'model.ckpt-41000'))
            
            n_events = 0
            n_errors = 0
            error_dict = {}
            
            error_rate = np.zeros((args.n_sample), dtype=np.float32)
            error_num = np.zeros((args.n_sample), dtype=np.int32)
            for i in range(args.n_sample):
                event_seq = model.sample(sess, args.prime_length, args.sample_length, args.sample)
                etr = EventtoReadable(event_seq, sample_txt_path, error_dict, output_txt_dir = output_txt_dir)
                n_events += etr.n_event
                n_errors += etr.n_error
                
                error_num[i] = etr.n_error
                error_rate[i] = float(etr.n_error)/etr.n_event
                
                error_dict = etr.get_error_dict()
        
    
    print('n_events: {}, n_errors: {}, error rate: {}.'.format(n_events, n_errors, float(n_errors)/n_events))
    print('avg error rate: {}, variance: {}, max number of error: {}.'.format(np.mean(error_rate), np.var(error_rate), np.max(error_rate)))
    print(error_dict)
    with open(sample_txt_path, 'a') as output:
        output.write('n_events: {}, n_errors: {}, error rate: {}.'.format(n_events, n_errors, float(n_errors)/n_events))
    with open(sample_json_path, 'wb') as fj:
        json.dump(error_dict, fj)
    print('Finished.')
    
if __name__ == '__main__':
    main()
