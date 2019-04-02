from __future__ import print_function
import numpy as np
import tensorflow as tf

import argparse
import time, string
import os
from six.moves import cPickle

from utils import DataLoader
from model import Model
from utils import DataLoaderForInfer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/all_multivideos',
                       help='data directory containing input csv files')
    parser.add_argument('--save_dir', type=str, default='ckpts/',
                       help='directory to store checkpointed models')
    parser.add_argument('--rnn_size', type=int, default=16,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=1,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=10,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=2000,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=0.002,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.999,
                       help='decay rate for rmsprop')
    
    parser.add_argument('--mode', type=str, default='train',
                       help='train/eval')
    
    args = parser.parse_args()
    if args.mode == 'train':
        train(args)
    else:
        eval(args)

def train(args): 
    data_loader = DataLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size_act = data_loader.vocab_size_act
    args.vocab_size_ho = data_loader.vocab_size_ho
    args.vocab_size_aho = data_loader.vocab_size_aho
    x, y = data_loader.next_batch()    
    
    foldername = string.split(args.data_dir,'/')[1].strip()
    foldername = '{}_r{}_s{}_b{}_gru_multitimes'.format(foldername, args.rnn_size, args.seq_length, args.batch_size)
    
    # create save folder with same folder name to data under folder save
    if args.save_dir == 'save':
        args.save_dir = os.path.join(args.save_dir, foldername)
    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
        
    print(args.save_dir)
    
    # create configuration file and vocabulary file
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        cPickle.dump(args, f)
        
    model = Model(args)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        for e in range(args.num_epochs):
            sess.run(tf.assign(model.learning_rate, args.learning_rate * (args.decay_rate ** e)))
            data_loader.reset_batch_pointer()
            #state = sess.run(model.initial_state)
            for b in range(data_loader.num_batches):
                start  = time.time()
                x, y   = data_loader.next_batch()
                x_act  =  x[:,:, 0]
                x_ho   =  x[:,:, 1]
                x_time = x[:,:, 5]
                y_aho  =  y[:,:, 2]
                y_time = y[:,:,5]

                feed = {model.input_act: x_act, model.input_ho: x_ho,model.input_time: x_time,\
                        model.target_aho: y_aho, model.target_time: y_time}
                #for i, (c, h) in enumerate(model.initial_state):
                #    feed[c] = state[i].c
                #    feed[h] = state[i].h
                train_loss, aho_loss, time_loss, state, _ = \
                        sess.run([model.mean_loss,model.mean_aho_loss,model.mean_time_loss,
                                  model.final_state, model.train_op], feed)
                end = time.time()
                print("{}/{} (epoch {}): train_loss = {:.3f}, time/batch = {:.3f}, aho_loss = {:.3f}, time_loss = {:.3f}." \
                    .format(e * data_loader.num_batches + b,
                            args.num_epochs * data_loader.num_batches,
                            e, train_loss, end - start, aho_loss, time_loss))
                
                if (e * data_loader.num_batches + b) % args.save_every == 0\
                    or (e==args.num_epochs-1 and b == data_loader.num_batches-1): # save for the last result
                    checkpoint_path = os.path.join(args.save_dir, 'model.ckpt')
                    saver.save(sess, checkpoint_path, global_step = e * data_loader.num_batches + b)
                    print("model saved to {}".format(checkpoint_path))


def eval(args): 
    data_loader = DataLoaderForInfer(args.data_dir)
    args.vocab_size_act = data_loader.vocab_size_act
    args.vocab_size_ho  = data_loader.vocab_size_ho
    args.vocab_size_aho = data_loader.vocab_size_aho
    x, y = data_loader.next_batch()    
    
    foldername = string.split(args.data_dir,'/')[1].strip()
    foldername = '{}_r{}_s{}_b{}'.format(foldername, args.rnn_size, args.seq_length, args.batch_size)
    save_dir = os.path.join(args.save_dir, foldername)
        
    model = Model(args, True)
    
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True

    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        
                
        data_loader.reset_batch_pointer()
        #state = sess.run(model.initial_state)
        train_loss_list = []
        action_acc_list = []
        time_abs_error_list = []
        time_relative_error_list = []
        state = sess.run(model.cell.zero_state(1, tf.float32))## here 1 is batch_size


        for b in range(data_loader.num_batches):
            start  = time.time()
            x, y   = data_loader.next_batch()
            x_act  =  x[:,:, 0]
            x_ho   =  x[:,:, 1]
            x_time = x[:,:, 5]
            y_aho  =  y[:,:, 2]
            y_time = y[:,:,5]

            feed = {model.input_act: x_act, model.input_ho: x_ho,model.input_time: x_time,\
                    model.target_aho: y_aho, model.target_time: y_time, model.initial_state:state}

            test_loss, action_acc, mean_time_loss, time_relative_error, state = \
                    sess.run([model.mean_loss, model.action_acc, model.mean_time_loss, model.time_relative_error,
                              model.final_state], feed)

            test_loss_list.append(test_loss)
            action_acc_list.append(action_acc)
            time_abs_error_list.append(mean_time_loss)
            time_relative_error_list.append(time_relative_error)


            end = time.time()

        test_loss_list = np.array(test_loss_list)
        action_acc_list = np.array(action_acc_list)
        time_abs_error_list = np.array(time_abs_error_list)
        time_relative_error_list = np.array(time_relative_error_list)

        avg_test_loss = np.mean(test_loss_list)
        avg_action_acc = np.mean(action_acc_list)
        avg_time_abs_error = np.mean(time_abs_error_list)
        avg_time_relative_error = np.mean(time_relative_error_list)
        
        print('avg test loss: {}, avg action acc: {}. avg time abs error: {}, avg time relative error: {}'.format(avg_test_loss, avg_action_acc, avg_time_abs_error, avg_time_relative_error))
            


if __name__ == '__main__':
    main()
