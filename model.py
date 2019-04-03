import tensorflow as tf
import os
from utils import *
import numpy as np
import tflearn


class Model():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        # specify the structure of the RNN cell
        cell      = tf.nn.rnn_cell.GRUCell(args.rnn_size)
        cell      = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers)
        self.cell = cell
        
        # placeholders for input and output
        self.input_act   = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.input_ho    = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.input_time  = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])
        self.target_aho  = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.target_time = tf.placeholder(tf.float32, [args.batch_size, args.seq_length])
    
        # initialize the state
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('act_embedding'):
            act_embedding_size = args.rnn_size
            act_embedding = tf.get_variable("embedding", [args.vocab_size_act, act_embedding_size])
            act_embed = tf.nn.embedding_lookup(act_embedding, self.input_act)

        with tf.variable_scope('ho_embedding'):
            ho_embedding_size = args.rnn_size
            ho_embedding = tf.get_variable("embedding", [args.vocab_size_ho, ho_embedding_size])
            ho_embed = tf.nn.embedding_lookup(ho_embedding, self.input_ho)
        
        with tf.variable_scope('time_embedding'):
            time_reshape   = tf.reshape(self.input_time, [-1, 1])
            time_embedding_size = args.rnn_size/2
            time_embed = tflearn.fully_connected(time_reshape, time_embedding_size, activation='linear')
            time_embed = tflearn.fully_connected(time_embed, time_embedding_size, activation='relu')
            time_embed = tflearn.fully_connected(time_embed, time_embedding_size, activation='linear')
            time_embed = tf.reshape(time_embed, [args.batch_size, args.seq_length, -1])
            
        
        # self.input_data still maintains the dimension of rnn_size on the dimension 2
        self.act_data = act_embed
        self.ho_data  = ho_embed
        self.time_data = time_embed
        
        #act_inputs = tf.unstack(self.act_data, axis = 1)        
        act_inputs = self.act_data
        outputs, final_state = tf.nn.dynamic_rnn(cell, act_inputs, initial_state=self.initial_state)
        self.final_state = final_state

        with tf.name_scope('flatten_rnn_ouputs'):
            # Flatten the outputs/inputs into one dimension.
            flat_outputs = tf.reshape(tf.concat(outputs, axis=1), [-1, args.rnn_size])
        
        with tf.variable_scope('rnn_output'):
            self.rnn_output = tf.reshape(flat_outputs, [args.batch_size, args.seq_length, args.rnn_size])
        
        with tf.variable_scope('fc_output'):
            final_size = int(time_embedding_size + ho_embedding_size + args.rnn_size)
            concat = tf.concat([self.rnn_output, self.ho_data, self.time_data], axis=2)
            concat = tf.reshape(concat, [-1, final_size])
            #concat = tflearn.fully_connected(concat, final_size, activation='linear')
            #concat = tflearn.fully_connected(concat, final_size, activation='relu')
                        
            self.fc_output_aho = tflearn.fully_connected(concat, args.vocab_size_aho, activation='linear')
            self.probs_aho = tf.nn.softmax(self.fc_output_aho)
            #shape: [args.batch_size*args.seq_length, args.vocab_size_aho]
            self.fc_output_time = tflearn.fully_connected(concat, args.vocab_size_aho, activation='softplus')
            #self.time_output = tf.reshape(self.fc_output_time, [args.batch_size, args.seq_length, args.vocab_size_aho])
            
            
        with tf.name_scope('flatten_targets'):
            # Flatten the targets too.
            flat_target_aho = tf.reshape(self.target_aho, [-1])

        with tf.name_scope('loss'):
            # loss on label component: Compute mean cross entropy loss for each output.
            aho_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=flat_target_aho, logits=self.fc_output_aho)
            self.mean_aho_loss = tf.reduce_mean(aho_loss)
            
            indices = tf.range(tf.shape(self.fc_output_time)[0])
            indices = tf.stack([indices, flat_target_aho], axis=1)
            
            self.time_loss_output = tf.gather_nd(self.fc_output_time, indices)
            self.time_loss_output = tf.reshape(self.time_loss_output, [args.batch_size, args.seq_length])
            
            abs_delta_time = tf.abs(self.target_time-self.time_loss_output)
            time_loss = tf.square(abs_delta_time)
            self.mean_time_loss = tf.reduce_mean(time_loss)
            
            self.mean_loss = self.mean_aho_loss + self.mean_time_loss

        self.learning_rate = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.mean_loss, tvars), args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
    
    
    def sample(self, sess, prime_length = 3, num=5000, sampling_type=1, obj_list=['hand','book', 'phone', 'cup', 'bottle', 'bowl', 'orange', 'banana']):
        #prime_dir = self.args.data_dir
        
        
        prime_dir = 'data/all_multivideos_test/'
        #prime_loader = DataLoaderwithVocab(prime_dir)
        prime_loader = DataLoaderForSampling(prime_dir)
        
        #(0, 'rnn_GOPR0013.csv')
        #(1, 'rnn_GOPR0034.csv')
        #(2, 'rnn_GOPR0037.csv')
        #(3, 'rnn_GOPR0078.csv')
        #(4, 'rnn_GOPR0083.csv')
        #(5, 'rnn_GOPR5053.csv')
        #(6, 'rnn_GOPR5056.csv')
        #(7, 'rnn_GOPR5057.csv')
        #(8, 'rnn_GOPR7040.csv')
        #prime_length = 21
        prime_length = 17
        
        prime_tensor = prime_loader.get_tensor()[1]
        prime_tensor = prime_tensor[:, :]
        
        
        (chars_act, chars_ho, chars_aho, chars_hand, chars_obj) = prime_loader.get_chars()
        (vocab_act, vocab_ho, vocab_aho, vocab_hand, vocab_obj) = prime_loader.get_vocab()

        vocab_aho_size = len(vocab_aho)
        time_limit = 100000/60
        total_time = 0
        
        mask = np.ones(vocab_aho_size, np.float32)
        for i in range(vocab_aho_size):
            aho_label = chars_aho[i]
            act_label = aho_label.split(';')[0].strip()
            obj_names = act_label.split(' ')[1:]
            if not set(obj_names).issubset(set(obj_list)):
                mask[i] = 0
        
        # event sequences for return
        event_seq = []
        state = sess.run(self.cell.zero_state(1, tf.float32))## here 1 is batch_size
        act = np.zeros((1, 1))
        ho = np.zeros((1, 1))
        time = np.zeros((1, 1))
        
        for i in range(prime_tensor.shape[0] - 1):
            # input to rnn
            act[0, 0] = prime_tensor[i,0]
            ho[0, 0]  = prime_tensor[i,1]
            time[0, 0]  = prime_tensor[i,5]
            feed = {self.input_act:act, self.input_ho:ho, self.input_time: time, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)
            
            # save to event_seq
            aho = prime_tensor[i,2]
            aho_label = chars_aho[int(aho)]
            act_label  = aho_label.split(';')[0].strip()
            hand_label = aho_label.split(';')[1].strip()
            obj_label  = aho_label.split(';')[2].strip()
            elapsed    = prime_tensor[i,5]
            total_time += elapsed
            
            new_event = [act_label, hand_label, obj_label, elapsed]
            event_seq.append(new_event)
            print(new_event)
            
        # sampling new frame based on the last frame of prime
        last = prime_tensor[-1,:]
        act[0, 0] = last[0]
        ho[0,0]   = last[1]
        time[0, 0]  = last[5]
        aho = last[2]
        aho_label = chars_aho[int(aho)]
        act_label  = aho_label.split(';')[0].strip()
        hand_label = aho_label.split(';')[1].strip()
        obj_label  = aho_label.split(';')[2].strip()
        elapsed    = last[5]
        total_time += elapsed
        new_event = [act_label, hand_label, obj_label, elapsed]
        event_seq.append(new_event)
        print(new_event)
        #print new_event, total_time
        
        previous_event = new_event
        for n in np.arange(prime_length, num):
            feed = {self.input_act:act, self.input_ho:ho, self.input_time: time, self.initial_state:state}
            [probs_aho, elapsed_times, state] = sess.run([self.probs_aho, self.fc_output_time, self.final_state], feed)
            
            # get new sampled char
            p_aho  = probs_aho[0]
            p_aho = np.multiply(p_aho, mask)
            p_aho = p_aho*1.0/np.sum(p_aho)
            
            
            if sampling_type == 0:
                sample = np.argmax(p_aho)
            elif sampling_type == 2:
                if act == ' ':
                    sample = weighted_pick(p_aho)
                else:
                    sample = np.argmax(p_aho)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p_aho)
            
            # update label sequence
            sample = int(sample)
            aho_label  = chars_aho[sample]
            act_label  = aho_label.split(';')[0].strip()
            hand_label = aho_label.split(';')[1].strip()
            obj_label  = aho_label.split(';')[2].strip()
            
            # use the sampled aho to get elapsed time
            elapsed = elapsed_times[0][sample]            
            total_time += elapsed
            if total_time >= time_limit:
                elapsed -= (total_time - time_limit)
                new_event  = [act_label, hand_label, obj_label, elapsed]
                event_seq.append(new_event)
                break
    
            new_event  = [act_label, hand_label, obj_label, elapsed]
            if new_event[:3] == previous_event[:3]:
                event_seq.pop()
                elapsed += previous_event[3]
                new_event[3] = elapsed
            event_seq.append(new_event)
            previous_event = new_event
            
            if new_event[0] == 'end':
                break
            
            # update char for next feeding 
            ho_label  = hand_label + ';' + obj_label
            act[0, 0] = vocab_act.get(act_label)
            ho[0,0]   = vocab_ho.get(ho_label)
            time[0, 0]  = float(elapsed)
            
    
        return event_seq
    
