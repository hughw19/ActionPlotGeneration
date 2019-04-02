import codecs
import os
import collections, csv, string, math, glob
from six.moves import cPickle
import numpy as np
import time

def weighted_pick(weights):
    t = np.cumsum(weights)
    s = np.sum(weights)
    return(int(np.searchsorted(t, np.random.rand(1)*s)))

class DataLoader():
    def __init__(self, data_dir, batch_size, seq_length, fps = 60):
        ## input
        self.data_dir = data_dir
        self.input_file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        self.input_file_list = sorted(self.input_file_list)
        print(self.input_file_list)
        
        ## parameters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.fps = fps
        
        ## output
        self.event_txt = os.path.join(self.data_dir, "training_events.txt")
        vocab_file_act  = os.path.join(self.data_dir, "vocab_act.pkl")
        vocab_file_ho   = os.path.join(self.data_dir, "vocab_ho.pkl")
        vocab_file_aho  = os.path.join(self.data_dir, "vocab_aho.pkl")
        vocab_file_hand = os.path.join(self.data_dir, "vocab_hand.pkl")
        vocab_file_obj  = os.path.join(self.data_dir, "vocab_obj.pkl")
        vocab_file      = (vocab_file_act, vocab_file_ho, vocab_file_aho, vocab_file_hand, vocab_file_obj)
        tensor_file     = os.path.join(self.data_dir, "data.npy")
        
        ## processing data
        self.preprocess(vocab_file, tensor_file)
        #self.create_batches()
        #self.reset_batch_pointer()        
        
    
    def encode(self, data, vocab_file):
        counter = collections.Counter(data)
        count_pairs = sorted(counter.items(), key=lambda x: -x[1])
        chars, _ = zip(*count_pairs)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        with open(vocab_file, 'wb') as f:
            cPickle.dump(chars, f)
        
        tensor = np.array(list(map(vocab.get, data)))
        tensor = np.expand_dims(tensor,axis = 1) 
        return (chars, vocab_size, vocab, tensor)
    
    def decode(self,vocab_file):
        with open(vocab_file, 'rb') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        return (chars, vocab_size, vocab)
    
    
    def preprocess(self, vocab_file, tensor_file):
        self.label_data = []
        self.hand_obj_data = []
        self.act_hand_obj_data = []
        self.hand_data= []
        self.obj_data = []
        
        self.n_input_files = len(self.input_file_list)
        file_info = np.zeros((self.n_input_files, 2),dtype=np.int32)
        
        print('loading data from input file...')
        for j, input_file in enumerate(self.input_file_list):
            with open(input_file, 'rb') as csv_file:
                reader = csv.reader(csv_file, dialect='excel')  
                row = reader.next() # skip the title row
                file_info[j, 0] = len(self.label_data)
                for i,row in enumerate(reader):
                    new_frame = int(row[0].strip())

                    # get new label
                    new_label =row[1].strip()

                    # get new hand state & object state
                    new_hand =row[2].strip()
                    new_obj =row[3].strip()

                    self.label_data.append(new_label)
                    self.hand_obj_data.append(new_hand + ';' +new_obj)
                    self.act_hand_obj_data.append(new_label + ';' + new_hand + ';' + new_obj)
                    self.hand_data.append(new_hand)
                    self.obj_data.append(new_obj)
                file_info[j, 1] = len(self.label_data)
        #if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
        print('preprocessing data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act, tensor_act) = self.encode(self.label_data, vocab_file[0])
        print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho, self.vocab_size_ho, self.vocab_ho, tensor_ho) = self.encode(self.hand_obj_data, vocab_file[1])
        print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho, tensor_aho) = self.encode(self.act_hand_obj_data, vocab_file[2])
        print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand, self.vocab_size_hand, self.vocab_hand, tensor_hand) = self.encode(self.hand_data, vocab_file[3])
        print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj, tensor_obj) = self.encode(self.obj_data, vocab_file[4])
        print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        assert tensor_act.shape[0] == tensor_ho.shape[0]


        self.tensor = []
        self.tensor_info = []
        for j in range(self.n_input_files):
            video_tensor = None
            s_len = 0
            event_seq = []
            previous_aho = None
            for i in range(file_info[j, 0], file_info[j, 1]):
                aho = tensor_aho[i]
                if previous_aho == None:
                    previous_aho = aho
                    s_len += 1
                elif aho == previous_aho:
                    s_len += 1
                else: #event != previous_event
                    start_time = float(i - s_len)/self.fps
                    end_time   = float(i)/self.fps
                    elapsed    = end_time - start_time
                    new_tensor = np.array([tensor_act[i-1], tensor_ho[i-1], 
                                           tensor_aho[i-1], tensor_hand[i-1], tensor_obj[i-1], elapsed])
                    new_tensor = np.expand_dims(new_tensor,axis = 1) 

                    aho_label  = self.chars_aho[int(previous_aho)]
                    act_label  = aho_label.split(';')[0].strip()
                    hand_label = aho_label.split(';')[1].strip()
                    obj_label  = aho_label.split(';')[2].strip()
                    new_event  = [act_label, hand_label, obj_label, elapsed]
                    event_seq.append(new_event)

                    if video_tensor is None:
                        video_tensor = new_tensor
                    else:
                        video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
                    
                    previous_aho = aho
                    s_len = 1
            
            elapsed = float(s_len)/self.fps
            end_frame = file_info[j, 1] - 1
            new_tensor = np.array([tensor_act[end_frame], 
                                   tensor_ho[end_frame], 
                                   tensor_aho[end_frame], 
                                   tensor_hand[end_frame], 
                                   tensor_obj[end_frame], 
                                   elapsed])
            new_tensor = np.expand_dims(new_tensor,axis = 1) 
            video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
            video_tensor = np.transpose(video_tensor)

            aho_label  = self.chars_aho[int(previous_aho)]
            act_label  = aho_label.split(';')[0].strip()
            hand_label = aho_label.split(';')[1].strip()
            obj_label  = aho_label.split(';')[2].strip()
            new_event = [act_label, hand_label, obj_label, elapsed]
            event_seq.append(new_event)
  
            file_name = os.path.basename(self.input_file_list[j])
            print('training events in video file {}: '.format(file_name))
            etr = EventtoReadable(event_seq, self.event_txt)
            
            if len(event_seq) <= self.seq_length:
                assert False,\
                    "Not enough data in video file {}. Make seq_length small.".format(file_name)
            
            self.tensor.append(video_tensor)
            self.tensor_info.append(len(event_seq))
        
        self.tensor_info = np.array(self.tensor_info)
        self.num_batches = int(np.sum(self.tensor_info)/(self.batch_size*self.seq_length))
        np.save(tensor_file, self.tensor)
        print('data tensor saved.')
        
        #else:
        #    self.load_preprocessed(vocab_file, tensor_file)
        #    print('data tensor loaded, shape: {}'.format(self.tensor.shape))
            
    
    def load_preprocessed(self, vocab_file, tensor_file):
        print('loading data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act) = self.decode(vocab_file[0])
        print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho,  self.vocab_size_ho, self.vocab_ho) = self.decode(vocab_file[1])
        print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho) = self.decode(vocab_file[2])
        print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand,self.vocab_size_hand, self.vocab_hand) = self.decode(vocab_file[3])
        print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj) = self.decode(vocab_file[4])
        print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.shape[0] / (self.batch_size *self.seq_length))
    
    def get_vocab(self):
        return (self.vocab_act,self.vocab_ho,self.vocab_aho)
    
    def get_chars(self):
        return (self.chars_act, self.chars_ho, self.chars_aho)
    
    def get_tensor(self):
        return self.tensor
    
    '''
    def create_batches(self):
        length = self.tensor.shape[0]
        print('data length:', length)
        dim= self.tensor.shape[1]
        print('dim:', dim)
    
        self.num_batches = int(np.floor(self.tensor.shape[0] / (self.batch_size *self.seq_length)))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length, :]
        
        print('data length in batches: ', self.tensor.shape[0])
        print('num of batches: ', self.num_batches)
        
        xdata = np.copy(self.tensor[:, :])
        ydata = np.concatenate((self.tensor[1:, :] ,self.tensor[-1:, :]), axis=0) 
        print ('ydata shape: ',ydata.shape)
        print ('xdata shape: ',xdata.shape)
        
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1, dim), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1, dim), self.num_batches, 1)
    '''

    def next_batch(self):
        flag = False
        for i in range(self.batch_size):
            video_id =  weighted_pick(self.tensor_info)

            video_data = self.tensor[video_id]
            video_length = self.tensor_info[video_id]

            high = video_length - self.seq_length - 1
            start = np.random.randint(low = 0, high=high)

            x_seq = np.copy(video_data[start: start + self.seq_length, :])
            y_seq = np.copy(video_data[start+1: start + self.seq_length + 1, :]) 
            x_seq = np.expand_dims(x_seq, axis=0)
            y_seq = np.expand_dims(y_seq, axis=0)
            
            if not flag:
                x_batch = x_seq
                y_batch = y_seq
            else:
                x_batch = np.concatenate((x_batch, x_seq), axis = 0)
                y_batch = np.concatenate((y_batch, y_seq), axis = 0)
            
        return x_batch, y_batch

    
    def reset_batch_pointer(self):
        self.pointer = 0


class DataLoaderForSampling():
    def __init__(self, data_dir):
        if os.path.exists(os.path.join(data_dir, "prime.csv")):
            self.input_file = os.path.join(data_dir, "prime.csv")
        else:
            self.input_file = os.path.join(data_dir, "input.csv")
            
        self.vocab_file_act = os.path.join(data_dir, "vocab_act.pkl")
        self.vocab_file_ho = os.path.join(data_dir, "vocab_ho.pkl")
        self.vocab_file_aho = os.path.join(data_dir, "vocab_aho.pkl")
        self.vocab_file_hand = os.path.join(data_dir, "vocab_hand.pkl")
        self.vocab_file_obj = os.path.join(data_dir, "vocab_obj.pkl")
        self.vocab_file = (self.vocab_file_act,self.vocab_file_ho, self.vocab_file_aho,
                           self.vocab_file_hand,self.vocab_file_obj)
        
        self.tensor_file = os.path.join(data_dir, "data.npy")
        self.tensor = None
        
        self.load_preprocessed(self.vocab_file, self.tensor_file)
        #print('data tensor loaded, shape: {}'.format(self.tensor.shape))
    
    def decode(self,vocab_file):
        with open(vocab_file, 'rb') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        return (chars, vocab_size, vocab)
    
    def load_preprocessed(self, vocab_file, tensor_file):
        #print('loading data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act) = self.decode(vocab_file[0])
        #print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho,  self.vocab_size_ho, self.vocab_ho) = self.decode(vocab_file[1])
        #print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho) = self.decode(vocab_file[2])
        #print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand,self.vocab_size_hand, self.vocab_hand) = self.decode(vocab_file[3])
        #print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj) = self.decode(vocab_file[4])
        #print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        
        self.tensor = np.load(tensor_file)
        
        #print self.tensor

    def get_chars(self):
        return (self.chars_act, self.chars_ho, self.chars_aho, self.chars_hand, self.chars_obj)
    
    def get_vocab(self):
        return (self.vocab_act, self.vocab_ho, self.vocab_aho, self.vocab_hand, self.vocab_obj)
            
    def get_tensor(self):
        return self.tensor


class DataLoaderwithVocab():
    def __init__(self, data_dir, batch_size=1, seq_length=10, fps = 60):
        ## input
        self.data_dir = data_dir
        self.input_file_list = glob.glob(os.path.join(data_dir, "*.csv"))
        self.input_file_list = sorted(self.input_file_list)
        print(self.input_file_list)
        
        vocab_file_act  = os.path.join(self.data_dir, "vocab_act.pkl")
        vocab_file_ho   = os.path.join(self.data_dir, "vocab_ho.pkl")
        vocab_file_aho  = os.path.join(self.data_dir, "vocab_aho.pkl")
        vocab_file_hand = os.path.join(self.data_dir, "vocab_hand.pkl")
        vocab_file_obj  = os.path.join(self.data_dir, "vocab_obj.pkl")
        vocab_file      = (vocab_file_act, vocab_file_ho, vocab_file_aho, vocab_file_hand, vocab_file_obj)
        
        ## parameters
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.fps = fps
        
        ## output
        self.event_txt = os.path.join(self.data_dir, "training_events.txt")
        tensor_file     = os.path.join(self.data_dir, "data.npy")
        
        ## processing data
        self.preprocess(vocab_file, tensor_file)
        #self.create_batches()
        #self.reset_batch_pointer()        
        
    
    def encode_with_vocab(self, data, vocab_file):
        with open(vocab_file, 'r') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        
        tensor = np.array(list(map(vocab.get, data)))
        tensor = np.expand_dims(tensor,axis = 1) 
        return (chars, vocab_size, vocab, tensor)
    
    def decode(self,vocab_file):
        with open(vocab_file, 'rb') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        return (chars, vocab_size, vocab)
    
    
    def preprocess(self, vocab_file, tensor_file):
        self.label_data = []
        self.hand_obj_data = []
        self.act_hand_obj_data = []
        self.hand_data= []
        self.obj_data = []
        
        self.n_input_files = len(self.input_file_list)
        file_info = np.zeros((self.n_input_files, 2),dtype=np.int32)
        
        print('loading data from input file...')
        for j, input_file in enumerate(self.input_file_list):
            with open(input_file, 'rb') as csv_file:
                reader = csv.reader(csv_file, dialect='excel')  
                row = reader.next() # skip the title row
                file_info[j, 0] = len(self.label_data)
                for i,row in enumerate(reader):
                    new_frame = int(row[0].strip())

                    # get new label
                    new_label =row[1].strip()

                    # get new hand state & object state
                    new_hand =row[2].strip()
                    new_obj =row[3].strip()

                    self.label_data.append(new_label)
                    self.hand_obj_data.append(new_hand + ';' +new_obj)
                    self.act_hand_obj_data.append(new_label + ';' + new_hand + ';' + new_obj)
                    self.hand_data.append(new_hand)
                    self.obj_data.append(new_obj)
                file_info[j, 1] = len(self.label_data)
        #if not (os.path.exists(vocab_file) and os.path.exists(tensor_file)):
        print('preprocessing data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act, tensor_act) = self.encode_with_vocab(self.label_data, 
                                                                                                   vocab_file[0])
        print('vocab_size of action is {}'.format(self.vocab_size_act))
        
        (self.chars_ho, self.vocab_size_ho, self.vocab_ho, tensor_ho) = self.encode_with_vocab(self.hand_obj_data,
                                                                                               vocab_file[1])
        
        print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho, tensor_aho) = self.encode_with_vocab(self.act_hand_obj_data,
                                                                                                   vocab_file[2])
        
        print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand, self.vocab_size_hand, self.vocab_hand, tensor_hand) = self.encode_with_vocab(self.hand_data,
                                                                                                       vocab_file[3])
        
        print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj, tensor_obj) = self.encode_with_vocab(self.obj_data, 
                                                                                                   vocab_file[4])
        
        print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        assert tensor_act.shape[0] == tensor_ho.shape[0]


        self.tensor = []
        self.tensor_info = []
        for j in range(self.n_input_files):
            video_tensor = None
            s_len = 0
            event_seq = []
            previous_aho = None
            for i in range(file_info[j, 0], file_info[j, 1]):
                aho = tensor_aho[i]
                if previous_aho == None:
                    previous_aho = aho
                    s_len += 1
                elif aho == previous_aho:
                    s_len += 1
                else: #event != previous_event
                    start_time = float(i - s_len)/self.fps
                    end_time   = float(i)/self.fps
                    elapsed    = end_time - start_time
                    new_tensor = np.array([tensor_act[i-1], tensor_ho[i-1], 
                                           tensor_aho[i-1], tensor_hand[i-1], tensor_obj[i-1], elapsed])
                    new_tensor = np.expand_dims(new_tensor,axis = 1) 

                    aho_label  = self.chars_aho[int(previous_aho)]
                    act_label  = aho_label.split(';')[0].strip()
                    hand_label = aho_label.split(';')[1].strip()
                    obj_label  = aho_label.split(';')[2].strip()
                    new_event  = [act_label, hand_label, obj_label, elapsed]
                    event_seq.append(new_event)

                    if video_tensor is None:
                        video_tensor = new_tensor
                    else:
                        video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
                    
                    previous_aho = aho
                    s_len = 1
            
            elapsed = float(s_len)/self.fps
            end_frame = file_info[j, 1] - 1
            new_tensor = np.array([tensor_act[end_frame], 
                                   tensor_ho[end_frame], 
                                   tensor_aho[end_frame], 
                                   tensor_hand[end_frame], 
                                   tensor_obj[end_frame], 
                                   elapsed])
            new_tensor = np.expand_dims(new_tensor,axis = 1) 
            video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
            video_tensor = np.transpose(video_tensor)

            aho_label  = self.chars_aho[int(previous_aho)]
            act_label  = aho_label.split(';')[0].strip()
            hand_label = aho_label.split(';')[1].strip()
            obj_label  = aho_label.split(';')[2].strip()
            new_event = [act_label, hand_label, obj_label, elapsed]
            event_seq.append(new_event)
  
            file_name = os.path.basename(self.input_file_list[j])
            print('training events in video file {}: '.format(file_name))
            etr = EventtoReadable(event_seq, self.event_txt)
            
            if len(event_seq) <= self.seq_length:
                assert False,\
                    "Not enough data in video file {}. Make seq_length small.".format(file_name)
            
            self.tensor.append(video_tensor)
            self.tensor_info.append(len(event_seq))
        
        self.tensor_info = np.array(self.tensor_info)
        self.num_batches = int(np.sum(self.tensor_info)/(self.batch_size*self.seq_length))
        np.save(tensor_file, self.tensor)
        print('data tensor saved.')
        
        #else:
        #    self.load_preprocessed(vocab_file, tensor_file)
        #    print('data tensor loaded, shape: {}'.format(self.tensor.shape))
            
    
    def load_preprocessed(self, vocab_file, tensor_file):
        print('loading data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act) = self.decode(vocab_file[0])
        print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho,  self.vocab_size_ho, self.vocab_ho) = self.decode(vocab_file[1])
        print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho) = self.decode(vocab_file[2])
        print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand,self.vocab_size_hand, self.vocab_hand) = self.decode(vocab_file[3])
        print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj) = self.decode(vocab_file[4])
        print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        
        self.tensor = np.load(tensor_file)
        self.num_batches = int(self.tensor.shape[0] / (self.batch_size *self.seq_length))
    
    def get_vocab(self):
        return (self.vocab_act,self.vocab_ho,self.vocab_aho)
    
    def get_chars(self):
        return (self.chars_act, self.chars_ho, self.chars_aho)
    
    def get_tensor(self):
        return self.tensor
    
    '''
    def create_batches(self):
        length = self.tensor.shape[0]
        print('data length:', length)
        dim= self.tensor.shape[1]
        print('dim:', dim)
    
        self.num_batches = int(np.floor(self.tensor.shape[0] / (self.batch_size *self.seq_length)))

        # When the data (tensor) is too small, let's give them a better error message
        if self.num_batches==0:
            assert False, "Not enough data. Make seq_length and batch_size small."

        self.tensor = self.tensor[:self.num_batches * self.batch_size * self.seq_length, :]
        
        print('data length in batches: ', self.tensor.shape[0])
        print('num of batches: ', self.num_batches)
        
        xdata = np.copy(self.tensor[:, :])
        ydata = np.concatenate((self.tensor[1:, :] ,self.tensor[-1:, :]), axis=0) 
        print ('ydata shape: ',ydata.shape)
        print ('xdata shape: ',xdata.shape)
        
        self.x_batches = np.split(xdata.reshape(self.batch_size, -1, dim), self.num_batches, 1)
        self.y_batches = np.split(ydata.reshape(self.batch_size, -1, dim), self.num_batches, 1)
    '''

    def next_batch(self):
        flag = False
        for i in range(self.batch_size):
            video_id =  weighted_pick(self.tensor_info)

            video_data = self.tensor[video_id]
            video_length = self.tensor_info[video_id]

            high = video_length - self.seq_length - 1
            start = np.random.randint(low = 0, high=high)

            x_seq = np.copy(video_data[start: start + self.seq_length, :])
            y_seq = np.copy(video_data[start+1: start + self.seq_length + 1, :]) 
            x_seq = np.expand_dims(x_seq, axis=0)
            y_seq = np.expand_dims(y_seq, axis=0)
            
            if not flag:
                x_batch = x_seq
                y_batch = y_seq
            else:
                x_batch = np.concatenate((x_batch, x_seq), axis = 0)
                y_batch = np.concatenate((y_batch, y_seq), axis = 0)
            
        return x_batch, y_batch

    
    def reset_batch_pointer(self):
        self.pointer = 0



class DataLoaderForInfer():
    def __init__(self, args, data_dir):

        self.input_file_list = glob.glob(os.path.join(data_dir, "*.unused"))
        self.input_file_list = sorted(self.input_file_list)
            
        self.vocab_file_act = os.path.join(data_dir, "vocab_act.pkl")
        self.vocab_file_ho = os.path.join(data_dir, "vocab_ho.pkl")
        self.vocab_file_aho = os.path.join(data_dir, "vocab_aho.pkl")
        self.vocab_file_hand = os.path.join(data_dir, "vocab_hand.pkl")
        self.vocab_file_obj = os.path.join(data_dir, "vocab_obj.pkl")
        self.vocab_file = (self.vocab_file_act,self.vocab_file_ho, self.vocab_file_aho,
                           self.vocab_file_hand,self.vocab_file_obj)
        
        self.tensor = None
        self.batch_size = args.batch_size
        self.fps = 60
        self.seq_length = args.seq_length
        
        self.event_txt = os.path.join(data_dir, "training_events.txt")
        self.tensor_file = os.path.join(data_dir, "test_data.npy")
        
        
        self.video_pointer = 0
        self.batch_pointer = 0
        
        if False:#os.path.exists(self.tensor_file):
            self.load_preprocessed(self.vocab_file, self.tensor_file)
        else:
            self.process(self.vocab_file)
        #print('data tensor loaded, shape: {}'.format(self.tensor.shape))
        
        self.create_batches()
    
    def decode(self,vocab_file):
        with open(vocab_file, 'rb') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))

        
        return (chars, vocab_size, vocab)
    
    def decode_tensor(self, data, vocab_file):
        with open(vocab_file, 'rb') as f:
            chars = cPickle.load(f)
        vocab_size = len(chars)
        vocab = dict(zip(chars, range(len(chars))))
        tensor = np.array(list(map(vocab.get, data)))
        tensor = np.expand_dims(tensor,axis = 1) 
        
        return (chars, vocab_size, vocab, tensor)
    
    
    
    
    def load_preprocessed(self, vocab_file, tensor_file):
        #print('loading data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act) = self.decode(vocab_file[0])
        #print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho,  self.vocab_size_ho, self.vocab_ho) = self.decode(vocab_file[1])
        #print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho) = self.decode(vocab_file[2])
        #print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand,self.vocab_size_hand, self.vocab_hand) = self.decode(vocab_file[3])
        #print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj) = self.decode(vocab_file[4])
        #print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        
        self.tensor = np.load(tensor_file)
    
    
    
    
    
    def process(self, vocab_file):
        #print('loading data...')
        
        self.label_data = []
        self.hand_obj_data = []
        self.act_hand_obj_data = []
        self.hand_data= []
        self.obj_data = []
        
        self.n_input_files = len(self.input_file_list)
        print(self.n_input_files)
        file_info = np.zeros((self.n_input_files, 2),dtype=np.int32)
        
        print('loading data from input file...')
        for j, input_file in enumerate(self.input_file_list):
            with open(input_file, 'rb') as csv_file:
                reader = csv.reader(csv_file, dialect='excel')  

                row = reader.next() # skip the title row
                print(row)
                
                file_info[j, 0] = len(self.label_data)
                for i,row in enumerate(reader):
                    new_frame = int(row[0].strip())

                    # get new label
                    new_label =row[1].strip()

                    # get new hand state & object state
                    new_hand =row[2].strip()
                    new_obj =row[3].strip()
                    
                    self.label_data.append(new_label)
                    self.hand_obj_data.append(new_hand + ';' +new_obj)
                    self.act_hand_obj_data.append(new_label + ';' + new_hand + ';' + new_obj)
                    self.hand_data.append(new_hand)
                    self.obj_data.append(new_obj)
                file_info[j, 1] = len(self.label_data)
        

        print('preprocessing data...')
        (self.chars_act, self.vocab_size_act, self.vocab_act, tensor_act) = self.decode_tensor(self.label_data, self.vocab_file[0])
        print('vocab_size of action is {}'.format(self.vocab_size_act))
        (self.chars_ho, self.vocab_size_ho, self.vocab_ho, tensor_ho) = self.decode_tensor(self.hand_obj_data, self.vocab_file[1])
        print('vocab_size of hand-object is {}'.format(self.vocab_size_ho))
        (self.chars_aho, self.vocab_size_aho, self.vocab_aho, tensor_aho) = self.decode_tensor(self.act_hand_obj_data, self.vocab_file[2])
        print('vocab_size of act-hand-obj is {}'.format(self.vocab_size_aho))
        (self.chars_hand, self.vocab_size_hand, self.vocab_hand, tensor_hand) = self.decode_tensor(self.hand_data, self.vocab_file[3])
        print('vocab_size of hand is {}'.format(self.vocab_size_hand))
        (self.chars_obj, self.vocab_size_obj, self.vocab_obj, tensor_obj) = self.decode_tensor(self.obj_data, self.vocab_file[4])
        print('vocab_size of obj is {}'.format(self.vocab_size_obj))
        assert tensor_act.shape[0] == tensor_ho.shape[0]
        


        self.tensor = []
        self.tensor_info = []
        for j in range(self.n_input_files):
            print(j)
            video_tensor = None
            s_len = 0
            event_seq = []
            previous_aho = None
            for i in range(file_info[j, 0], file_info[j, 1]):
                aho = tensor_aho[i]
                if previous_aho == None:
                    previous_aho = aho
                    s_len += 1
                elif aho == previous_aho:
                    s_len += 1
                else: #event != previous_event
                    start_time = float(i - s_len)/self.fps
                    end_time   = float(i)/self.fps
                    elapsed    = end_time - start_time
                    new_tensor = np.array([tensor_act[i-1], tensor_ho[i-1], 
                                           tensor_aho[i-1], tensor_hand[i-1], tensor_obj[i-1], elapsed])
                    new_tensor = np.expand_dims(new_tensor,axis = 1) 
                    
                    aho_label  = self.chars_aho[int(previous_aho)]
                    act_label  = aho_label.split(';')[0].strip()
                    hand_label = aho_label.split(';')[1].strip()
                    obj_label  = aho_label.split(';')[2].strip()
                    new_event  = [act_label, hand_label, obj_label, elapsed]
                    event_seq.append(new_event)

                    if video_tensor == None:
                        video_tensor = new_tensor
                    else:
                        video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
                    
                    previous_aho = aho
                    s_len = 1
            
            elapsed = float(s_len)/self.fps
            new_tensor = np.array([tensor_act[-2], tensor_ho[-2], tensor_aho[-2], tensor_hand[-2], tensor_obj[-2], elapsed])
            new_tensor = np.expand_dims(new_tensor,axis = 1) 
            video_tensor = np.concatenate((video_tensor,new_tensor), axis = 1)
            video_tensor = np.transpose(video_tensor)

            aho_label  = self.chars_aho[int(previous_aho)]
            act_label  = aho_label.split(';')[0].strip()
            hand_label = aho_label.split(';')[1].strip()
            obj_label  = aho_label.split(';')[2].strip()
            new_event = [act_label, hand_label, obj_label, elapsed]
            event_seq.append(new_event)
  
            file_name = os.path.basename(self.input_file_list[j])
            print('training events in video file {}: '.format(file_name))
            etr = EventtoReadable(event_seq, self.event_txt)
            
            
            
            if len(event_seq) <= self.seq_length:
                assert False,\
                (len(event_seq),  self.seq_length,
                    "Not enough data in video file {}. Make seq_length small.".format(file_name))
            
            self.tensor.append(video_tensor)
            self.tensor_info.append(len(event_seq))
        
        self.num_videos = len(self.tensor)
        self.tensor_info = np.array(self.tensor_info)
        #self.num_batches = int(np.sum(self.tensor_info)/(self.batch_size*self.seq_length))
        np.save(self.tensor_file, self.tensor)
        #print('data tensor saved.')
        
        #print self.tensor

    def get_chars(self):
        return (self.chars_act, self.chars_ho, self.chars_aho, self.chars_hand, self.chars_obj)
    
    def get_vocab(self):
        return (self.vocab_act, self.vocab_ho, self.vocab_aho, self.vocab_hand, self.vocab_obj)
            
    def get_tensor(self):
        return self.tensor
    
    
    def create_batches(self):
        self.x_batches_byvideo = []
        self.y_batches_byvideo = []
        
        print(len(self.tensor))
        for i in range(len(self.tensor)):
        
            self.tensor_video = self.tensor[i]
            length = self.tensor_video.shape[0]
            print('data length:', length)
            dim= self.tensor_video.shape[1]
            print('dim:', dim)
            num_batches = int(self.tensor_video.shape[0] / (self.batch_size *
                                                       self.seq_length))

            # When the data (tensor) is too small, let's give them a better error message
            if num_batches==0:
                assert False, "Not enough data. Make seq_length and batch_size small."

            self.tensor_video = self.tensor_video[:num_batches * self.batch_size * self.seq_length, :]
            #print(self.tensor.shape)
            #print(self.num_batches)
            xdata = np.copy(self.tensor_video[:, :])
            ydata = np.concatenate((self.tensor_video[1:, :], self.tensor_video[-1:, :]), axis=0) 
            print ('ydata shape:', ydata.shape)
            print ('xdata shape:', xdata.shape)

            #print(xdata[0])
            #print(xdata.reshape(self.batch_size, -1, dim).shape)
            #print(self.num_batches)

            x_batches = np.split(xdata.reshape(self.batch_size, -1, dim), num_batches, 1)
            y_batches = np.split(ydata.reshape(self.batch_size, -1, dim), num_batches, 1)
            
            self.x_batches_byvideo.append(x_batches)
            self.y_batches_byvideo.append(y_batches)
        
        #print(self.x_batches[0].shape)
        #print(self.y_batches[0].shape)
    
    
    def next_video(self):
        self.current_x_batches = self.x_batches_byvideo[self.video_pointer]
        self.current_y_batches = self.y_batches_byvideo[self.video_pointer]
        
        self.num_batches = len(self.current_x_batches)
        self.video_pointer += 1
    

    def next_batch(self):
        x, y = self.current_x_batches[self.batch_pointer], self.current_y_batches[self.batch_pointer]
        self.batch_pointer += 1
        return x, y
    
    def reset_batch_pointer(self):
        self.batch_pointer = 0
    
       
class EventtoReadable:
    def __init__(self,event_seq, output_path, error_dict = {}, fps = 60, output_txt_dir=None):
        self.event_seq = event_seq
        self.fps = fps
        self.output_path = output_path
        self.output_txt_dir = output_txt_dir
        self.output = open(self.output_path, 'a')
        self.obj_states = {'cup':('0','1'),
                           'bowl':('0','1'),
                           'bottle':('0'),
                           'book':('0','1')}
        self.error_dict = error_dict
        self.n_event = 0
        self.n_error = 0
        self.make_readable()
        self.output.close()

    
    def event_description(self, event):
        assert event!=None
        obj_in_hand = None
        obj_state = None
        if event[1] != 'unoccupied':
            obj_in_hand = event[1]
            #obj_state = self.obj_states[obj_in_hand][int(event[2])]
            obj_state = event[2]
            dscpt = 'action: {:16s} hand: {:13s} object state: {:10s}'.format\
                        ('['+event[0]+']', '['+obj_in_hand+']', '['+obj_state+']')
        else:
            dscpt = 'action: {:16s} hand: {:13s}'.format('['+event[0]+']', '['+event[1]+']')
        return dscpt
    
    def event_description_output(self, event):
        assert event!=None
        obj_in_hand = None
        obj_state = None
        
        #if event[1] != 'unoccupied':
        #    obj_in_hand = event[1]
        #    obj_state = self.obj_states[obj_in_hand][int(event[2])]
        #    dscpt = 'action: {:16s} hand: {:12s} object state: {:10s}'.format\
        #                ('['+event[0]+']', '['+obj_in_hand+']', '['+obj_state+']')
        #else:
        #    dscpt = 'action: {:16s} hand: {:12s}'.format('['+event[0]+']', '['+event[1]+']')
        #return dscpt
        
        
        if event[0] == 'idle':
            dscpt = 'idle, undefined, 0, undefined, 0'
        else:
            action_words = event[0].split()
            if len(action_words) == 2:
                dscpt = '{}, {}, 0, undefined, 0'.format(action_words[0], action_words[1])
            elif len(action_words) == 3:
                dscpt = '{}, {}, 0, {}, 0'.format(action_words[0], action_words[1], action_words[2])
            else:
                assert False
        
        return dscpt
    
    def get_error_dict(self):
        return self.error_dict

    def print_to_file(self, str_):
        self.event_output.write(str_)
        print(str_[:-1])
    
    def make_readable(self):
        if self.output_txt_dir:
            event_file = os.path.join(self.output_txt_dir, 'sampled_events_{}.txt'.format(time.time()))
        else:
            temp_dir = os.path.dirname(self.output_path)
            event_file = os.path.join(temp_dir, 'sampled_events.txt')
        self.event_output = open(event_file, 'a')
        
        s_len = 0
        start_time = 0
        end_time = 0
        previous_event = None
        next_event = None
        current_obj_label = None
        
        self.print_to_file('Sampled Plot: \n')
        event_len = len(self.event_seq)
        i = 0
        while i < event_len - 1:
            event = self.event_seq[i]
                    
            act_label = event[0]
            hand_label = event[1]
            obj_label = event[2]
            elapsed = float(event[3])
            
            next_event = self.event_seq[i+1]
            next_act_label = next_event[0]
            
            end_time += elapsed
            if previous_event == None:
                previous_event = event
                self.n_event +=1
                if next_act_label!= act_label:
                    dscpt = self.event_description_output(event)

                    if obj_label =='N/A':
                        obj_label1 = 0
                        obj_label2 = 0
                    else:
                        obj_label1 = obj_label
                        obj_label2 = obj_label
                    self.print_to_file('{}, {}, {}, {}, {}\n'.format(dscpt, start_time, end_time, obj_label1, obj_label2))
                    start_time = end_time
            else: #event != previous_event
                #self.print_to_file('{:<5.2f}s - {:<5.2f}s ({:<5.2f}s): {}\n'.format(start_time, end_time, elapsed, dscpt))
                if next_act_label!= act_label:
                    dscpt = self.event_description_output(event)
                    if previous_event[0] != act_label:
                        if obj_label =='N/A':
                            obj_label1 = 0
                            obj_label2 = 0
                        else:
                            obj_label1 = obj_label
                            obj_label2 = obj_label
                    else:
                        obj_label1 = previous_event[2] if previous_event[2]!='N/A' else obj_label
                        obj_label2 = obj_label if obj_label!='N/A' else previous_event[2]
                        
                    self.print_to_file('{}, {}, {}, {}, {}\n'.format(dscpt, start_time, end_time, obj_label1, obj_label2))
                    start_time = end_time
                self.n_event +=1
                # check if physical conditions of the new event
                error_warning = self.check_error_new(previous_event, event)
                if error_warning != None:
                    self.n_error += 1
                    self.print_to_file(error_warning)
                    self.output.write('old event: {}\n'.format(self.event_description(previous_event)))
                    self.output.write('new event: {}\n'.format(self.event_description(event)))
                    self.output.write(error_warning)
                    if error_warning in self.error_dict:
                        self.error_dict[error_warning] +=1
                    else:
                        self.error_dict[error_warning] =1
                # update previous_event, event description and reset event_length                    
                previous_event = event
            i+=1
            
        self.print_to_file('In total: {} events, {} errors.\n'.format(self.n_event, self.n_error))
        self.print_to_file('\n')
        self.event_output.close()

    def check_error(self, previous_event, event):
        # only look at new different event
        if previous_event == None or event == previous_event:
            return None        
        old_action = previous_event[0]
        old_hand = previous_event[1]
        old_obj_state = previous_event[2]

        new_action = event[0]
        new_hand = event[1]
        new_obj_state = event[2]
        
        if new_action == 'end':
            return None
        
        if new_hand == 'unoccupied' and new_obj_state != 'N/A':
            return('#-----------------------# Error: Object state can\'t exist without object.\n')

        if new_action == 'idle':
            if old_hand!=new_hand:
                return('#-----------------------# Error: object in hand can\'t change in this action.\n')

            if old_obj_state!=new_obj_state:
                return('#-----------------------# Error: object state can\'t change in this action.\n')

        else: ## for non-idle action
            action_words = [_.strip() for _ in new_action.split(' ')]
            new_verb = action_words[0]
            new_obj  = action_words[1]
            second_obj = None
            if len(action_words) >= 3:
                second_obj  = action_words[2]
            
            
            
            if old_action == 'idle':
                old_verb = 'idle'
                old_obj =  'N/A'
            else:
                old_verb = old_action.split(' ')[0]
                old_obj = old_action.split(' ')[1]

            if new_verb == 'grasp':
                # check prerequisite: free hand
                if old_hand != 'unoccupied': 
                    return('#-----------------------# Error: hand must be unoccupied before grasp.\n')

                # check action consequence
                elif new_hand!= 'unoccupied':
                    # if already get something
                    if old_verb != new_verb:
                        return('#-----------------------# Error: action can\'t take effect immediately.\n')

                    else:# old_verb == 'grasp'
                        # check action continuity
                        if new_obj!= old_obj:
                            return('#-----------------------# Error: object of action is discontinuous.\n')

                        # check object continuity
                        if new_obj!= old_obj:
                            return('#-----------------------# Error: object in hand is discontinuous.\n')

                        # check hand - action consistence
                        if new_obj!= new_hand:
                            return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

            elif new_verb == 'move':
                # special case for moving hand
                if new_obj == 'hand': 
                    if old_hand != 'unoccupied' or new_hand != 'unoccupied': #move hand only with free hand
                        return('#-----------------------# Error: move hand is only with free hand.\n')

                # for moving object
                else:
                    # check prerequisite: object is in hand
                    if old_hand != new_obj: 
                        return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                    else:
                        # check hand - action consistence
                        if new_obj != new_hand:
                            return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                        # check object state continuity
                        if old_obj_state != new_obj_state and second_obj == None:
                            return('#-----------------------# Error: object state can\'t change in this action.\n')

            elif new_verb == 'release':
                # check prerequisite: object is in hand
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                # check continuity
                elif new_hand != 'unoccupied':
                    # check hand - action consistence
                    if new_obj != new_hand:
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    # check object state continuity
                    if old_obj_state != new_obj_state:
                        print(len(action_words))
                        print(action_words)
                        return('#-----------------------# Error: object state can\'t change in this action.\n')

                # check action consequence
                else: #new_hand == 'unoccupied':
                    if old_verb != new_verb:
                        return('#-----------------------# Error: action can\'t take effect immediately.\n')

                    else: # old_verb == 'release':
                        if new_obj!=old_obj:
                            return('#-----------------------# Error: object of action is discontinous.\n')

            elif new_verb in ['open', 'turn-on', 'pour']:
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '0':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    # check action consequence
                    elif new_obj_state == '1':
                        if old_verb != new_verb:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')
            
            elif new_verb == 'move-in':
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '0':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    # check action consequence
                    elif new_obj_state == '1':
                        if old_verb != new_verb:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')
            
            elif new_verb == 'move-out':
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '1':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    # check action consequence
                    elif new_obj_state == 0:
                        if old_verb != new_verb:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')

            elif new_verb in ['read']:
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '1':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    if new_obj_state != old_obj_state: 
                        return('#-----------------------# Error: object state can\'t change in this action.\n')
            
            elif new_verb in ['use']:
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '1':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')


            elif new_verb in ['close', 'turn-off']:
                # check prerequisite: book is in hand and closed
                if old_hand != new_obj: 
                    return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')

                elif old_obj_state != '1':
                    return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                else:
                    if new_hand != new_obj: 
                        return('#-----------------------# Error: object of action is not consistent with what in hand.\n')

                    # check action consequence
                    elif new_obj_state == 0:
                        if old_verb != new_verb:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')


            else:
                print(new_verb)
                return('#-----------------------# Error: unseen action class!\n')

        return None  
    
    def check_error_new(self, previous_event, event):
        # only look at new different event
        if previous_event == None or event == previous_event:
            return None        
        old_action = previous_event[0]
        old_hand = previous_event[1]
        old_obj_state = previous_event[2]

        new_action = event[0]
        new_hand = event[1]
        new_obj_state = event[2]
        
        if new_action == 'end':
            return None
        

        if new_action == 'idle':
            if old_obj_state!=new_obj_state:
                return('#-----------------------# Error: object state can\'t change in this action.\n')

        else: ## for non-idle action
            action_words = [_.strip() for _ in new_action.split(' ')]
            new_verb = action_words[0]
            new_obj  = action_words[1]
            second_obj = None
            if len(action_words) >= 3:
                second_obj  = action_words[2]
            
            
            
            if old_action == 'idle':
                old_verb = 'idle'
                old_obj =  'N/A'
            else:
                old_verb = old_action.split(' ')[0]
                old_obj = old_action.split(' ')[1]

            if new_verb == 'move' and new_obj == 'hand':
                # check prerequisite: free hand
                if old_verb == 'pour' or old_verb == 'use': 
                    return('#-----------------------# Error: hand can\'t move away from the object now.\n')
                if new_obj_state != 'N/A':
                    return('#-----------------------# Error: hand doesn\'t have a state.\n')
            else:
                if new_obj_state == 'N/A':
                    return('#-----------------------# Error: object needs a state.\n')
                if old_action != 'move hand' and old_action != 'idle' and old_obj != new_obj: 
                        return('#-----------------------# Error: before operating an object, the object has to be in hand.\n')
                

                if new_verb == 'move':# move object
                    if old_verb == 'move' and old_obj != new_obj and old_obj != 'hand':
                        return('#-----------------------# Error: lack of move hand between move this obj.\n')

                    if old_verb == 'move' and old_obj == new_obj:
                        return('#-----------------------# Error: object state can\'t change in this action.\n')


                elif new_verb in ['open', 'turn-on', 'pour']:
                    # check prerequisite: book is in hand and closed
                    if old_obj == new_obj and old_obj_state != '0':
                        return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                    # check action consequence
                    if new_obj_state == '1':
                        if old_action != new_action:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')

                elif new_verb == 'move-in':
                    # check prerequisite: book is in hand and closed
                    if (old_obj == new_obj and old_obj_state != '0'):
                        return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                    # check action consequence
                    if new_obj_state == '1':
                        if old_action != new_action:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')

                elif new_verb == 'move-out':
                    # check prerequisite: book is in hand and closed
                    if (old_obj == new_obj and old_obj_state != '1'):
                        return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                    if new_obj_state == 0:
                        if old_action != new_action:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')

                elif new_verb in ['read', 'use', 'drink']:
                    
                    # check prerequisite: book is in hand and closed
                    if (old_obj == new_obj and old_obj_state != '1') or new_obj_state != '1':
                        return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')
                    if old_obj == new_obj and new_obj_state != old_obj_state: 
                        return('#-----------------------# Error: object state can\'t change in this action.\n')


                elif new_verb in ['close', 'turn-off']:
                    # check prerequisite: book is in hand and closed
                    if old_obj == new_obj and old_obj_state != '1':
                        return('#-----------------------# Error: object state doesn\'t allow action to happen.\n')

                    # check action consequence
                    if new_obj_state == 0:
                        if old_action != new_action:
                            return('#-----------------------# Error: action can\'t take effect immediately.\n')


                else:
                    print(new_verb)
                    return('#-----------------------# Error: unseen action class!\n')

        return None   

        
        
        
        
    





