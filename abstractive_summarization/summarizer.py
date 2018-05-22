# -*- coding: utf-8 -*-

import pickle
import string
import codecs

import numpy as np
import tensorflow as tf

from words_and_vecs import word2vec, vec2word, np_nearest_neighbour

path_vec_summs_train = 'prep_data\\reduced_vec_summaries_train'
path_vec_texts_train = 'prep_data\\reduced_vec_texts_train'
path_vec_summs_valid = 'prep_data\\reduced_vec_summaries_valid'
path_vec_texts_valid = 'prep_data\\reduced_vec_texts_valid'
path_vec_summs_test = 'prep_data\\reduced_vec_summaries_test'
path_vec_texts_test = 'prep_data\\reduced_vec_texts_test'

path_vocab = 'prep_data\\vocab_limit'
path_embedding = 'prep_data\\embd_limit'

train_log_file = 'log\\log_train.txt'
train_avg_loss_file = 'log\\running_avg_loss_train.txt'
valid_log_file = 'log\\log_valid.txt'
valid_avg_loss_file = 'log\\running_avg_loss_valid.txt'
test_log_file = 'log\\log_test.txt'
test_avg_loss_file = 'log\\running_avg_loss_test.txt'
    

MAX_SUMMARY_LEN = 80
MAX_TEXT_LEN = 1300

D = 10
    
class Summarizer(object):
    
    def __init__(self):    
        # load vocabs
        with open (path_vocab, 'rb') as fp:
            self.vocab_limit = pickle.load(fp)
        with open (path_embedding, 'rb') as fp:
            self.embd_limit = pickle.load(fp)
        self.np_embd_limit = np.asarray(self.embd_limit, dtype=np.float32)
        
        # load train
        with open (path_vec_summs_train, 'rb') as fp:
            self.vec_summaries_train = pickle.load(fp)
        with open (path_vec_texts_train, 'rb') as fp:
            self.vec_texts_train = pickle.load(fp)
            
        # load valid 
        with open (path_vec_summs_valid, 'rb') as fp:
            self.vec_summaries_valid = pickle.load(fp)
        with open (path_vec_texts_valid, 'rb') as fp:
            self.vec_texts_valid = pickle.load(fp)
            
        # load test    
        with open (path_vec_summs_test, 'rb') as fp:
            self.vec_summaries_test = pickle.load(fp)
        with open (path_vec_texts_test, 'rb') as fp:
            self.vec_texts_test = pickle.load(fp)
            
        self.word_vec_dim = len(self.embd_limit[0])
        self.vocab_len = len(self.vocab_limit)
        self.train_len = len(self.vec_summaries_train)
        self.valid_len = len(self.vec_summaries_valid)
        self.test_len = len(self.vec_summaries_test)
    
        # set up params
        self.hidden_size = 500
        self.learning_rate = 0.003
        self.K = 5       
        self.training_iters = 50         
        self.SOS = self.embd_limit[self.vocab_limit.index('<SOS>')]
        
        # placeholders
        self.tf_text = tf.placeholder(tf.float32, [None, self.word_vec_dim])
        self.tf_seq_len = tf.placeholder(tf.int32)
        self.tf_summary = tf.placeholder(tf.int32, [None])
        self.tf_output_len = tf.placeholder(tf.int32)
            
        
    def transform_out(self, output_text):
        output_len = len(output_text)
        transformed_output = np.zeros([output_len], dtype=np.int32)
        for i in range(0, output_len):
            transformed_output[i] = self.vocab_limit.index(vec2word(output_text[i], self.embd_limit, self.vocab_limit))
        return transformed_output  
    
    
    
    def forward_encoder(self, inp, hidden, cell,
                    wf, uf, bf,
                    wi, ui, bi,
                    wo, uo, bo,
                    wc, uc, bc,
                    Wattention, seq_len, inp_dim):

        Wattention = tf.nn.softmax(Wattention, 0)
        hidden_forward = tf.TensorArray(size=seq_len, dtype=tf.float32)
        
        hidden_residuals = tf.TensorArray(size=self.K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
        hidden_residuals = hidden_residuals.unstack(tf.zeros([self.K, self.hidden_size], dtype=tf.float32))
        
        i = 0
        j = self.K
        
        def cond(i, j, hidden, cell, hidden_forward, hidden_residuals):
            return i < seq_len
        
        def body(i, j, hidden, cell, hidden_forward, hidden_residuals):
            
            x = tf.reshape(inp[i], [1, inp_dim])
            
            hidden_residuals_stack = hidden_residuals.stack()
            
            RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-self.K : j], Wattention), 0)
            RRA = tf.reshape(RRA, [1, self.hidden_size])
            
            # LSTM with RRA
            fg = tf.sigmoid( tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
            ig = tf.sigmoid( tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
            og = tf.sigmoid( tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
            cell = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid( tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
            hidden = tf.multiply(og, tf.tanh(cell + RRA))
            
            hidden_residuals = tf.cond(tf.equal(j, seq_len - 1 + self.K),
                                       lambda: hidden_residuals,
                                       lambda: hidden_residuals.write(j, tf.reshape(hidden, [self.hidden_size])))
    
            hidden_forward = hidden_forward.write(i, tf.reshape(hidden, [self.hidden_size]))
            
            return i+1, j+1, hidden, cell, hidden_forward, hidden_residuals
        
        _, _, _, _, hidden_forward, hidden_residuals = tf.while_loop(cond, body, [i, j, hidden, cell, hidden_forward, hidden_residuals])
        
        hidden_residuals.close().mark_used()
        
        return hidden_forward.stack()
        

    def backward_encoder(self, inp, hidden, cell,
                         wf, uf, bf,
                         wi, ui, bi,
                         wo, uo, bo,
                         wc, uc, bc,
                         Wattention, seq_len, inp_dim):
        
        Wattention = tf.nn.softmax(Wattention, 0)
        hidden_backward = tf.TensorArray(size=seq_len, dtype=tf.float32)
        
        hidden_residuals = tf.TensorArray(size=self.K, dynamic_size=True, dtype=tf.float32, clear_after_read=False)
        hidden_residuals = hidden_residuals.unstack(tf.zeros([self.K, self.hidden_size], dtype=tf.float32))
        
        i = seq_len - 1
        j = self.K
        
        def cond(i, j, hidden, cell, hidden_backward, hidden_residuals):
            return i > -1
        
        def body(i, j, hidden, cell, hidden_backward, hidden_residuals):
            
            x = tf.reshape(inp[i], [1, inp_dim])
            
            hidden_residuals_stack = hidden_residuals.stack()
            
            RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-self.K : j], Wattention), 0)
            RRA = tf.reshape(RRA, [1, self.hidden_size])
            
            # LSTM with RRA
            fg = tf.sigmoid( tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
            ig = tf.sigmoid( tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
            og = tf.sigmoid( tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
            cell = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid( tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
            hidden = tf.multiply(og, tf.tanh(cell + RRA))
    
            hidden_residuals = tf.cond(tf.equal(j, seq_len - 1 + self.K),
                                       lambda: hidden_residuals,
                                       lambda: hidden_residuals.write(j, tf.reshape(hidden, [self.hidden_size])))
            
            hidden_backward = hidden_backward.write(i, tf.reshape(hidden, [self.hidden_size]))
            
            return i-1, j+1, hidden, cell, hidden_backward, hidden_residuals
        
        _, _, _, _, hidden_backward, hidden_residuals = tf.while_loop(cond, body, [i, j, hidden, cell, hidden_backward, hidden_residuals])
    
        hidden_residuals.close().mark_used()
        
        return hidden_backward.stack()
    
    
    def decoder(self, x, hidden, cell,
            wf, uf, bf,
            wi, ui, bi,
            wo, uo, bo,
            wc, uc, bc, RRA):
    
        # LSTM with RRA
        fg = tf.sigmoid( tf.matmul(x, wf) + tf.matmul(hidden, uf) + bf)
        ig = tf.sigmoid( tf.matmul(x, wi) + tf.matmul(hidden, ui) + bi)
        og = tf.sigmoid( tf.matmul(x, wo) + tf.matmul(hidden, uo) + bo)
        cell_next = tf.multiply(fg, cell) + tf.multiply(ig, tf.sigmoid( tf.matmul(x, wc) + tf.matmul(hidden, uc) + bc))
        hidden_next = tf.multiply(og, tf.tanh(cell + RRA))
        
        return hidden_next, cell_next
    
    
    def score(self, hs, ht, Wa, seq_len):
        return tf.reshape(tf.matmul(tf.matmul(hs, Wa), tf.transpose(ht)), [seq_len])
    
    def align(self, hs, ht, Wp, Vp, Wa, tf_seq_len):
       
        pd = tf.TensorArray(size=(2*D + 1), dtype=tf.float32)
        
        positions = tf.cast(tf_seq_len - 1 - 2*D, dtype=tf.float32)
        
        sigmoid_multiplier = tf.nn.sigmoid(tf.matmul(tf.tanh(tf.matmul(ht, Wp)), Vp))
        sigmoid_multiplier = tf.reshape(sigmoid_multiplier, [])
        
        pt_float = positions*sigmoid_multiplier
        
        pt = tf.cast(pt_float, tf.int32)
        pt = pt + D
        
        sigma = tf.constant(D/2, dtype=tf.float32)
        
        i = 0
        pos = pt - D
        
        def cond(i, pos, pd):            
            return i < (2*D + 1)
                          
        def body(i, pos, pd):           
            comp_1 = tf.cast(tf.square(pos - pt), tf.float32)
            comp_2 = tf.cast(2*tf.square(sigma), tf.float32)                
            pd = pd.write(i, tf.exp(-(comp_1 / comp_2)))                
            return i + 1, pos + 1, pd
                          
        i, pos, pd = tf.while_loop(cond, body, [i, pos, pd])
        
        local_hs = hs[(pt - D) : (pt + D + 1)]
        
        normalized_scores = tf.nn.softmax(self.score(local_hs, ht, Wa, 2*D + 1))
        
        pd = pd.stack()
        
        G = tf.multiply(normalized_scores, pd)
        G = tf.reshape(G, [2*D + 1, 1])
        
        return G, pt
        
    
    def model(self, tf_text, tf_seq_len, tf_output_len):
            
        ##############
        # Parameters #
        ##############
    
        
        ##############################
        # Forward encoder parameters #
        ##############################
                
        initial_hidden_f = tf.get_variable(name='hidden_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        cell_f = tf.get_variable(name='cell_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wf_f = tf.get_variable(name='wf_f', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uf_f = tf.get_variable(name='uf_f', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bf_f = tf.get_variable(name='bf_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wi_f = tf.get_variable(name='wi_f', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        ui_f = tf.get_variable(name='ui_f', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bi_f = tf.get_variable(name='bi_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wo_f = tf.get_variable(name='wo_f', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uo_f = tf.get_variable(name='uo_f', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bo_f = tf.get_variable(name='bo_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wc_f = tf.get_variable(name='wc_f', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uc_f = tf.get_variable(name='uc_f', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bc_f = tf.get_variable(name='bc_f', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        Wattention_f = tf.get_variable(name='Wattention_f', shape=[self.K, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        ###############################
        # Backward encoder parameters #
        ###############################
        
        initial_hidden_b = tf.get_variable(name='hidden_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        cell_b = tf.get_variable(name='cell_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wf_b = tf.get_variable(name='wf_b', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uf_b = tf.get_variable(name='uf_b', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bf_b = tf.get_variable(name='bf_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wi_b = tf.get_variable(name='wi_b', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        ui_b = tf.get_variable(name='ui_b', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bi_b = tf.get_variable(name='bi_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wo_b = tf.get_variable(name='wo_b', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uo_b = tf.get_variable(name='uo_b', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bo_b = tf.get_variable(name='bo_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wc_b = tf.get_variable(name='wc_b', shape=[self.word_vec_dim, self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uc_b = tf.get_variable(name='uc_b', shape=[self.hidden_size, self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(self.hidden_size)))
        bc_b = tf.get_variable(name='bc_b', shape=[1, self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        Wattention_b = tf.get_variable(name='Wattention_b', shape=[self.K, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        ########################
        # Attention parameters #
        ########################
                
        Wp = tf.get_variable(name='Wp', shape=[2*self.hidden_size, 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        Vp = tf.get_variable(name='Vp', shape=[50, 1], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        Wa = tf.get_variable(name='Wa', shape=[2*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        Wc = tf.get_variable(name='Wc', shape=[4*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        
        ######################
        # Decoder parameters #
        ######################
        
        Ws = tf.get_variable(name='Ws', shape=[2*self.hidden_size, self.vocab_len], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        
        cell_d = tf.get_variable(name='cell_d', shape=[1, 2*self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wf_d = tf.get_variable(name='wf_d', shape=[self.word_vec_dim, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uf_d = tf.get_variable(name='uf_d', shape=[2*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(2*self.hidden_size)))
        bf_d = tf.get_variable(name='bf_d', shape=[1, 2*self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wi_d = tf.get_variable(name='wi_d', shape=[self.word_vec_dim, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        ui_d = tf.get_variable(name='ui_d', shape=[2*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(2*self.hidden_size)))
        bi_d = tf.get_variable(name='bi_d', shape=[1, 2*self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wo_d = tf.get_variable(name='wo_d', shape=[self.word_vec_dim, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uo_d = tf.get_variable(name='uo_d', shape=[2*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(2*self.hidden_size)))
        bo_d = tf.get_variable(name='bo_d', shape=[1, 2*self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        wc_d = tf.get_variable(name='wc_d', shape=[self.word_vec_dim, 2*self.hidden_size], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.01))
        uc_d = tf.get_variable(name='uc_d', shape=[2*self.hidden_size, 2*self.hidden_size], dtype=tf.float32, initializer=tf.constant_initializer(np.eye(2*self.hidden_size)))
        bc_d = tf.get_variable(name='bc_d', shape=[1, 2*self.hidden_size], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        
        hidden_residuals_d = tf.TensorArray(size=self.K, dynamic_size=True, dtype=tf.float32, clear_after_read=False, tensor_array_name='hidden_residuals_d', colocate_with_first_write_call=True)
        hidden_residuals_d = hidden_residuals_d.unstack(tf.zeros([self.K, 2*self.hidden_size], dtype=tf.float32))
        
        Wattention_d = tf.get_variable(name='Wattention_d', shape=[self.K, 1], dtype=tf.float32, initializer=tf.zeros_initializer)
        
        output = tf.TensorArray(size=tf_output_len, dtype=tf.float32, tensor_array_name='output', colocate_with_first_write_call=True)
                                   
        
        #################################
        # Neural network's architecture #
        #################################
        
        
        #######################
        # Bi-directional LSTM #
        #######################
                                   
        hidden_forward = self.forward_encoder(tf_text,
                                         initial_hidden_f, cell_f,
                                         wf_f, uf_f, bf_f,
                                         wi_f, ui_f, bi_f,
                                         wo_f, uo_f, bo_f,
                                         wc_f, uc_f, bc_f,
                                         Wattention_f,
                                         tf_seq_len,
                                         self.word_vec_dim)
        
        hidden_backward = self.backward_encoder(tf_text,
                                         initial_hidden_b, cell_b,
                                         wf_b, uf_b, bf_b,
                                         wi_b, ui_b, bi_b,
                                         wo_b, uo_b, bo_b,
                                         wc_b, uc_b, bc_b,
                                         Wattention_b,
                                         tf_seq_len,
                                         self.word_vec_dim)
        
        encoded_hidden = tf.concat([hidden_forward, hidden_backward], 1)
        
        #########################
        # Attention and decoder #
        #########################
        
        decoded_hidden = encoded_hidden[0]
        decoded_hidden = tf.reshape(decoded_hidden, [1, 2*self.hidden_size])
        Wattention_d_normalized = tf.nn.softmax(Wattention_d)
        tf_embd_limit = tf.convert_to_tensor(self.np_embd_limit)
        
        y = tf.convert_to_tensor(self.SOS) #inital decoder token <SOS> vector
        y = tf.reshape(y, [1, self.word_vec_dim])
        
        j = self.K
        
        hidden_residuals_stack = hidden_residuals_d.stack()
        
        RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-self.K : j], Wattention_d_normalized), 0)
        RRA = tf.reshape(RRA, [1, 2*self.hidden_size])
        
        decoded_hidden_next,cell_d = self.decoder(y, decoded_hidden, cell_d,
                                      wf_d, uf_d, bf_d,
                                      wi_d, ui_d, bi_d,
                                      wo_d, uo_d, bo_d,
                                      wc_d, uc_d, bc_d,
                                      RRA)
        decoded_hidden = decoded_hidden_next
        
        hidden_residuals_d = hidden_residuals_d.write(j, tf.reshape(decoded_hidden, [2*self.hidden_size]))
        
        j = j + 1
                               
        i = 0
        
        def attention_decoder_cond(i, j, decoded_hidden, cell_d, hidden_residuals_d, output):
            return i < tf_output_len
        
        def attention_decoder_body(i, j, decoded_hidden, cell_d ,hidden_residuals_d, output):
            
            ###################
            # Local attention #
            ###################
            
            G, pt = self.align(encoded_hidden, decoded_hidden, Wp, Vp, Wa, tf_seq_len)
            local_encoded_hidden = encoded_hidden[pt-D : pt+D+1]
            weighted_encoded_hidden = tf.multiply(local_encoded_hidden, G)
            context_vector = tf.reduce_sum(weighted_encoded_hidden, 0)
            context_vector = tf.reshape(context_vector, [1, 2*self.hidden_size])
            
            attended_hidden = tf.tanh(tf.matmul(tf.concat([context_vector, decoded_hidden], 1), Wc))
            
            ###########
            # Decoder #
            ###########
            
            y = tf.matmul(attended_hidden, Ws)
            
            output = output.write(i, tf.reshape(y, [self.vocab_len]))
            
            y = tf.nn.softmax(y)
            
            y_index = tf.cast(tf.argmax(tf.reshape(y, [self.vocab_len])), tf.int32)
            y = tf_embd_limit[y_index]
            y = tf.reshape(y, [1, self.word_vec_dim])
            
            hidden_residuals_stack = hidden_residuals_d.stack()
            
            RRA = tf.reduce_sum(tf.multiply(hidden_residuals_stack[j-self.K : j], Wattention_d_normalized), 0)
            RRA = tf.reshape(RRA, [1, 2*self.hidden_size])
            
            decoded_hidden_next,cell_d = self.decoder(y, decoded_hidden, cell_d,
                                      wf_d, uf_d, bf_d,
                                      wi_d, ui_d, bf_d,
                                      wo_d, uo_d, bf_d,
                                      wc_d, uc_d, bc_d,
                                      RRA)
            
            decoded_hidden = decoded_hidden_next
            
            hidden_residuals_d = tf.cond(tf.equal(j, tf_output_len+self.K), #(+1 for <SOS>)
                                       lambda: hidden_residuals_d,
                                       lambda: hidden_residuals_d.write(j, tf.reshape(decoded_hidden, [2*self.hidden_size])))
            
            return i+1, j+1, decoded_hidden, cell_d, hidden_residuals_d, output
        
        i, j, decoded_hidden, cell_d, hidden_residuals_d, output = tf.while_loop(attention_decoder_cond,
                                                attention_decoder_body,
                                                [i, j, decoded_hidden, cell_d, hidden_residuals_d, output])
        hidden_residuals_d.close().mark_used()
        
        output = output.stack()
        
        return output
    
    
    def train(self):
        output = self.model(self.tf_text, self.tf_seq_len, self.tf_output_len)
    
        # Optimizer
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.tf_summary))
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        
        # Prediction        
        pred = tf.TensorArray(size=self.tf_output_len, dtype=tf.int32)
        
        i=0   
        
        def cond_pred(i, pred):
            return i < self.tf_output_len
        def body_pred(i, pred):
            pred = pred.write(i, tf.cast(tf.argmax(output[i]), tf.int32))
            return i+1, pred
        
        i, pred = tf.while_loop(cond_pred, body_pred, [i, pred])         
        prediction = pred.stack()
        
        init = tf.global_variables_initializer()
        
        test_freq = 3
        
        with tf.Session() as sess:
    
            saver = tf.train.Saver() 
            sess.run(init)
            
            step = 0   
            display_step = 10
            
            while step < self.training_iters:
                print('\r' + str(step) + '/' + str(self.training_iters), end = '')
                for i in range(0, self.train_len):                   
                    train_out = self.transform_out(self.vec_summaries_train[i][0: len(self.vec_summaries_train[i]) - 1])
                    
                    if i % display_step == 0:
                        saver.save(sess, "log\\tmp\\model.ckpt")
                        with codecs.open(train_log_file, 'a+', 'utf8') as writer:
                            writer.write('\nIteration: ' + str(i))
                            writer.write('\nTraining input sequence length: ' + str(len(self.vec_texts_train[i])))
                            writer.write('\nTraining target outputs sequence length: ' + str(len(train_out)))
                            writer.write('\nTEXT: ')
        
                            flag = 0
                            for vec in self.vec_texts_train[i]:
                                if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                                    writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                                else:
                                    writer.write(' ' + str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                                flag = 1
            
                            writer.write('\n')
        
        
                    # Backpropagation
                    _, loss, pred = sess.run([optimizer, cost, prediction], feed_dict={self.tf_text: self.vec_texts_train[i], 
                                                            self.tf_seq_len: len(self.vec_texts_train[i]), 
                                                            self.tf_summary: train_out,
                                                            self.tf_output_len: len(train_out)})
                    
                 
                    if i % display_step==0:
                        with codecs.open(train_log_file, 'a+', 'utf8') as writer:
                            writer.write('PREDICTED SUMMARY:\n')
                            flag = 0
                            for index in pred:
                                if self.vocab_limit[int(index)] in string.punctuation or flag==0:
                                    writer.write(str(self.vocab_limit[int(index)]))
                                else:
                                    writer.write(' ' + str(self.vocab_limit[int(index)]))
                                flag=1
                            writer.write('\n')
                            
                            writer.write('ACTUAL SUMMARY:\n')
                            flag = 0
                            for vec in self.vec_summaries_train[i]:
                                if vec2word(vec, self.embd_limit, self.vocab_limit) != 'eos':
                                    if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                                        writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                                    else:
                                        writer.write(' '+ str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                                flag=1
            
                            writer.write("\n")
                            writer.write("loss = " + str(loss))
                        with codecs.open(train_avg_loss_file, 'a+', 'utf8') as writer:
                            writer.write(str(step) + ' ' + str(loss) + '\n')
                self.validate(output, sess, step)  
                if ((step + 1) % test_freq == 0):
                    self.test(output, sess, step)
                step=step+1
                
                
    def validate(self, output, sess, step):   
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.tf_summary))
                
        pred = tf.TensorArray(size=self.tf_output_len, dtype=tf.int32)
        
        i=0   
        
        def cond_pred(i, pred):
            return i < self.tf_output_len
        def body_pred(i, pred):
            pred = pred.write(i, tf.cast(tf.argmax(output[i]), tf.int32))
            return i+1, pred
        
        i, pred = tf.while_loop(cond_pred, body_pred, [i, pred])         
        prediction = pred.stack()
        
        
        for i in range(0, self.valid_len):                   
            valid_out = self.transform_out(self.vec_summaries_valid[i][0: len(self.vec_summaries_valid[i]) - 1])
                    
            with codecs.open(valid_log_file, 'a+', 'utf8') as writer:
                writer.write('\nIteration: ' + str(i))
                writer.write('\nValidation input sequence length: ' + str(len(self.vec_texts_valid[i])))
                writer.write('\nValidation target outputs sequence length: ' + str(len(valid_out)))
                writer.write('\nTEXT: ')
        
                flag = 0
                for vec in self.vec_texts_valid[i]:
                    if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                        writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                    else:
                        writer.write(' ' + str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                    flag = 1
            
                writer.write('\n')
        
            loss, pred = sess.run([cost, prediction], feed_dict={self.tf_text: self.vec_texts_valid[i], 
                                                            self.tf_seq_len: len(self.vec_texts_valid[i]), 
                                                            self.tf_summary: valid_out,
                                                            self.tf_output_len: len(valid_out)})
                    
            with codecs.open(valid_log_file, 'a+', 'utf8') as writer:
                writer.write('PREDICTED SUMMARY:\n')
                flag = 0
                for index in pred:
                    if self.vocab_limit[int(index)] in string.punctuation or flag==0:
                        writer.write(str(self.vocab_limit[int(index)]))
                    else:
                        writer.write(' ' + str(self.vocab_limit[int(index)]))
                    flag=1
                writer.write('\n')
                            
                writer.write('ACTUAL SUMMARY:\n')
                flag = 0
                for vec in self.vec_summaries_valid[i]:
                    if vec2word(vec, self.embd_limit, self.vocab_limit) != 'eos':
                        if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                            writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                        else:
                            writer.write(' '+ str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                        flag=1
            
                writer.write("\n")
                writer.write("loss = " + str(loss))
                with codecs.open(valid_avg_loss_file, 'a+', 'utf8') as writer:
                    writer.write(str(step) + ' ' + str(loss) + '\n')
                    
    def test(self, output, sess, step):   
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=output, labels=self.tf_summary))
               
        pred = tf.TensorArray(size=self.tf_output_len, dtype=tf.int32)
        
        i=0   
        
        def cond_pred(i, pred):
            return i < self.tf_output_len
        def body_pred(i, pred):
            pred = pred.write(i, tf.cast(tf.argmax(output[i]), tf.int32))
            return i+1, pred
        
        i, pred = tf.while_loop(cond_pred, body_pred, [i, pred])         
        prediction = pred.stack()
        
        
        for i in range(0, self.test_len):                   
            test_out = self.transform_out(self.vec_summaries_test[i][0: len(self.vec_summaries_test[i]) - 1])
                    
            with codecs.open(test_log_file, 'a+', 'utf8') as writer:
                writer.write('\nIteration: ' + str(i))
                writer.write('\nTest input sequence length: ' + str(len(self.vec_texts_test[i])))
                writer.write('\nTest target outputs sequence length: ' + str(len(test_out)))
                writer.write('\nTEXT: ')
        
                flag = 0
                for vec in self.vec_texts_test[i]:
                    if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                        writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                    else:
                        writer.write(' ' + str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                    flag = 1
            
                writer.write('\n')
        
            loss, pred = sess.run([cost, prediction], feed_dict={self.tf_text: self.vec_texts_test[i], 
                                                            self.tf_seq_len: len(self.vec_texts_test[i]), 
                                                            self.tf_summary: test_out,
                                                            self.tf_output_len: len(test_out)})
                    
            with codecs.open(test_log_file, 'a+', 'utf8') as writer:
                writer.write('PREDICTED SUMMARY:\n')
                flag = 0
                for index in pred:
                    if self.vocab_limit[int(index)] in string.punctuation or flag==0:
                        writer.write(str(self.vocab_limit[int(index)]))
                    else:
                        writer.write(' ' + str(self.vocab_limit[int(index)]))
                    flag=1
                writer.write('\n')
                            
                writer.write('ACTUAL SUMMARY:\n')
                flag = 0
                for vec in self.vec_summaries_test[i]:
                    if vec2word(vec, self.embd_limit, self.vocab_limit) != 'eos':
                        if vec2word(vec, self.embd_limit, self.vocab_limit) in string.punctuation or flag==0:
                            writer.write(str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                        else:
                            writer.write(' '+ str(vec2word(vec, self.embd_limit, self.vocab_limit)))
                        flag=1
            
                writer.write("\n")
                writer.write("loss = " + str(loss))
                with codecs.open(test_avg_loss_file, 'a+', 'utf8') as writer:
                    writer.write(str(step) + ' ' + str(loss) + '\n')