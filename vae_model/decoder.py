import tensorflow as tf
from tensorflow import layers
from tensorflow.python.util.nest import flatten
import zhusuan as zs
from utils.rnn_model import make_rnn_cell, rnn_placeholders
import numpy as np


class Decoder():
    """Decoder class
    """
    def __init__(self, images_fv, captions, lengths, params, data_dict):
        """
        Args:
            images_fv: image features mapping to word embeddings
            captions: captions input placeholder
            lengths: caption length without zero-padding, used for tensorflow
            params: Parameters() class instance
            data_dict: Dictionary() class instance, used for caption generators
        dynamic_rnn lengths
        """
        self.images_fv = images_fv
        self.captions = captions
        self.lengths = lengths
        self.params = params
        self.data_dict = data_dict
        # c_i - optional cluster_vectors, can be specified separately
        self.c_i = None

    def px_z_fi(self, observed, gen_mode = False):
        """
        Args:
            observed: for q, parametrized by encoder, used during training
        Returns:
            model: zhusuan model object, can be used for getting probabilities
        """
        with zs.BayesianNet(observed) as model:
            z_mean = tf.zeros([tf.shape(self.images_fv)[0],
                               self.params.latent_size])
            # TODO: add std as a parameter
            z = zs.Normal('z', mean=z_mean, std=0.1,
                          group_event_ndims=1,
                          n_samples=self.params.gen_z_samples)
            # encoder and decoder have different embeddings but the same image features
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "dec_embeddings", [self.params.vocab_size,
                                           self.params.embed_size],
                        dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, self.captions)
            # captions dropout
            if self.params.dec_keep_rate < 1 and not gen_mode:
               vect_inputs = tf.nn.dropout(vect_inputs,
                                           self.params.dec_keep_rate)
            # map image feature vector to embed_dimension
            with tf.variable_scope("decoder") as scope:
                cell_0 = make_rnn_cell(
                    [self.params.decoder_hidden for _ in range(
                        self.params.decoder_rnn_layers)],
                    base_cell=tf.contrib.rnn.LSTMCell,
                    dropout_keep_prob=self.params.dec_lstm_drop)
                zero_state0 = cell_0.zero_state(
                    batch_size=tf.shape(self.images_fv)[0],
                    dtype=tf.float32)
                # run this cell to get initial state
                _, initial_state0 = cell_0(self.images_fv, zero_state0)
                if self.c_i != None:
                    _, initial_state0 = cell_0(self.c_i, initial_state0)
                if self.params.no_encoder:
                    if not gen_mode:
                        print("Not using q(z|x)")
                    initial_state = rnn_placeholders(initial_state0)
                else:
                    # vector z, mapped into embed_dim
                    z = tf.reshape(z, [-1, self.params.latent_size *
                                       self.params.gen_z_samples])
                    z_dec = layers.dense(z, self.params.embed_size)
                    _, z_state = cell_0(z_dec, initial_state0)
                    initial_state = rnn_placeholders(z_state)
                # captions LSTM
                for tensor in flatten(initial_state):
                    tf.add_to_collection('rnn_dec_inp', tensor)
                outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                         inputs=vect_inputs,
                                                         sequence_length=None,
                                                         initial_state=initial_state,
                                                         swap_memory=True,
                                                         dtype=tf.float32)
                for tensor in flatten(final_state):
                    tf.add_to_collection('rnn_dec_out', tensor)
            # output shape [batch_size, seq_length, self.params.decoder_hidden]
            if gen_mode:
                # only interested in the last output
                outputs = outputs[:, -1, :]
            outputs_r = tf.reshape(outputs, [-1, cell_0.output_size])
            x_logits = tf.layers.dense(outputs_r,
                                       units=self.data_dict.vocab_size)
            shpe = (tf.shape(z), tf.shape(outputs_r),
                    tf.shape(outputs))
            # for generating
            sample = None
            if gen_mode:
                if self.params.sample_gen == 'sample':
                    sample = tf.multinomial(
                        x_logits / self.params.temperature, 1)[0][0]
                elif self.params.sample_gen == 'beam_search':
                    pass
                else:
                    sample = tf.nn.softmax(x_logits)
        return model, x_logits, shpe, (initial_state, final_state, sample)

    def online_inference(self, sess, picture_ids, in_pictures, image_f_inputs,
                         to_json=False, stop_word='<EOS>'):
        """Generate caption, given batch of pictures and their ids (names).
        Args:
            sess: tf.Session() object
            picture_ids: list of picture ids in shape [batch_size]
            in_pictures: input pictures
            to_json: whether to write captions into json file
            stop_word: when stop caption generation
        Returns:
            cap_list: list of format [{'image_id', caption: ''}]
            cap_raw: list of generated caption indices
        """
        # get stop word index from dictionary
        stop_word_idx = self.data_dict.word2idx['<EOS>']
        cap_list = [None] * in_pictures.shape[0]
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            _, _, shpe, states = self.px_z_fi({}, gen_mode = True)
        init_state, out_state, sample = states
        cap_raw = []
        for i in range(len(in_pictures)):
            state = None
            cap_list[i] = {'image_id': picture_ids[i], 'caption': ' '}
            sentence = ['<BOS>']
            cur_it = 0
            gen_word_idx = 0
            cap_raw.append([])
            while (cur_it < self.params.gen_max_len):
                input_seq = [self.data_dict.word2idx[word] for word in sentence]
                feed = {self.captions: np.array(input_seq)[-1].reshape([1, 1]),
                        self.lengths: [1],
                        image_f_inputs: np.expand_dims(in_pictures[i], 0)}
                # for the first decoder step, the state is None
                if state is not None:
                     feed.update({init_state: state})
                next_word_probs, state = sess.run([sample, out_state],
                                                  feed)
                if self.params.sample_gen == 'greedy':
                    next_word_probs = next_word_probs.ravel()
                    t = self.params.temperature
                    next_word_probs = next_word_probs**(
                        1/t) / np.sum(next_word_probs**(1/t))
                    gen_word_idx = np.argmax(next_word_probs)
                elif self.params.sample_gen == 'sample':
                    gen_word_idx = next_word_probs
                gen_word = self.data_dict.idx2word[gen_word_idx]
                sentence += [gen_word]
                cap_raw[i].append(gen_word_idx)
                cur_it += 1
                if gen_word_idx == stop_word_idx:
                    break
            cap_list[i]['caption'] = ' '.join([word for word in sentence
                                               if word not in ['<BOS>', '<EOS>']])
        return cap_list, cap_raw
# TODO: finish beam search implementation
    # def beam_search(self, sess, picture_ids, in_pictures, stop_word='<EOS>',
    #                 beam_size=2):
    #     seed = self.data_dict.word2idx[seed]
    #     stop_word = self.data_dict.word2idx['<EOS>']
    #     for i in range(len(in_pictures)):
    #         # need to get n highest probabilities, than for each of them find p(x1)*p(x2|x1)...
    #         beam = [[seed] for _ in range(beam_size)]
    #         beam_prob = np.zeros([beam_size, self.params.gen_length])
    #         # initial feed
    #         feed = {self.captions: np.array(seed).reshape([1, 1]),
    #                 self.lengths: [1] * beam_size,
    #                 self.images_fv: np.expand_dims(in_pictures[i], 0)}
    #         # to tf add log_prob returring op norm_log_prob = tf.log(tf.softmax(...))
    #         # sample = tf.multinomial(...)
    #         index_arr, state, probs = sess.run([sample, out_state, log_prob], feed)
    #         #print(index_arr)
    #         index_arr = index_arr[0]
    #         # keeping previous states of hypothesis sentences
    #         states = [state] * beam_size
    #         # probabilities
    #         probs = probs[0][:beam_size]*-1
    #         # append to beam
    #         beam_prob[:, 0] = np.ones([beam_size])
    #         for j in range(beam_size):
    #              beam[j].append(index_arr[j])
    #              beam_prob[j][1] = probs[j]
    #         for st in range(2, gen_seq_len):
    #             for i in range(beam_size):
    #                 if stop_word in beam[i]:
    #                     continue
    #                 len_ = len(beam[i])
    #                 feed = {inputs_ps: np.array(beam[i]).reshape([1, len_]),
    #                     length: [len_], state_ps: states[i]}
    #                 # feed to network get probs
    #                 index_arr, state, probs = sess.run([sample, out_state, log_prob], feed)
    #                 index_arr = index_arr[0]
    #                 states[i] = state
    #                 # probabilities
    #                 probs = probs[0][:beam_size]*-1
    #                 max_sum_index = 0
    #                 # find probability max_sum_index and append to a beam
    #                 choose_index = -1
    #                 for j in range(beam_size):
    #                     temp_sum = probs[j] + np.sum(beam_prob[i])
    #                     if temp_sum > max_sum_index:
    #                         max_sum_index = temp_sum
    #                         choose_index += 1
    #                 beam_prob[i][st] = probs[choose_index]
    #                 beam[i].append(index_arr[choose_index])
    #         # find the best beam
    #         best_beam =beam[np.argmax(np.sum(beam_prob, 1))]
    #         #print("Best beam", best_beam)
    #         #print(beam)
    #     return [data_dict.idx2word[word] for word in best_beam]
