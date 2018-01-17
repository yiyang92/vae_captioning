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
            images_fv: image features placeholder
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
            z_logstd = tf.zeros([tf.shape(self.images_fv)[0],
                                 self.params.latent_size]) + 0.1
            # TODO: add n_samples
            # TODO: add std as a parameter
            z = zs.Normal('z', mean=z_mean, logstd=z_logstd,
                          group_event_ndims=0)
            # encoder and decoder have different embeddings but the same image features
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "dec_embeddings", [self.params.vocab_size, self.params.embed_size],
                        dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, self.captions)
            # captions dropout
            if self.params.dec_keep_rate < 1 and not gen_mode:
               vect_inputs = tf.nn.dropout(vect_inputs, self.params.dec_keep_rate)
            # map image feature vector to embed_dimension
            # TODO: place this mapping outside to feed into decoder
            images_fv = layers.dense(self.images_fv, self.params.embed_size)
            with tf.variable_scope("decoder") as scope:
                cell_0 = make_rnn_cell(
                    [self.params.decoder_hidden for _ in range(self.params.decoder_rnn_layers)],
                    base_cell=tf.contrib.rnn.LSTMCell)
                zero_state0 = cell_0.zero_state(
                    batch_size=tf.shape(images_fv)[0],
                    dtype=tf.float32)
                # run this cell to get initial state
                _, initial_state0 = cell_0(images_fv, zero_state0)
                if self.params.no_encoder:
                    if not gen_mode:
                        print("Not using z")
                    initial_state = rnn_placeholders(initial_state0)
                else:
                    # vector z, mapped into embed_dim
                    z_dec = layers.dense(z, self.params.embed_size)
                    _, z_state = cell_0(z_dec, initial_state0)
                    initial_state = rnn_placeholders(z_state)
                # captions LSTM
                for tensor in flatten(initial_state):
                    tf.add_to_collection('rnn_dec_inp', tensor)
                outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                         inputs=vect_inputs,
                                                         sequence_length=self.lengths,
                                                         initial_state=initial_state,
                                                         swap_memory=True,
                                                         dtype=tf.float32)
                for tensor in flatten(final_state):
                    tf.add_to_collection('rnn_dec_out', tensor)
            # output shape [batch_size, seq_length, self.params.decoder_hidden]
            outputs_r = tf.reshape(outputs, [-1, self.params.decoder_hidden])
            x_logits = tf.layers.dense(outputs_r, units=self.params.vocab_size)
            shpe = (tf.shape(z), tf.shape(self.images_fv),
                    tf.shape(vect_inputs))
            # for generating
            sample = None
            if gen_mode:
                sample = tf.multinomial(tf.reshape(x_logits, [tf.shape(outputs)[0],
                                                                 tf.shape(outputs)[1],
                                                                 self.params.vocab_size])[:, -1]
                                        /self.params.temperature, 1)[:, 0][:]
        return model, x_logits, shpe, (initial_state, final_state, sample)

    def online_inference(self, sess, picture_ids, in_pictures,
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
            while (cur_it < 40):
                input_seq = [self.data_dict.word2idx[word] for word in sentence]
                feed = {self.captions: np.array(input_seq).reshape([1, len(input_seq)]),
                        self.lengths: [len(input_seq)],
                        self.images_fv: np.expand_dims(in_pictures[i], 0)}
                # for the first decoder step, the state is None
                if state is not None:
                     feed.update({init_state: state})
                index, state = sess.run([sample, out_state], feed)
                gen_word_idx = int(index)
                gen_word = self.data_dict.idx2word[gen_word_idx]
                sentence += [gen_word]
                cap_raw[i].append(gen_word_idx)
                cur_it += 1
                if gen_word_idx == stop_word_idx:
                    break
            cap_list[i]['caption'] = ' '.join([word for word in sentence
                                               if word not in ['<BOS>', '<EOS>']])
        return cap_list, cap_raw
