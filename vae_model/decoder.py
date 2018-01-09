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
            z_mean = tf.zeros([tf.shape(self.images_fv)[0], self.params.latent_size])
            z_logstd = tf.zeros([tf.shape(self.images_fv)[0], self.params.latent_size])
            z = zs.Normal('z', mean=z_mean, logstd=z_logstd, group_event_ndims=0)
            # flatten image feature vector
            inp_flatten = layers.flatten(self.images_fv)
            inp_flatten = tf.expand_dims(inp_flatten, 1)
            # encoder and decoder have different embeddings but the same image features
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                        "dec_embeddings", [self.params.vocab_size, self.params.embed_size],
                        dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, self.captions)
            # dropout
            #if params.dec_keep_rate < 1:
            #    dec_inps = tf.nn.dropout(dec_inps, params.dec_keep_rate)
            # IDEA: use MLP for mapping. f(i) = [batch_size, feature_v_size] -> [bs, 1, embed]
            # IDEA: z = [batch_size, latent_dim] -> [batch_size, 1, embed]
            #z_dec = tf.expand_dims(z_dec, 1)
            #z_dec = layers.dense(z, params.decoder_hidden)
            #lstm_input = tf.concat([inp_flatten, z_dec, vect_inputs], 1)
            #lengths = lengths + 2
            #inp_h = tf.layers.dense(z_dec, params.decoder_hidden)
            #inp_c = tf.layers.dense(z_dec, params.decoder_hidden)
            #initial_state = rnn_placeholders((tf.contrib.rnn.LSTMStateTuple(inp_c, inp_h), ))
            #for tensor in flatten(initial_state):
            #    tf.add_to_collection('rnn_decoder_state1_input', tensor)
            #max_sl = tf.shape(self.captions)[1]
            z_dec = tf.expand_dims(z, 1)
            if not self.params.no_encoder:
                cell_0 = make_rnn_cell([self.params.decoder_hidden for _ in range(self.params.decoder_rnn_layers)],
                                           base_cell=tf.contrib.rnn.LSTMCell)
                _, final_state_0 = tf.nn.dynamic_rnn(cell_0, inputs=z_dec,
                                                        sequence_length=None,
                                                        initial_state=None,
                                                        swap_memory=True, dtype=tf.float32)
            with tf.variable_scope("decoder1") as scope2:
                cell_1 = make_rnn_cell([self.params.decoder_hidden for _ in range(self.params.decoder_rnn_layers)],
                                           base_cell=tf.contrib.rnn.LSTMCell)
                if self.params.no_encoder:
                    print("Not using encoder")
                    final_state_0 = None
                _, final_state_1 = tf.nn.dynamic_rnn(cell_1, inputs=inp_flatten,
                                                        sequence_length=None,
                                                        initial_state=final_state_0,
                                                        swap_memory=True, dtype=tf.float32, scope=scope2)
            for tensor in flatten(final_state_1):
                tf.add_to_collection('rnn_decoder_state1_output', tensor)
            initial_state = rnn_placeholders(final_state_1)
            cell = make_rnn_cell([self.params.decoder_hidden for _ in range(self.params.decoder_rnn_layers)],
                                            base_cell=tf.contrib.rnn.LSTMCell)
            with tf.variable_scope("decoder2") as scope3:
                outputs, final_state = tf.nn.dynamic_rnn(cell, inputs=vect_inputs,
                                                          sequence_length=self.lengths,
                                                          initial_state=initial_state,
                                                          swap_memory=True, dtype=tf.float32, scope=scope3)
            for tensor in flatten(final_state):
                tf.add_to_collection('rnn_decoder_state_output', tensor)
            # output shape [batch_size, seq_length, self.params.decoder_hidden]
            outputs_r = tf.reshape(outputs, [-1, self.params.decoder_hidden])
            x_logits = tf.layers.dense(outputs_r, units=self.params.vocab_size)
            shpe = (tf.shape(x_logits), tf.shape(inp_flatten), tf.shape(self.images_fv))
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
        """
        # get stop word index from dictionary
        stop_word_idx = self.data_dict.word2idx['<EOS>']
        state = None
        cap_list = [None] * in_pictures.shape[0]
        with tf.variable_scope("decoder", reuse=tf.AUTO_REUSE):
            _, _, shpe, states = self.px_z_fi({}, gen_mode = True)
        init_state, out_state, sample = states
        for i in range(len(cap_list)):
            cap_list[i] = {'image_id': picture_ids[i], 'caption': ' '}
            sentence = ['<BOS>']
            cur_it = 0
            gen_word_idx = 0
            while (cur_it < 40):
                if gen_word_idx == stop_word_idx:
                    break
                #sess.run(generator)
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
                cur_it += 1
            cap_list[i]['caption'] = ' '.join([word for word in sentence
                                               if word not in ['<BOS>', '<EOS>']])
        return cap_list
