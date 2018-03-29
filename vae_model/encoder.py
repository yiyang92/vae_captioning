import tensorflow as tf
from tensorflow import layers
import zhusuan as zs
from utils.rnn_model import make_rnn_cell, rnn_placeholders

class Encoder():
    def __init__(self, images_fv, captions, lengths, params):
        """
        Args:
            images_fv: image features mapping to word embeddings
            captions: captions input placeholder
            lengths: caption length without zero-padding, used for tensorflow
            params: Parameters() class instance
        """
        self.images_fv = images_fv
        self.captions = captions
        self.lengths = lengths
        self.params = params
        # c_i - optional cluster_vectors, can be specified separately
        self.c_i = None #
        self.c_i_ph = None

    def q_net(self):
        """Calculate approximate posterior q(z|x, f(I))
        Returns:
            model: zhusuan model object, can be used for getting probabilities
        """
        with zs.BayesianNet() as model:
            # encoder and decoder have different embeddings but the same image features
            with tf.device("/cpu:0"):
                embedding = tf.get_variable(
                            "enc_embeddings", [self.params.vocab_size,
                                               self.params.embed_size],
                            dtype=tf.float32)
                vect_inputs = tf.nn.embedding_lookup(embedding, self.captions)
            with tf.name_scope(name="encoder0") as scope1:
                cell_0 = make_rnn_cell(
                    [self.params.encoder_hidden
                     for _ in range(self.params.encoder_rnn_layers)],
                    base_cell=tf.contrib.rnn.LSTMCell)
                zero_state0 = cell_0.zero_state(
                    batch_size=tf.shape(self.images_fv)[0],
                    dtype=tf.float32)
                # run this cell to get initial state
                _, initial_state0 = cell_0(self.images_fv, zero_state0)
                if self.c_i != None and self.params.use_c_v:
                    _, initial_state0 = cell_0(self.c_i, initial_state0)
                outputs, final_state = tf.nn.dynamic_rnn(cell_0,
                                                         inputs=vect_inputs,
                                                         sequence_length=self.lengths,
                                                         initial_state=initial_state0,
                                                         swap_memory=True,
                                                         dtype=tf.float32,
                                                         scope=scope1)
            # [batch_size, 2 * lstm_hidden_size]
            # final_state = ((c, h), )
            final_state = tf.concat(values=final_state[0], axis=1,
                                    name="encoder_hidden")
            if self.params.prior == 'Normal':
                lz_mean = layers.dense(inputs=final_state,
                                       units=self.params.latent_size,
                                       activation=None)
                lz_logstd = layers.dense(inputs=final_state,
                                         units=self.params.latent_size,
                                         activation=None)
            # define latent variable`s Stochastic Tensor
            # add mu_k, sigma_k, CVAe ag-cvae
            tm_list = [] # means
            tl_list = [] # variances
            if self.params.prior == 'GMM':
                cluster = tf.squeeze(tf.multinomial(self.c_i_ph, 1))
                indices = tf.squeeze(tf.range(tf.shape(self.c_i_ph)[0]))
                cluster = tf.stack([indices,
                                    tf.cast(cluster, tf.int32)], 1)
                for i in range(90):
                    with tf.variable_scope("gmm_ll_{}".format(i)):
                        lz_mean = layers.dense(inputs=final_state,
                                               units=self.params.latent_size)
                        lz_logstd = layers.dense(inputs=final_state,
                                                 units=self.params.latent_size)
                        tm_list.append(tf.expand_dims(lz_mean, 1))
                        tl_list.append(tf.expand_dims(lz_logstd, 1))
                # [batch_size, 90, z_dim]
                tm_list = tf.concat(tm_list, 1)
                tl_list = tf.concat(tl_list, 1)
                lz_mean = tf.gather_nd(tm_list, cluster)
                lz_logstd = tf.gather_nd(tl_list, cluster)

            if self.params.prior == 'AG':
                # ck*N(mu, sigma)
                for i in range(90):
                    with tf.variable_scope("ag_ll_{}".format(i)):
                        lz_mean = layers.dense(inputs=final_state,
                                               units=self.params.latent_size)
                        lz_logstd = layers.dense(inputs=final_state,
                                                 units=self.params.latent_size)
                        tm_list.append(tf.expand_dims(lz_mean, 1))
                        tl_list.append(tf.expand_dims(lz_logstd, 1))
                # [batch_size, 90, 150]
                # ob_vector [batch_size, 90]
                # need [batch_size, 150]
                tm_list = tf.concat(tm_list, 1)
                tl_list = tf.concat(tl_list, 1)
                c_i_exp = tf.expand_dims(self.c_i_ph, 1)
                lz_mean = tf.squeeze(tf.matmul(c_i_exp, tm_list), 1)
                lz_logstd = tf.squeeze(tf.matmul(c_i_exp, tl_list), 1)
            z = zs.Normal('z', lz_mean, lz_logstd, group_event_ndims=1,
                          n_samples=self.params.gen_z_samples)
        return z, tm_list, tl_list
