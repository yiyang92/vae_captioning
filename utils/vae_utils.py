import os
import pickle
import numpy as np
import tensorflow as tf

def init_clusters(num_clusters, latent_size,
                  c_m_file='./pickles/cluster_means.pickle'):
    """Initialize clusters."""
    # initialize sigma as constant, mu drawn randomly
    c_sigma = tf.constant(0.1)
    cluster_mu_matrix = []
    # generate or restore cluster matrix
    if os.path.exists(c_m_file):
        # load existing matrix
        with open(c_m_file, 'rb') as rf:
            c_means = pickle.load(rf)
    else:
        # generate clusters
        print("Generating clusters, saving to the {}".format(c_m_file))
        for id_cluster in range(num_clusters):
            with tf.variable_scope("cl_init_mean_{}".format(id_cluster)):
                cluster_item = 2*np.random.random_sample(
                    (1, latent_size)) - 1
                cluster_item = cluster_item/(np.sqrt(
                    np.sum(cluster_item**2)))
                cluster_mu_matrix.append(cluster_item)
        c_means = np.stack(cluster_mu_matrix)
        with open(c_m_file, 'wb') as wf:
            pickle.dump(c_means, wf)
    c_means = tf.squeeze(tf.cast(c_means, dtype=tf.float32))
    return c_means, c_sigma
