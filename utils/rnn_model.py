from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def rnn_placeholders(state):
    """Convert RNN state tensors to placeholders with the zero state as default."""
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder_with_default(c, c.shape, c.op.name)
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    # if using GRU Cells
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder_with_default(h, h.shape, h.op.name)
        return h
    else:
        structure = [rnn_placeholders(x) for x in state]
        return tuple(structure)

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell):
      """Makes a RNN cell from the given hyperparameters.
      Args:
        rnn_layer_sizes: A list of integer sizes (in units) for each layer of the
            RNN.
        dropout_keep_prob: The float probability to keep the output of any given
            sub-cell.
        attn_length: The size of the attention vector.
        base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
      Returns:
          A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
      """
      cells = []
      for num_units in rnn_layer_sizes:
        cell = base_cell(num_units)
        if attn_length and not cells:
          # Add attention wrapper to first layer.
          cell = tf.contrib.rnn.AttentionCellWrapper(
              cell, attn_length, state_is_tuple=True)
        cell = tf.contrib.rnn.DropoutWrapper(
            cell, output_keep_prob=dropout_keep_prob)
        cells.append(cell)

      cell = tf.contrib.rnn.MultiRNNCell(cells)

      return cell

def weight_bias(W_shape, b_shape, bias_init=0.1):
    W = tf.get_variable(name='weight1',
                        shape=W_shape, dtype=tf.float32,
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
    b = tf.get_variable(name='bias1',
                        shape=b_shape, dtype=tf.float32,
                        initializer=tf.constant_initializer(0.1))
    return W, b

def highway_network(x, size, carry_bias=-1.0, scope='enc'):
    W, b = weight_bias([size, size], [size])

    with tf.variable_scope('transform_gate{}'.format(scope)):
        W_T, b_T = weight_bias([size, size], [size], bias_init=carry_bias)

    T = tf.sigmoid(tf.matmul(x, W_T) + b_T, name="transform_gate")
    H = tf.nn.sigmoid(tf.matmul(x, W) + b, name="activation")
    C = tf.subtract(1.0, T, name="carry_gate")

    y = tf.add(tf.multiply(H, T), tf.multiply(x, C), "y")

    return y
# from ruotianluo github
import collections, six
def clip_by_value(t_list, clip_value_min, clip_value_max, name=None):
    if (not isinstance(t_list, collections.Sequence)
            or isinstance(t_list, six.string_types)):
        raise TypeError("t_list should be a sequence")
    t_list = list(t_list)

    with tf.name_scope(name or "clip_by_value") as name:
        values = [
            tf.convert_to_tensor(
                t.values if isinstance(t, tf.IndexedSlices) else t,
                name="t_%d" % i)
            if t is not None else t
            for i, t in enumerate(t_list)]
        values_clipped = []
        for i, v in enumerate(values):
            if v is None:
                values_clipped.append(None)
            else:
                with tf.get_default_graph().colocate_with(v):
                    values_clipped.append(
                        tf.clip_by_value(v, clip_value_min, clip_value_max))

        list_clipped = [
            tf.IndexedSlices(c_v, t.indices, t.dense_shape)
            if isinstance(t, tf.IndexedSlices)
            else c_v
            for (c_v, t) in zip(values_clipped, t_list)]

    return list_clipped
