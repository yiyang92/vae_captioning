import tensorflow as tf
from utils.rnn_model import clip_by_value
def non_cnn_optimizer(loss, params):
    decoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     'decoder')
    encoder_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                     'encoder')
    other_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'cv_emb')
    other_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'imf_emb')
    other_vars += decoder_vars
    if not params.no_encoder:
        other_vars += encoder_vars
    gradients = tf.gradients(loss, other_vars)
    # clipped_grad = clip_by_value(gradients, -0.1, 0.1)
    clipped_grad, global_norm = tf.clip_by_global_norm(gradients,
                                                       params.lstm_clip_by_norm)
    grads_vars = zip(clipped_grad, other_vars)
    # learning rate decay
    learning_rate = tf.constant(params.learning_rate)
    global_step = tf.Variable(initial_value=0, name="global_step",
                              trainable=False,
                              collections=[tf.GraphKeys.GLOBAL_STEP,
                                           tf.GraphKeys.GLOBAL_VARIABLES])
    num_batches_per_epoch = params.num_ex_per_epoch / (
        params.batch_size + 0.001)
    decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)
    learning_rate_decay = tf.train.exponential_decay(learning_rate,
                                               global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=0.5,
                                               staircase=True)
    # lstm parameters update
    if params.optimizer == 'SGD':
        optimize = tf.train.GradientDescentOptimizer(
            learning_rate_decay).apply_gradients(grads_vars,
                                                 global_step=global_step)
    elif params.optimizer == 'Adam':
        optimize = tf.train.AdamOptimizer(
            params.learning_rate, beta1=0.8).apply_gradients(grads_vars,
                                                  global_step=global_step)
    elif params.optimizer == 'Momentum':
        momentum = 0.90
        optimize = tf.train.MomentumOptimizer(learning_rate_decay,
                                              momentum).apply_gradients(
                                                  grads_vars,
                                                  global_step=global_step)
    return optimize, global_step, global_norm
# fine-tuning CNN
def cnn_optimizer(loss, params):
    cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'cnn')
    gradients = tf.gradients(loss, cnn_vars)
    grads_vars = zip(gradients, cnn_vars)
    # learning rate decay
    learning_rate = tf.constant(params.cnn_lr)
    global_step = tf.Variable(initial_value=0, name="global_step",
                              trainable=False,
                              collections=[tf.GraphKeys.GLOBAL_STEP,
                                           tf.GraphKeys.GLOBAL_VARIABLES])
    num_batches_per_epoch = params.num_ex_per_epoch / (
        params.batch_size + 0.001)
    decay_steps = int(num_batches_per_epoch * params.num_epochs_per_decay)
    learning_rate_decay = tf.train.exponential_decay(learning_rate,
                                               global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=0.5,
                                               staircase=True)
    # lstm parameters update
    if params.cnn_optimizer == 'SGD':
        optimize = tf.train.GradientDescentOptimizer(
            learning_rate_decay).apply_gradients(grads_vars,
                                                 global_step=global_step)
    elif params.cnn_optimizer == 'Adam':
        optimize = tf.train.AdamOptimizer(
            params.cnn_lr, beta1=0.8).apply_gradients(grads_vars,
                                                  global_step=global_step)
    elif params.cnn_optimizer == 'Momentum':
        momentum = 0.90
        optimize = tf.train.MomentumOptimizer(learning_rate_decay,
                                              momentum).apply_gradients(
                                                  grads_vars,
                                                  global_step=global_step)
    return optimize, global_step
