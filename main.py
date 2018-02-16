import os
import json
import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorflow import layers
from tensorflow.python.util.nest import flatten

# import utils
from utils.data import Data
from utils.rnn_model import make_rnn_cell, rnn_placeholders
from utils.parameters import Parameters
from vae_model.decoder import Decoder
from vae_model.encoder import Encoder

print("Tensorflow version: ", tf.__version__)

def main(params):
    # load data, class data contains captions, images, image features (if avaliable)
    base_model = tf.contrib.keras.applications.VGG16(weights='imagenet',
                                                     include_top=True)
    model = tf.contrib.keras.models.Model(inputs=base_model.input,
                                          outputs=base_model.get_layer('fc2').output)
    if params.gen_val_captions < 0:
        repartiton = False
    else:
        repartiton = True
    data = Data(coco_dir, True, model, repartiton=repartiton,
                gen_val_cap=params.gen_val_captions)
    # load batch generator, repartiton to use more val set images in train
    batch_gen = data.load_train_data_generator(params.batch_size)
    val_gen = data.get_valid_data(500, val_tr_unused=batch_gen.unused_cap_in)
    test_gen = data.get_test_data(500)
    # annotations vector of form <EOS>...<BOS><PAD>...
    ann_inputs_enc = tf.placeholder(tf.int32, [None, None])
    ann_inputs_dec = tf.placeholder(tf.int32, [None, None])
    ann_lengths = tf.placeholder(tf.int32, [None])
    # use prepared image features [batch_size, 4096] (fc2)
    image_f_inputs = tf.placeholder(tf.float32, [None, 4096])
    # dictionary
    cap_dict = data.dictionary
    params.vocab_size = cap_dict.vocab_size
    # image features [b_size + f_size(4096)] -> [b_size + embed_size]
    images_fv = layers.dense(image_f_inputs, params.embed_size, name='imf_emb')
    # encoder, input fv and ...<BOS>,get z
    if not params.no_encoder:
        encoder = Encoder(images_fv, ann_inputs_enc, ann_lengths, params)
    # decoder, input_fv, get x, x_logits (for generation)
    decoder = Decoder(images_fv, ann_inputs_dec, ann_lengths, params,
                      cap_dict)
    if params.use_c_v:
        # cluster vectors from "Diverse and Accurate Image Description.." paper.
        # 80 is number of classes, for now hardcoded
        # for GMM-CVAE must be specified
        c_i = tf.placeholder(tf.float32, [None, 90])
        c_i_emb = layers.dense(c_i, params.embed_size, name='cv_emb')
        # map cluster vectors into embedding space
        decoder.c_i = c_i_emb
        decoder.c_i_ph = c_i
        if not params.no_encoder:
            encoder.c_i = c_i_emb
            encoder.c_i_ph = c_i
    if not params.no_encoder:
        qz = encoder.q_net()
        if params.prior == 'Normal':
            # kld between normal distributions KL(q, p), see Kingma et.al
            kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + qz.distribution.logstd
                    - tf.square(qz.distribution.mean)
                    - tf.exp(qz.distribution.logstd),1))
        elif params.prior == 'GMM':
            kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + qz.distribution.logstd
                    - tf.square(qz.distribution.mean)
                    - tf.exp(qz.distribution.logstd),1))
        elif params.prior == 'AG':
            kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + qz.distribution.logstd
                    - tf.square(qz.distribution.mean)
                    - tf.exp(qz.distribution.logstd),1))
    with tf.variable_scope("decoder"):
        if params.no_encoder:
            dec_model, x_logits, shpe, _ = decoder.px_z_fi({})
        else:
            dec_model, x_logits, shpe, _ = decoder.px_z_fi({'z': qz})
    # calculate rec. loss, mask padded part
    labels_flat = tf.reshape(ann_inputs_enc, [-1])
    # debugging print
    prnt1 = tf.Print(labels_flat, [shpe[0], tf.shape(labels_flat), shpe[1],
                                   shpe[2]],
                     message="shapes")
    ce_loss_padded = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=x_logits,labels=labels_flat)
    loss_mask = tf.sign(tf.to_float(labels_flat))
    masked_loss = loss_mask * ce_loss_padded
    # restore original shape
    masked_loss = tf.reshape(masked_loss, tf.shape(ann_inputs_enc))
    mean_loss_by_example = tf.reduce_sum(
        masked_loss, 1) / tf.to_float(ann_lengths)
    rec_loss = tf.reduce_mean(mean_loss_by_example)
    # kld weight annealing
    anneal = tf.placeholder_with_default(0, [])
    annealing = (tf.tanh(
        (tf.to_float(anneal) - 1000 * params.ann_param)/1000) + 1)/2
    # overall loss reconstruction loss - kl_regularization
    if not params.no_encoder:
        # tf.reduce_mean(tf.to_float(ann_lengths)) for numerical stability
        lower_bound = tf.reduce_mean(
            tf.to_float(ann_lengths)) * rec_loss + tf.multiply(
                tf.to_float(annealing), tf.to_float(kld))
    else:
        lower_bound = rec_loss
        kld = tf.constant(0.0)
    # we need to maximize lower_bound
    gradients = tf.gradients(lower_bound, tf.trainable_variables())
    clipped_grad, _ = tf.clip_by_global_norm(gradients, 5.0)
    grads_vars = zip(clipped_grad, tf.trainable_variables())
    # learning rate decay
    learning_rate = tf.constant(params.learning_rate)
    global_step = tf.placeholder(tf.int32)
    num_batches_per_epoch = data.num_examples // params.batch_size
    decay_steps = num_batches_per_epoch * params.num_epochs_per_decay
    learning_rate_decay = tf.train.exponential_decay(learning_rate,
                                               global_step,
                                               decay_steps=decay_steps,
                                               decay_rate=0.5,
                                               staircase=True)
    if params.optimizer == 'SGD':
        optimize = tf.train.GradientDescentOptimizer(
            learning_rate_decay).apply_gradients(grads_vars)
    elif params.optimizer == 'Adam':
        optimize = tf.train.AdamOptimizer(
            params.learning_rate).apply_gradients(grads_vars)
    # model restore
    saver = tf.train.Saver(tf.trainable_variables())
    # m_builder = tf.saved_model.builder.SavedModelBuilder('./saved_model')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        # train using batch generator, every iteration get
        # f(I), [batch_size, max_seq_len], seq_lengths
        if params.restore:
            print("Restoring from checkpoint")
            saver.restore(sess, "./checkpoints/{}.ckpt".format(
                params.checkpoint))
        cur_t = 0
        for e in range(params.num_epochs):
            for f_images_batch, captions_batch, cl_batch, c_v in batch_gen.next_batch(
                use_obj_vectors=params.use_c_v):
                feed = {image_f_inputs: f_images_batch,
                        ann_inputs_enc: captions_batch[1],
                        ann_inputs_dec: captions_batch[0],
                        ann_lengths: cl_batch,
                        anneal: cur_t,
                        learning_rate: params.learning_rate,
                        global_step: cur_t}
                if params.use_c_v:
                    feed.update({c_i: c_v[:, 1:]})
                # debuging print
                if cur_t == 0:
                    _ = sess.run(prnt1, feed_dict=feed)
                kl, rl, lb, _, ann = sess.run([kld, rec_loss, lower_bound,
                                               optimize, annealing],
                                              feed_dict=feed)
                cur_t += 1
            print("Epoch: {} Iteration: {} VLB: {} Rec Loss: {}".format(e,
                                                                        cur_t,
                                                                        lb,
                                                                        rl,
                                                                        ))
            print("Annealing coefficient: {} KLD: {}".format(ann, kl))
            val_vlb, val_rec = [], []
            for f_images_batch, captions_batch, cl_batch, c_v in val_gen.next_batch(
                use_obj_vectors=params.use_c_v):
                feed = {image_f_inputs: f_images_batch,
                        ann_inputs_enc: captions_batch[1],
                        ann_inputs_dec: captions_batch[0],
                        ann_lengths: cl_batch,
                        anneal: cur_t}
                if params.use_c_v:
                    feed.update({c_i: c_v[:, 1:]})
                kl, rl, lb = sess.run([kld, rec_loss, lower_bound],
                                      feed_dict=feed)
                val_vlb.append(lb)
                val_rec.append(rl)
            print("Validation VLB: {} Rec_loss: {}".format(np.mean(val_vlb),
                                                           np.mean(val_rec)))
            print("-----------------------------------------------")
            # anneal lr
            if e % 5 == 0 and cur_t != 0:
                params.learning_rate = params.learning_rate / 2
        # save model
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        # builder.add_meta_graph_and_variables(sess, ["main_model"])
        if params.num_epochs > 0:
            save_path = saver.save(sess, "./checkpoints/{}.ckpt".format(
                params.checkpoint))
            print("Model saved in file: %s" % save_path)
        # run inference
        # validation set
        captions_gen = []
        print("Generating captions for val file")
        acc, caps = [], []
        for f_images_batch, _, _, image_ids, c_v in val_gen.next_batch(
            get_image_ids=True, use_obj_vectors=params.use_c_v):
            if params.use_c_v:
                # 0 element doesnt matter
                c_v = c_v[:, 1:]
            if params.sample_gen == 'beam_search':
                sent = decoder.beam_search(sess, image_ids, f_images_batch,
                                           image_f_inputs, c_v,
                                           beam_size=params.beam_size)
            else:
                sent, _ = decoder.online_inference(sess, image_ids, f_images_batch,
                                                   image_f_inputs, c_v=c_v)
            captions_gen += sent
        print("Generated {} captions".format(len(captions_gen)))
        val_gen_file = "./val_{}.json".format(params.gen_name)
        if os.path.exists(val_gen_file):
            print("Exists ", val_gen_file)
            os.remove(val_gen_file)
        with open(val_gen_file, 'w') as wj:
            print("saving val json file into ", val_gen_file)
            json.dump(captions_gen, wj)
        # test set
        captions_gen = []
        print("Generating captions for test file")
        for f_images_batch, image_ids, c_v in test_gen.next_test_batch(
            params.use_c_v):
            if params.use_c_v:
                c_v = c_v[:, 1:]
            sent, _ = decoder.online_inference(sess, image_ids, f_images_batch,
                                               image_f_inputs, c_v=c_v)
            captions_gen += sent
        test_gen_file = "./test_{}.json".format(params.gen_name)
        if os.path.exists(test_gen_file):
            os.remove(test_gen_file)
        with open(test_gen_file, 'w') as wj:
            print("saving test json file into", test_gen_file)
            json.dump(captions_gen, wj)

if __name__ == '__main__':
    params = Parameters()
    params.parse_args()
    coco_dir = params.coco_dir
    # save parameters for futher usage
    if params.save_params:
        import pickle
        param_fn = "./pickles/params_{}_{}_{}_{}.pickle".format(params.prior,
                                        params.no_encoder,
                                        params.checkpoint,
                                        params.use_c_v)
        print("Saving params to: ", param_fn)
        with open(param_fn, 'wb') as wf:
            pickle.dump(file=wf, obj=params)
    # train model, generate captions for val-test sets
    main(params)
