import os
from utils.data import Data
from utils.rnn_model import make_rnn_cell, rnn_placeholders
from utils.parameters import Parameters
from vae_model.decoder import Decoder
from vae_model.encoder import Encoder

import json
import numpy as np
import tensorflow as tf
import zhusuan as zs
from tensorflow import layers
from tensorflow.python.util.nest import flatten
# import utils
print(tf.__version__)


# for embeddings use pretrained VGG16, fine tune?
# encoder - decoder, try to write class? Look at zhusuan new classes

def main():
    # load data, class data contains captions, images, image features (if avaliable)
    base_model = tf.contrib.keras.applications.VGG16(weights='imagenet',
                                                     include_top=True)
    model = tf.contrib.keras.models.Model(inputs=base_model.input,
                                          outputs=base_model.get_layer('fc2').output)
    data = Data(coco_dir, True, model)
    val_gen = data.get_valid_data(500)
    # load batch generator
    batch_gen = data.load_train_data_generator(params.batch_size)
    # annotations vector of form <EOS>...<BOS><PAD>...
    ann_inputs_enc = tf.placeholder(tf.int32, [None, None])
    ann_inputs_dec = tf.placeholder(tf.int32, [None, None])
    ann_lengths = tf.placeholder(tf.int32, [None])
    # use prepared image features [batch_size, 4096] (fc2)
    image_f_inputs = tf.placeholder(tf.float32, [None, 4096])
    # dictionary
    cap_dict = data.dictionary
    params.vocab_size = cap_dict.vocab_size
    # encoder, input fv and ...<BOS>,get z
    encoder = Encoder(image_f_inputs, ann_inputs_enc, ann_lengths, params)
    qz = encoder.q_net()
    # decoder, input_fv, get x, x_logits (for generation)
    decoder = Decoder(image_f_inputs, ann_inputs_dec, ann_lengths, params, cap_dict)
    with tf.variable_scope("decoder"):
        dec_model, x_logits, shpe, _ = decoder.px_z_fi({'z': qz})
    # generation, gen_states-tuple, z~N(0, 1 (or another variance))
    # decoder.px_z_fi({})
    # calculate rec. loss, mask padded part
    labels_flat = tf.reshape(ann_inputs_enc, [-1])
    prnt1 = tf.Print(labels_flat, [shpe[0], tf.shape(labels_flat), shpe[1], shpe[2]],
                     message="shape")
    ce_loss_padded = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=x_logits,
                                                                labels=labels_flat)
    loss_mask = tf.sign(tf.to_float(labels_flat))
    masked_loss = loss_mask * ce_loss_padded
    # restore original shape
    masked_loss = tf.reshape(masked_loss, tf.shape(ann_inputs_enc))
    mean_loss_by_example = tf.reduce_sum(masked_loss, 1) / tf.to_float(ann_lengths)
    rec_loss = tf.reduce_mean(mean_loss_by_example)
    # kld, see Kingma et.al
    kld = -0.5 * tf.reduce_mean(tf.reduce_sum(1 + qz.distribution.logstd
                                                  - tf.square(qz.distribution.mean)
                                                  - tf.exp(qz.distribution.logstd), 1))
    # kld weight annealing
    anneal = tf.placeholder_with_default(0, [])
    annealing = (tf.tanh((tf.to_float(anneal) - 5500)/1000) + 1)/2
    # overall loss reconstruction loss - kl_regularization
    lower_bound = tf.reduce_mean(tf.to_float(ann_lengths)) * rec_loss + tf.multiply(tf.to_float(annealing), tf.to_float(kld))
    #lower_bound = rec_loss + tf.to_float(kld)
    #lower_bound = rec_loss
    # we need to maximize lower_bound
    gradients = tf.gradients(lower_bound, tf.trainable_variables())
    grads_vars = zip(gradients, tf.trainable_variables())
    # TODO: look tf doc for collections + look at vae theory to clarify loss calculation
    optimize = tf.train.AdamOptimizer(params.learning_rate).apply_gradients(grads_vars)
    # model restore
    saver = tf.train.Saver()
    with tf.Session() as sess:
        # TODO: look documentation: local-global variables
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        # train using batch generator, every iteration get
        # f(I), [batch_size, max_seq_len], seq_lengths
        cur_t = 0
        if params.restore:
            # TODO: add checkpoint naming
            saver.restore(sess, "./checkpoints/last_run.ckpt")
        for e in range(params.num_epochs):
            # TODO: add shuffle !
            i = 0
            for tr_f_images_batch, tr_captions_batch, tr_cl_batch in batch_gen.next_batch():
                feed = {image_f_inputs: tr_f_images_batch,
                        ann_inputs_enc: tr_captions_batch[1],
                        ann_inputs_dec: tr_captions_batch[0],
                        ann_lengths: tr_cl_batch,
                        anneal: cur_t}
                # debuging print
                if cur_t == 0:
                    _ = sess.run(prnt1, feed_dict=feed)
                kl, rl, lb, _ = sess.run([kld, rec_loss, lower_bound, optimize], feed_dict=feed)
                if i % 500 == 0 and i != 0:
                    print("Epoch: {} Iteration: {} VLB: {} Rec Loss: {} KLD: {}".format(e,
                                                                                        i,
                                                                                        lb,
                                                                                        rl,
                                                                                        kl))
                    val_vlb, val_rec = [], []
                    for f_images_batch, captions_batch, cl_batch in val_gen.next_batch():
                        feed = {image_f_inputs: f_images_batch,
                                ann_inputs_enc: captions_batch[1],
                                ann_inputs_dec: captions_batch[0],
                                ann_lengths: cl_batch,
                                anneal: cur_t}
                        kl, rl, lb = sess.run([kld, rec_loss, lower_bound], feed_dict=feed)
                        val_vlb.append(lb)
                        val_rec.append(rl)
                    print("Validation VLB: {} Rec_loss: {}".format(np.mean(val_vlb), np.mean(val_rec)))
                i += 1
                cur_t += 1
            # evaluate
            # captions_gen = []
            # print("Generating captions for val file")
            # for f_images_batch, captions_batch, cl_batch, image_ids in val_gen.next_batch(get_image_ids=True):
            #     captions_gen += decoder.online_inference(sess, image_ids, f_images_batch)
            # with open("./val_gen.json", 'w') as wj:
            #     json.dump(captions_gen, wj)
        # evaluate
        captions_gen = []
        print("Generating captions for val file")
        for f_images_batch, captions_batch, cl_batch, image_ids in val_gen.next_batch(get_image_ids=True):
            captions_gen += decoder.online_inference(sess, image_ids, f_images_batch)
        with open("./val_gen.json", 'w') as wj:
            print("saving json file")
            json.dump(captions_gen, wj)
        # save model
        if not os.path.exists("./checkpoints"):
            os.makedirs("./checkpoints")
        save_path = saver.save(sess, "./checkpoints/last_run.ckpt")
        print("Model saved in file: %s" % save_path)

if __name__ == '__main__':
    params = Parameters()
    params.parse_args()
    coco_dir = params.coco_dir
    main()
