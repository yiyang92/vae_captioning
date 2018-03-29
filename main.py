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
from utils.image_embeddings import vgg16
from utils.caption_utils import preprocess_captions
# vae model
from vae_model.decoder import Decoder
from vae_model.encoder import Encoder
from ops import inference, optimizers

print("Tensorflow version: ", tf.__version__)

def main(params):
    # load data, class data contains captions, images, image features (if avaliable)
    if params.gen_val_captions < 0:
        repartiton = False
    else:
        repartiton = True
    data = Data(params, True, params.image_net_weights_path,
                repartiton=repartiton, gen_val_cap=params.gen_val_captions)
    # load batch generator, repartiton to use more val set images in train
    gen_batch_size = params.batch_size
    if params.fine_tune:
        gen_batch_size = params.batch_size
    batch_gen = data.load_train_data_generator(gen_batch_size,
                                               params.fine_tune)
    # whether use presaved pretrained imagenet features (saved in pickle)
    # feature extractor after fine_tune will be saved in tf checkpoint
    # caption generation after fine_tune must be made with params.fine_tune=True
    pretrained = not params.fine_tune
    val_gen = data.get_valid_data(gen_batch_size,
                                  val_tr_unused=batch_gen.unused_cap_in,
                                  pretrained=pretrained)
    test_gen = data.get_test_data(gen_batch_size,
                                  pretrained=pretrained)
    # annotations vector of form <EOS>...<BOS><PAD>...
    ann_inputs_enc = tf.placeholder(tf.int32, [None, None])
    ann_inputs_dec = tf.placeholder(tf.int32, [None, None])
    ann_lengths = tf.placeholder(tf.int32, [None])
    if params.fine_tune:
        # if fine_tune dont just load images_fv
        image_f_inputs = tf.placeholder(tf.float32, [None, 224, 224, 3])
    else:
        # use prepared image features [batch_size, 4096] (fc2)
        image_f_inputs = tf.placeholder(tf.float32, [None, 4096])
    if params.use_c_v or (
        params.prior == 'GMM' or params.prior == 'AG'):
        c_i = tf.placeholder(tf.float32, [None, 90])
    else:
        c_i = ann_lengths # dummy tensor
    # because of past changes
    image_batch, cap_enc, cap_dec, cap_len, cl_vectors = image_f_inputs,\
    ann_inputs_enc, ann_inputs_dec, ann_lengths, c_i
    # features, params.fine_tune stands for not using presaved imagenet weights
    # here, used this dummy placeholder during fine_tune, will remove it in
    # future releases, thats for saving image_net weights for futher usage
    image_f_inputs2 = tf.placeholder_with_default(
        tf.ones([1, 224, 224, 3]), shape=[None, 224, 224, 3], name='dummy_ps')
    if params.fine_tune:
        image_f_inputs2 = image_batch
    if params.mode == 'training' and params.fine_tune:
        cnn_dropout = params.cnn_dropout
        weights_regularizer = tf.contrib.layers.l2_regularizer(
            params.weight_decay)
    else:
        cnn_dropout = 1.0
        weights_regularizer = None
    with tf.variable_scope("cnn", regularizer=weights_regularizer):
        image_embeddings = vgg16(image_f_inputs2,
                                 trainable_fe=params.fine_tune_fe,
                                 trainable_top=params.fine_tune_top,
                                 dropout_keep=cnn_dropout)
    if params.fine_tune:
        features = image_embeddings.fc2
    else:
        features = image_batch
    # forward pass is expensive, so can use this method to reduce computation
    if params.num_captions > 1 and params.mode == 'training': #[batch_size, 4096]
        features_tiled = tf.tile(tf.expand_dims(features, 1),
                                 [1, params.num_captions, 1])
        features = tf.reshape(features_tiled,
                              [tf.shape(features)[0] * params.num_captions,
                               params.cnn_feature_size])
    # dictionary
    cap_dict = data.dictionary
    params.vocab_size = cap_dict.vocab_size
    # image features [b_size + f_size(4096)] -> [b_size + embed_size]
    images_fv = layers.dense(features, params.embed_size, name='imf_emb')
    # images_fv = tf.Print(images_fv, [tf.shape(features), features[0][0:10],
    #                                   image_embeddings.imgs[0][:10], images_fv])
    # encoder, input fv and ...<BOS>,get z
    if not params.no_encoder:
        encoder = Encoder(images_fv, cap_enc, cap_len, params)
    # decoder, input_fv, get x, x_logits (for generation)
    decoder = Decoder(images_fv, cap_dec, cap_len, params,
                      cap_dict)
    if params.use_c_v or (
        params.prior == 'GMM' or params.prior == 'AG'):
        # cluster vectors from "Diverse and Accurate Image Description.." paper.
        # 80 is number of classes, for now hardcoded
        # for GMM-CVAE must be specified
        c_i_emb = layers.dense(cl_vectors, params.embed_size, name='cv_emb')
        # map cluster vectors into embedding space
        decoder.c_i = c_i_emb
        decoder.c_i_ph = cl_vectors
        if not params.no_encoder:
            encoder.c_i = c_i_emb
            encoder.c_i_ph = cl_vectors
    if not params.no_encoder:
        with tf.variable_scope("encoder"):
            qz, tm_list, tv_list = encoder.q_net()
        def init_clusters(num_clusters):
            # initialize sigma as constant, mu drawn randomly
            z_size = params.latent_size
            c_sigma = tf.constant(0.1)
            cluster_mu_matrix = []
            for id_cluster in range(params.num_clusters):
                with tf.variable_scope("cl_init_mean_{}".format(id_cluster)):
                    cluster_item = 2*np.random.random_sample(
                        (1, params.latent_size)) - 1
                    cluster_item = cluster_item/(tf.sqrt(
                        tf.reduce_sum(cluster_item**2)))
                    cluster_mu_matrix.append(tf.cast(cluster_item, tf.float32))
            c_means = tf.concat(cluster_mu_matrix, 0)
            return c_means, c_sigma
        if params.prior == 'Normal':
            # kld between normal distributions KL(q, p), see Kingma et.al
            kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + tf.log(tf.square(qz.distribution.std) + 0.00001)
                    - tf.square(qz.distribution.mean)
                    - tf.square(qz.distribution.std), 1))
        elif params.prior == 'GMM':
            # initialize sigma as constant, mu drawn randomly
            # TODO: finish GMM loss implementation
            c_means, c_sigma = init_clusters(90)
            kld = -0.5 * tf.reduce_mean(
                tf.reduce_sum(
                    1 + tf.log(tf.square(qz.distribution.std) + 0.00001)
                    - tf.square(qz.distribution.mean)
                    - tf.square(qz.distribution.std), 1))
        elif params.prior == 'AG':
            c_means, c_sigma = init_clusters(90)
            kld_clusters = 1 + tf.log(qz.distribution.std+ 0.00001)\
             -  tf.log(c_sigma + 0.00001) - (
                 tf.square(qz.distribution.mean - tf.matmul(
                     tf.squeeze(c_i), c_means)) + tf.square(
                         qz.distribution.std))/(tf.square(c_sigma)+0.0000001)
            kld = -0.5 * tf.reduce_sum(kld_clusters, 1)
    with tf.variable_scope("decoder"):
        if params.no_encoder:
            dec_model, x_logits, shpe, _ = decoder.px_z_fi({})
        else:
            dec_model, x_logits, shpe, _ = decoder.px_z_fi({'z': qz})
    # calculate rec. loss, mask padded part
    labels_flat = tf.reshape(cap_enc, [-1])
    ce_loss_padded = tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=x_logits, labels=labels_flat)
    loss_mask = tf.sign(tf.to_float(labels_flat))
    batch_loss = tf.div(tf.reduce_sum(tf.multiply(ce_loss_padded, loss_mask)),
                          tf.reduce_sum(loss_mask),
                          name="batch_loss")
    tf.losses.add_loss(batch_loss)
    rec_loss = tf.losses.get_total_loss()
    # kld weight annealing
    anneal = tf.placeholder_with_default(0, [])
    if params.fine_tune:
        annealing = tf.constant(1.0)
    else:
        annealing = (tf.tanh(
            (tf.to_float(anneal) - 1000 * params.ann_param)/1000) + 1)/2
    # overall loss reconstruction loss - kl_regularization
    if not params.no_encoder:
        lower_bound = rec_loss + tf.multiply(
                tf.to_float(annealing), tf.to_float(kld))/10
    else:
        lower_bound = rec_loss
        kld = tf.constant(0.0)
    # optimization, can print global norm for debugging
    optimize, global_step, global_norm = optimizers.non_cnn_optimizer(lower_bound,
                                                                      params)
    optimize_cnn = tf.constant(0.0)
    if params.fine_tune and params.mode == 'training':
        optimize_cnn, _ = optimizers.cnn_optimizer(lower_bound, params)
    # cnn parameters update
    # model restore
    vars_to_save = tf.trainable_variables()
    if not params.fine_tune_fe or not params.fine_tune_top:
        cnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'cnn')
        vars_to_save += cnn_vars
    saver = tf.train.Saver(vars_to_save,
                           max_to_keep=params.max_checkpoints_to_keep)
    # m_builder = tf.saved_model.builder.SavedModelBuilder('./saved_model')
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run([tf.global_variables_initializer(),
                  tf.local_variables_initializer()])
        # train using batch generator, every iteration get
        # f(I), [batch_size, max_seq_len], seq_lengths
        if params.mode == "training":
            if params.logging:
                summary_writer = tf.summary.FileWriter(params.LOG_DIR,
                                                       sess.graph)
                summary_writer.add_graph(sess.graph)
            if not params.restore:
                print("Loading imagenet weights for futher usage")
                image_embeddings.load_weights(params.image_net_weights_path,
                                              sess)
            if params.restore:
                print("Restoring from checkpoint")
                saver.restore(sess, "./checkpoints/{}.ckpt".format(
                    params.checkpoint))
            for e in range(params.num_epochs):
                gs = tf.train.global_step(sess, global_step)
                gs_epoch = 0
                while True:
                    def stop_condition():
                        num_examples = gs_epoch * params.batch_size
                        if num_examples > params.num_ex_per_epoch:
                            return True
                        return False
                    for f_images_batch,\
                    captions_batch, cl_batch, c_v in batch_gen.next_batch(
                        use_obj_vectors=params.use_c_v,
                        num_captions=params.num_captions):
                        if params.num_captions > 1:
                            captions_batch, cl_batch, c_v = preprocess_captions(
                                captions_batch, cl_batch, c_v)
                        feed = {image_f_inputs: f_images_batch,
                                ann_inputs_enc: captions_batch[1],
                                ann_inputs_dec: captions_batch[0],
                                ann_lengths: cl_batch,
                                anneal: gs}
                        if params.use_c_v or (
                            params.prior == 'GMM' or params.prior == 'AG'):
                            feed.update({c_i: c_v[:, 1:]})
                        gs = tf.train.global_step(sess, global_step)
                        feed.update({anneal: gs})
                        # if gs_epoch == 0:
                        # print(sess.run(debug_print, feed))
                        kl, rl, lb, _,_, ann = sess.run([kld, rec_loss,
                                                       lower_bound, optimize,
                                                       optimize_cnn, annealing],
                                                      feed)
                        gs_epoch += 1
                        if gs % 500 == 0:
                            print("Epoch: {} Iteration: {} VLB: {} "
                                  "Rec Loss: {}".format(e, gs, np.mean(lb),rl))
                            if not params.no_encoder:
                                print("Annealing coefficient:"
                                      "{} KLD: {}".format(ann, np.mean(kl)))
                        if stop_condition():
                            break
                    if stop_condition():
                        break
                print("Epoch: {} Iteration: {} VLB: {} Rec Loss: {}".format(e,
                                                                            gs,
                                                                            np.mean(lb),
                                                                            rl,
                                                                            ))
                val_vlb, val_rec = [], []
                def validate():
                    for f_images_batch, captions_batch, cl_batch, c_v in val_gen.next_batch(
                        use_obj_vectors=params.use_c_v,
                        num_captions=params.num_captions):
                        gs = tf.train.global_step(sess, global_step)
                        if params.num_captions > 1:
                            captions_batch, cl_batch, c_v= preprocess_captions(
                                captions_batch, cl_batch,c_v)
                        feed = {image_f_inputs: f_images_batch,
                                ann_inputs_enc: captions_batch[1],
                                ann_inputs_dec: captions_batch[0],
                                ann_lengths: cl_batch,
                                anneal: gs}
                        if params.use_c_v or (
                            params.prior == 'GMM' or params.prior == 'AG'):
                            feed.update({c_i: c_v[:, 1:]})
                        kl, rl, lb = sess.run([kld, rec_loss, lower_bound],
                                              feed_dict=feed)
                        val_vlb.append(lb)
                        val_rec.append(rl)
                    print("Validation VLB: {} Rec_loss: {}".format(np.mean(val_vlb),
                                                                   np.mean(val_rec)))
                    print("-----------------------------------------------")
                validate()
                # save model
                if not os.path.exists("./checkpoints"):
                    os.makedirs("./checkpoints")
                save_path = saver.save(sess, "./checkpoints/{}.ckpt".format(
                    params.checkpoint))
                print("Model saved in file: %s" % save_path)
        # builder.add_meta_graph_and_variables(sess, ["main_model"])
        if params.use_hdf5 and params.fine_tune:
            batch_gen.h5f.close()
        # run inference
        if params.mode == "inference":
            inference.inference(params, decoder, val_gen,
                                test_gen, image_f_inputs, saver, sess)

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
