# test batch generation

import tensorflow as tf
from utils.data import Data

coco_dir = "/home/luoyy16/datasets-large/mscoco/coco/"


data = Data(coco_dir)
model = tf.contrib.keras.applications.VGG16(weights='imagenet', include_top=False)
train_dir = coco_dir + "images/train2014/"
#data.exract_features(data.valid_dir, model)
data.exract_features(data.test_dir, model)
#data.exract_features(data.train_dir, model)
# for im_batch, cp_batch, ln_batch in test_batch_gen.next_batch():
#     #print(batch[0].shape())
#     print(im_batch.shape)
#     print(cp_batch.shape)
#     print(ln_batch.shape)
#     print(cp_batch[0:2])
#     feature = model.predict(im_batch)
#     break
                # print("train feature shape: ", tr_f_images_batch.shape)
                # print("train capions shape: ", tr_captions_batch[0].shape)
                # ann_inputs_enc = tf.placeholder(tf.int32, [None, None])
                # ann_inputs_dec = tf.placeholder(tf.int32, [None, None])
                # ann_lengths = tf.placeholder(tf.int32, [None])
                # image_f_inputs
                #print(params.vocab_size)
                #print(tr_captions_batch[0].shape)
