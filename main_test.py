# test batch generation
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]='1'
coco_dir = "/home/luoyy16/datasets-large/mscoco/coco/"
import tensorflow as tf
import numpy as np

from utils.data import Data

sess = tf.InteractiveSession()

#data.exract_features(data.valid_dir, model)
#data.exract_features(data.test_dir, model)
#data.exract_features(data.train_dir, model)
repartiton = True
data = Data(params, True, None,
            repartiton=repartiton, gen_val_cap=params.gen_val_captions)

# load batch generator, repartiton to use more val set images in train
batch_gen = data.load_train_data_generator(10)
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
val_gen = data.get_valid_data(10,
                              val_tr_unused=batch_gen.unused_cap_in,
                              pretrained=False)
for f_images_batch, captions_batch, cl_batch, c_v in batch_gen.next_batch():
    f_images_batch = data.preprocess_images(f_images_batch)
    feed = {images: f_images_batch}
    f_vector = sess.run(features, feed)
    print(f_vector)
    break
