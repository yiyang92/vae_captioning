# test batch generation

import tensorflow as tf
import numpy as np

from utils.data import Data

sess = tf.InteractiveSession()

coco_dir = "/home/luoyy/datasets_large/mscoco/coco/"
base_model = tf.contrib.keras.applications.VGG16(weights='imagenet',
                                                 include_top=True)
model = tf.contrib.keras.models.Model(inputs=base_model.input,
                                      outputs=base_model.get_layer('fc2').output)
#data.exract_features(data.valid_dir, model)
#data.exract_features(data.test_dir, model)
#data.exract_features(data.train_dir, model)
repartiton = True
data = Data(coco_dir, True, model, repartiton=repartiton,
            gen_val_cap=4000)

# load batch generator, repartiton to use more val set images in train
batch_gen = data.load_train_data_generator(10)

for f_images_batch, captions_batch, cl_batch, c_v in batch_gen.next_batch():
    # print(f_images_batch)
    break

images = tf.placeholder(tf.float32, [None, 224, 224, 3])
features = model(images)
print(features)

batch_gen = data.load_train_data_generator(10, fine_tune=True)
# for f_images_batch, captions_batch, cl_batch, c_v in batch_gen.next_batch():
#     f_images_batch = data.preprocess_images(f_images_batch)
#     feed = {images: f_images_batch}
#     #print(data.extract_features(f_images_batch[0]))
#     f_vector = sess.run(features, feed)
#     print(f_vector)
#     break
val_gen = data.get_valid_data(10,
                              val_tr_unused=batch_gen.unused_cap_in,
                              pretrained=False)

for f_images_batch, captions_batch, cl_batch, c_v in val_gen.next_batch():
    f_images_batch = data.preprocess_images(f_images_batch)
    feed = {images: f_images_batch}
    f_vector = sess.run(features, feed)
    print(f_vector)
    break
