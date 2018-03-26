# data class
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import os
import cv2
import h5py

from utils.batch_gen import Batch_Generator
from utils.captions import Captions, Dictionary
from utils.image_embeddings import vgg16
from utils.image_utils import load_image

class Data():
    def __init__(self, params, extract_features=False,
                 weights_path=None, repartiton=False, gen_val_cap=None):
        # captions
        coco_path = params.coco_dir
        self.params = params
        self.train_cap_json = coco_path + "annotations/captions_train2014.json"
        self.valid_cap_json = coco_path + "annotations/captions_val2014.json"
        self.test_cap_json = coco_path + "annotations/image_info_test2014.json"
        # image paths
        self.train_dir = coco_path + "images/train2014/"
        self.valid_dir = coco_path + "images/val2014/"
        self.test_dir = coco_path + "images/test2014/"
        # load captions into objects
        self.captions_tr = Captions(self.train_cap_json, params.cap_max_length)
        self.captions_val = Captions(self.valid_cap_json, params.cap_max_length)
        # form dictionary (idx to words and words to idx)
        self.dictionary = Dictionary(self.captions_tr.captions,
                                     params.keep_words)
        self.captions_tr.index_captions(self.dictionary.word2idx)
        self.captions_val.index_captions(self.dictionary.word2idx)
        self.train_feature_dict = None
        self.num_examples = self.captions_tr.num_captions
        self.repartiton = repartiton
        self.gen_val_cap = gen_val_cap
        if repartiton and not gen_val_cap:
            raise ValueError("If using repartition must specify how many val "
                             "images to use")
        if extract_features:
            # prepare image features or load them from pickle file
            self.weights_path = weights_path
            if not weights_path:
                raise ValueError("Specify imagenet weights path")
            self.train_feature_dict = self.extract_features_from_dir(
                self.train_dir)

    def load_train_data_generator(self, batch_size, fine_tune=False,
                                  usehdf5=True):
        """
        Args:
            batch_size: batch size
            pre_extr_features_model: keras VGG16 model, ex.:
        model = tf.contrib.keras.applications.VGG16(weights='imagenet',
        include_top=False)
        """
        feature_dict = self.train_feature_dict
        val_cap, valid_feature_dict = None, None
        if self.repartiton:
            val_cap = self.captions_val
            valid_feature_dict = self.extract_features_from_dir(self.valid_dir)
        if fine_tune or not feature_dict:
            self.train_batch_gen = Batch_Generator(self.train_dir,
                                                  self.train_cap_json,
                                                  self.captions_tr,
                                                  batch_size,
                                                  use_hdf5=self.params.use_hdf5,
                                                  hdf5_file=self.params.hdf5_file,
                                                  feature_dict=None)
        else:
            self.train_batch_gen = Batch_Generator(self.train_dir,
                                                  self.train_cap_json,
                                                  self.captions_tr,
                                                  batch_size,
                                                  feature_dict=feature_dict)
        if self.repartiton:
            self.train_batch_gen.repartiton(val_cap,
                                            valid_feature_dict,
                                            self.gen_val_cap)
        return self.train_batch_gen

    def extract_features_from_dir(self, data_dir, save_pickle=True,
                                  im_shape=(224, 224)):
        """
        Args:
            data_dir: image data directory
            model: tf.contrib.keras model, CNN, used for feature extraction
            save_pickle: bool, will serialize feature_dict and save it into
        ./pickle directory
            im_shape: desired images shape
        Returns:
            feature_dict: dictionary of the form {image_name: feature_vector}
        """
        feature_dict = {}
        try:
            with open(
                "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'rb') as rf:
                print("Loading prepared feature vector from {}".format(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle'))
                feature_dict = pickle.load(rf)
        except:
            print("Extracting features")
            if not os.path.exists("./pickles"):
                os.makedirs("./pickles")
            im_embed = tf.Graph()
            with im_embed.as_default():
                input_img = tf.placeholder(tf.float32, [None,
                                                        im_shape[0],
                                                        im_shape[1], 3])
                image_embeddings = vgg16(input_img)
                features = image_embeddings.fc2
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
            with tf.Session(graph=im_embed) as sess:
                image_embeddings.load_weights(self.weights_path, sess)
                for img_path in tqdm(glob(data_dir + '*.jpg')):
                    img = load_image(img_path)
                    img = np.expand_dims(img, axis=0)
                    f_vector = sess.run(features, {input_img: img})
                    # ex. COCO_val2014_0000000XXXXX.jpg
                    feature_dict[img_path.split('/')[-1]] = f_vector
            if save_pickle:
                with open(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict

    def get_valid_data(self, val_batch_size=None, val_tr_unused=None,
                       pretrained=True):
        """
        Get validation data, used Batch_Generator() without specifying batch
        size parameter (meaning will generate all data at once) for convenience.
        """
        if pretrained:
            valid_feature_dict = self.extract_features_from_dir(self.valid_dir)
        else:
            valid_feature_dict = None
        self.valid_batch_gen = Batch_Generator(self.valid_dir,
                                              self.valid_cap_json,
                                              self.captions_val,
                                              val_batch_size,
                                              feature_dict=valid_feature_dict,
                                              get_image_ids=True,
                                              val_tr_unused=val_tr_unused,
                                              use_hdf5=self.params.use_hdf5,
                                              hdf5_file=self.params.hdf5_file)
        return self.valid_batch_gen

    def get_test_data(self, test_batch_size=None, pretrained=True):
        """
        Get test data images, evaluation is done on a test server.
        Args:
            test_batch_size: set size of generated batches
            pretrained: whether or not use presaved imagenet features
        Returns:
            Test batch generator
        """
        if pretrained:
            test_feature_dict = self.extract_features_from_dir(self.test_dir)
        else:
            test_feature_dict = None
        self.train_batch_gen = Batch_Generator(self.test_dir,
                                               train_cap_json=self.test_cap_json,
                                               batch_size=test_batch_size,
                                               feature_dict=test_feature_dict,
                                               get_image_ids=True,
                                               get_test_ids=True)
        return self.train_batch_gen
