# data class
import tensorflow as tf
from glob import glob
from tqdm import tqdm
import numpy as np
import pickle
import os

from utils.batch_gen import Batch_Generator
from utils.captions import Captions, Dictionary

class Data():
    def __init__(self, coco_path, extract_features=False, ex_features_model=None):
        # captions
        # TODO: implement test captions evaluations on mscoco server
        self.train_cap_json = coco_path + "annotations/captions_train2014.json"
        self.valid_cap_json = coco_path + "annotations/captions_val2014.json"
        # image paths
        self.train_dir = coco_path + "images/train2014/"
        self.valid_dir = coco_path + "images/val2014/"
        self.test_dir = coco_path + "images/test2014/"
        # load captions into objects
        self.captions_tr = Captions(self.train_cap_json)
        self.captions_val = Captions(self.valid_cap_json)
        # form dictionary (idx to words and words to idx)
        self.dictionary = Dictionary(self.captions_tr.captions)
        self.captions_tr.index_captions(self.dictionary.word2idx)
        self.captions_val.index_captions(self.dictionary.word2idx)
        self.train_feature_dict = None
        self.ex_features_model = None
        if extract_features:
            assert ex_features_model != None, "Specify tf.contrib.keras model"
            # prepare image features or load them from pickle file
            self.train_feature_dict = self.extract_features(self.train_dir, ex_features_model)
            self.ex_features_model = ex_features_model

    def load_train_data_generator(self, batch_size):
        """
        Args:
            batch_size: batch size
            pre_extr_features_model: keras VGG16 model, ex.:
        model = tf.contrib.keras.applications.VGG16(weights='imagenet', include_top=False)
        """
        feature_dict = self.train_feature_dict
        self.train_batch_gen = Batch_Generator(self.train_dir,
                                              self.train_cap_json,
                                              self.captions_tr,
                                              batch_size,
                                              feature_dict=feature_dict)
        return self.train_batch_gen

    def extract_features(self, data_dir, model=None, save_pickle=True, im_shape=(224, 224)):
        """
        Args:
            data_dir: image data directory
            model: tf.contrib.keras model, CNN, used for feature extraction
            save_pickle: bool, will serialize feature_dict and save it into ./pickle directory
            im_shape: desired images shape
        Returns:
            feature_dict: dictionary of the form {image_name: feature_vector}
        """
        feature_dict = {}
        assert model != None, "Specify tf.contrib.keras model"
        if not os.path.exists("./pickles"):
            os.makedirs("./pickles")
        try:
            with open("./pickles/" + data_dir.split('/')[-2] + '.pickle', 'rb') as rf:
                print("Loading prepared feature vector from {}".format("./pickles/" + data_dir.split('/')[-2] + '.pickle'))
                feature_dict = pickle.load(rf)
        except:
            print("Extracting features")
            for img_path in tqdm(glob(data_dir + '*.jpg')):
                img = tf.contrib.keras.preprocessing.image.load_img(img_path, target_size=im_shape)
                x = tf.contrib.keras.preprocessing.image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = tf.contrib.keras.applications.vgg16.preprocess_input(x)
                features = model.predict(x)
                # ex. COCO_val2014_0000000XXXXX.jpg
                feature_dict[img_path.split('/')[-1]] = features
            if save_pickle:
                with open("./pickles/" + data_dir.split('/')[-2] + '.pickle', 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict

    def get_valid_data(self, val_batch_size=None):
        """
        Get validation data, used Batch_Generator() without specifying batch
        size parameter (meaning will generate all data at once) for convenience.
        """
        valid_feature_dict = self.extract_features(self.valid_dir, self.ex_features_model)
        self.valid_batch_gen = Batch_Generator(self.valid_dir,
                                              self.valid_cap_json,
                                              self.captions_val,
                                              val_batch_size,
                                              feature_dict=valid_feature_dict,
                                              get_image_ids=True)
        return self.valid_batch_gen

    def get_test_data(self, test_batch_size=None):
        """
        Get test data images, evaluation is done on a test server.
        Args:
            test_batch_size: set size of generated batches
        Returns:
            Test batch generator
        """
        # TODO: finish implementation
        test_feature_dict = self.extract_features(self.train_dir, self.ex_features_model)
        self.train_batch_gen = Batch_Generator(self.train_dir,
                                               batch_size=test_batch_size,
                                               feature_dict=test_feature_dict,
                                               get_image_ids=True)
