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
    def __init__(self, coco_path, extract_features=False,
                 ex_features_model=None, repartiton=False,
                 gen_val_cap=None):
        # captions
        self.train_cap_json = coco_path + "annotations/captions_train2014.json"
        self.valid_cap_json = coco_path + "annotations/captions_val2014.json"
        self.test_cap_json = coco_path + "annotations/image_info_test2014.json"
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
        self.num_examples = self.captions_tr.num_captions
        self.repartiton = repartiton
        self.gen_val_cap = gen_val_cap
        if repartiton and not gen_val_cap:
            raise ValueError("If using repartition must specify how many val "
                             "images to use")
        assert ex_features_model != None, "Specify tf.contrib.keras model"
        self.ex_features_model = ex_features_model
        if extract_features:
            # prepare image features or load them from pickle file
            self.train_feature_dict = self.extract_features_from_dir(self.train_dir,
                                                                     ex_features_model)

    def load_train_data_generator(self, batch_size, fine_tune=False):
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
            valid_feature_dict = self.extract_features_from_dir(self.valid_dir,
                                                   self.ex_features_model)
        if fine_tune or not feature_dict:
            self.train_batch_gen = Batch_Generator(self.train_dir,
                                                  self.train_cap_json,
                                                  self.captions_tr,
                                                  batch_size,
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

    def extract_features_from_dir(self, data_dir, model=None, save_pickle=True,
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
        assert model != None, "Specify tf.contrib.keras model"
        if not os.path.exists("./pickles"):
            os.makedirs("./pickles")
        try:
            with open(
                "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'rb') as rf:
                print("Loading prepared feature vector from {}".format(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle'))
                feature_dict = pickle.load(rf)
        except:
            print("Extracting features")
            for img_path in tqdm(glob(data_dir + '*.jpg')):
                img = tf.contrib.keras.preprocessing.image.load_img(img_path,
                                                                    target_size=im_shape)
                x = tf.contrib.keras.preprocessing.image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = tf.contrib.keras.applications.vgg16.preprocess_input(x)
                features = model.predict(x)
                # ex. COCO_val2014_0000000XXXXX.jpg
                feature_dict[img_path.split('/')[-1]] = features
            if save_pickle:
                with open(
                    "./pickles/" + data_dir.split('/')[-2] + '.pickle', 'wb') as wf:
                    pickle.dump(feature_dict, wf)
        return feature_dict

    def extract_features(self, image, im_shape=(224, 224, 3)):
        """
        Args:
            image: input image
        Returns:
            image features
        """
        image = np.resize(image, im_shape)
        image = np.expand_dims(image, axis=0)
        image = tf.contrib.keras.applications.vgg16.preprocess_input(image)
        return self.ex_features_model.predict(image)

    @staticmethod
    def preprocess_images(images, shape=(224, 224, 3)):
        """Preprocess for VGG16
        Args:
            images: np.array of shape [batch_size, None, None, 3]
        Returns:
            np.array of shape [batch_size, 224, 224, 3]
        """
        im_list = []
        for image in images:
            image = np.resize(image, shape)
            image = tf.contrib.keras.applications.vgg16.preprocess_input(image)
            im_list.append(image)
        return np.stack(im_list)

    def get_valid_data(self, val_batch_size=None, val_tr_unused=None,
                       pretrained=True):
        """
        Get validation data, used Batch_Generator() without specifying batch
        size parameter (meaning will generate all data at once) for convenience.
        """
        if pretrained:
            valid_feature_dict = self.extract_features_from_dir(self.valid_dir,
                                                       self.ex_features_model)
        else:
            valid_feature_dict = None
        self.valid_batch_gen = Batch_Generator(self.valid_dir,
                                              self.valid_cap_json,
                                              self.captions_val,
                                              val_batch_size,
                                              feature_dict=valid_feature_dict,
                                              get_image_ids=True,
                                              val_tr_unused=val_tr_unused)
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
            test_feature_dict = self.extract_features_from_dir(self.test_dir,
                                                      self.ex_features_model)
        else:
            test_feature_dict = None
        self.train_batch_gen = Batch_Generator(self.test_dir,
                                               train_cap_json=self.test_cap_json,
                                               batch_size=test_batch_size,
                                               feature_dict=test_feature_dict,
                                               get_image_ids=True,
                                               get_test_ids=True)
        return self.train_batch_gen
