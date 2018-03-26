import os
from glob import glob
import numpy as np
import cv2
import json
import pickle
import tensorflow as tf
import tqdm
import h5py

from random import shuffle
from utils.captions import Captions
from utils.image_utils import load_image

# batch generator
class Batch_Generator():
    def __init__(self, train_dir, train_cap_json=None,
                 captions=None, batch_size=None,
                 use_hdf5=False,
                 hdf5_file=None,
                 feature_dict=None,
                 get_image_ids=False, get_test_ids=False,
                 val_tr_unused=None):
        """
        Args:
            train_dir: coco training images directory path
            train_cap_json: json with coco annotations
            captions : instance of Captions() class
            batch_size: batch size, can be changed later
            feature_dict: if given, use feauture vectors instead of images in generator
            get_image_ids: whether or not return image_id list (used for test/val)
            val_tr_unused : (optional), dont use used in training for validation
        """
        self.use_hdf5 = use_hdf5
        if self.use_hdf5:
            if not hdf5_file:
                raise ValueError("Specify hdf5 file path")
            with open('./pickles/itoi.pickle', 'rb') as rf:
                self.imtoi = pickle.load(rf)
            # "images"
            self.h5f = h5py.File(hdf5_file, 'r')
            self.images = self.h5f['images']
        self._batch_size = batch_size
        if val_tr_unused == None:
            self._iterable = list(glob(train_dir + '*.jpg'))
        else:
            print("Val captions for generation : ", len(val_tr_unused))
            self._iterable = val_tr_unused
        self._train_dir = train_dir
        if not batch_size:
            print("use all data")
            self._batch_size = len(self._iterable)
        if len(self._iterable) == 0:
            print("Check images files avaliability")
            print("Coco dir: ", train_dir)
            raise FileNotFoundError
        # test set doesnt contain true captions
        self._train_cap_json = train_cap_json
        if get_test_ids:
            self._fn_to_id = self._test_images_to_imid()
        if captions:
            self.cap_instance = captions
            self.captions = self.cap_instance.captions_indexed
        # seed for reproducibility
        self.random_seed = 42
        np.random.seed(self.random_seed)
        self.feature_dict = feature_dict
        self.get_image_ids = get_image_ids
        self.unused_cap_in = None

    def repartiton(self, val_cap_instance, val_feature_dict, gen_val_cap):
        self.gen_val_cap = gen_val_cap
        if not val_cap_instance:
            raise ValueError("If use validation set images for "
                             "training need to specify val_cap instance")
        self.val_cap_instance = val_cap_instance
        self.val_captions = self.val_cap_instance.captions_indexed
        # assume that validation data stored in coco folder
        val_set_path = '/'.join(self._train_dir.split('/')[:-2] + ['val2014/'])
        val_list = list(glob(val_set_path + '*.jpg'))
        shuffle(val_list)
        # choose some images for validation, get images unused in training
        if self.gen_val_cap != None and self.gen_val_cap < 0:
            self.gen_val_cap = None
        # get unused captions image names
        if self.gen_val_cap:
            self._iterable.extend(val_list[:-self.gen_val_cap])
            self.unused_cap_in = val_list[-self.gen_val_cap:]
        else:
            self._iterable.extend(val_list)
        print("Train + Validation set size (use repartition): ", len(
            self._iterable))
        if not val_feature_dict:
            raise ValueError("If use validation set images for "
                             "training need to specify val_feature_dict")
        self.val_feature_dict = val_feature_dict

    def _images_c_v(self, imn_batch, c_v, indices=None):
        """Internal method, returns [batch_size, I] and [batch_size, c(I)]
        Returns:
            images, cl_v: images and cluster vectors
        """
        images, cl_v = [], []
        for imn in imn_batch:
            if not c_v and not self.feature_dict:
                break
            def get_features(imn):
                try:
                    image = self.feature_dict[imn.split('/')[-1]]
                except:
                    image = self.val_feature_dict[imn.split('/')[-1]]
                images.append(image)
            if c_v:
                try:
                    vector = c_v[imn.split('/')[-1]]
                except:
                    vector = np.zeros(91)
                cl_v.append(vector)
            if self.feature_dict:
                get_features(imn)
        cl_v = np.array(cl_v)
        if self.feature_dict:
            images = np.squeeze(np.array(images), 1)
        else:
            images = self._get_images(imn_batch, indices)
        return images, cl_v

    def _get_imid(self, imn_batch, test=False):
        """Internal method, get image ids using Caption object dict
        """
        image_ids = []
        for fn in imn_batch:
            if not test:
                try:
                    id_ = self.cap_instance.filename_to_imid[
                        fn.split('/')[-1]]
                except:
                    id_ = self.val_cap_instance.filename_to_imid[
                        fn.split('/')[-1]]
                image_ids.append(id_)
            else:
                for fn in imn_batch:
                    id_ = self._fn_to_id[fn.split('/')[-1]]
                    image_ids.append(id_)
        return image_ids

    def _next_imn(self):
        indices = np.random.choice(range(len(self._iterable)),
                                self._batch_size, replace=False)
        return np.array(self._iterable)[indices]

    def _get_indices(self, imn_batch):
        """Get indices+sorted names of images in hdf5 file.
        """
        imn_index = []
        for name in imn_batch:
            name = name.split('/')[-1]
            index = self.imtoi[name]
            imn_index.append((name, index))
        imn_index = sorted(imn_index, key= lambda x: (x[1], x[0]))
        imn_batch, indices = list(zip(*imn_index)) # return tuples
        return list(imn_batch), list(indices)

    def next_batch(self, use_obj_vectors=False, num_captions=1):
        """
        Args:
            use_obj_vectors: whether or not include object vectors
            num_captions: number of captions, if 1 will return randomly
        selected caption. All captions are padded to maximum length of the batch
        """
        # separately specify whether to use cluster obj_vectors
        self.use_obj_vectors = use_obj_vectors
        if self.use_obj_vectors:
            c_v = self._get_cluster_vectors()
        else:
            c_v = None
        # if select inly one caption
        random_select = False
        if num_captions == 1:
            random_select = True
        # separately specify whether to use cluster obj_vectors
        imn_batch  = [None] * self._batch_size
        shuffle(self._iterable)
        for i, item in enumerate(self._iterable):
            inx = i % self._batch_size
            imn_batch[inx] = item
            if inx == self._batch_size - 1:
                indices = None
                if self.use_hdf5:
                    imn_batch, indices = self._get_indices(imn_batch)
                images, cl_v = self._images_c_v(imn_batch, c_v, indices)
                # concatenate to obtain [images, caption_indices, lengths]
                inp_captions, l_captions, lengths = self._form_captions_batch(
                    imn_batch, random_select, num_captions)
                yield images, (inp_captions, l_captions), lengths, cl_v
                imn_batch = [None] * self._batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            indices = None
            if self.use_hdf5:
                imn_batch, indices = self._get_indices(imn_batch)
            images, cl_v = self._images_c_v(imn_batch, c_v, indices)
            inp_captions, l_captions, lengths = self._form_captions_batch(
                imn_batch, random_select, num_captions)
            yield images, (inp_captions, l_captions), lengths, cl_v

    def _test_images_to_imid(self):
        with open(self._train_cap_json) as rf:
            try:
                j = json.loads(rf.read())
            except FileNotFoundError as e:
                raise
        return {img['file_name']:img['id'] for img in j['images']}

    def next_val_batch(self, get_image_ids = False, use_obj_vectors=False):
        self.get_image_ids = get_image_ids
        # separately specify whether to use cluster obj_vectors
        self.use_obj_vectors = use_obj_vectors
        if self.use_obj_vectors:
            c_v = self._get_cluster_vectors()
        else:
            c_v = None
        imn_batch  = [None] * self._batch_size
        for i, item in enumerate(self._iterable):
            inx = i % self._batch_size
            imn_batch[inx] = item
            if inx == self._batch_size - 1:
                indices = None
                if self.use_hdf5:
                    imn_batch, indices = self._get_indices(imn_batch)
                images, cl_v = self._images_c_v(imn_batch, c_v, indices)
                # concatenate to obtain [images, caption_indices, lengths]
                inp_captions, l_captions, lengths = self._form_captions_batch(
                    imn_batch)
                if self.get_image_ids:
                    image_ids = self._get_imid(imn_batch)
                    yield images, (inp_captions,
                                   l_captions), lengths, image_ids, cl_v
                else:
                    yield images, (inp_captions, l_captions), lengths, cl_v
                imn_batch = [None] * self._batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            indices = None
            if self.use_hdf5:
                imn_batch, indices = self._get_indices(imn_batch)
            images, cl_v = self._images_c_v(imn_batch, c_v, indices)
            inp_captions, l_captions, lengths = self._form_captions_batch(
                imn_batch)
            if self.get_image_ids:
                image_ids = self._get_imid(imn_batch)
                yield images, (inp_captions,
                               l_captions), lengths, image_ids, cl_v
            else:
                yield images, (inp_captions, l_captions), lengths, cl_v

    def next_test_batch(self, use_obj_vectors=False):
        imn_batch  = [None] * self._batch_size
        self.use_obj_vectors = use_obj_vectors
        if self.use_obj_vectors:
            c_v = self._get_cluster_vectors(True)
        else:
            c_v = None
        for i, item in enumerate(self._iterable):
            inx = i % self._batch_size
            imn_batch[inx] = item
            if inx == self._batch_size - 1:
                images, cl_v = self._images_c_v(imn_batch, c_v)
                image_ids = self._get_imid(imn_batch, True)
                yield images, image_ids, cl_v
                imn_batch = [None] * self._batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            images, cl_v = self._images_c_v(imn_batch, c_v)
            image_ids = self._get_imid(imn_batch, True)
            yield images, image_ids, cl_v

    def _get_images(self, names, indices=None):
        """Load images
        Args:
            names: image filenames
            indices: using hdf5, indices of images in hdf5 file
        Returns:
            np.array of shape [batch_size, 224, 224, 3]
        """
        if self.use_hdf5:
            # indices must be in an increasing order
            return self.images[indices]
        else:
            images = []
            for name in names:
                img = load_image(name)
                images.append(img)
            return np.stack(images)

    def _form_captions_batch(self, imn_batch, random_select=True,
                             num_captions=1):
        """
        Args:
            imn_batch: image file names in the batch
            random_select: every time just choose random captions, not add all
            num_captions: number of captions for every image
        Returns :
            list of np arrays [[batch_size, caption], [lengths]], where lengths have
            batch_size shape
        """
        # use image_names to get caption, add padding, put it into numpy array
        # calculate length of every sequence and make a list
        # randomly choose caption for the current iteration
        # use static array for efficiency
        if random_select:
            num_captions = 1
        labels_captions_list = [[[0] for _ in range(
            num_captions)] for j in range(len(imn_batch))]
        input_captions_list = [[[0] for _ in range(
            num_captions)] for j in range(len(imn_batch))]
        lengths = np.zeros((len(imn_batch), num_captions))
        for idx, fn in enumerate(imn_batch):
            fn = fn.split('/')[-1]
            captions = self.captions[fn]
            if len(captions) == 0: # using defaultdict will not give error
                captions = self.val_captions[fn]
            if random_select:
                captions = [captions[np.random.randint(len(captions))]]
            # split into labels/inputs (encoder/decoder inputs)
            for i, caption in enumerate(captions):
                if i >= num_captions: # limit number of captions
                    break
                input_captions_list[idx][i] = caption[:-1] # <BOS>...
                labels_captions_list[idx][i] = caption[1:] # ...<EOS>
                lengths[idx][i] = len(caption) - 1
        # add padding and put captions into np array of shape [batch_size,
        # [max_batch_seq_len]*num_captions]
        pad = len(max([max(input_captions_list[i], key=len) for i in range(
            len(input_captions_list))], key=len))
        input_captions_list = np.array([[cap + [0] * (
            pad - len(cap)) for cap in caps] for caps in input_captions_list])
        labels_captions_list = np.array([[cap + [0] * (
            pad - len(cap)) for cap in caps] for caps in labels_captions_list])
        # if using random_select, temporary
        if input_captions_list.shape[1] == 1:
            input_captions_list = np.squeeze(input_captions_list, 1)
            labels_captions_list = np.squeeze(labels_captions_list, 1)
            lengths = np.squeeze(lengths)
        return input_captions_list, labels_captions_list, lengths

    def _get_cluster_vectors(self, load_test=False):
        """Load f_n: cluster_vector dictionary (for train/val).
        Args:
            load_test: for test dataset
        Return:
            custer vector dict {file_name: [cluster_vector]}
        """
        # load train and val, concatenate
        if load_test:
            T_V_PATH = './obj_vectors/c_v_test.pickle'
        else:
            T_V_PATH = './obj_vectors/c_v.pickle'
        with open(T_V_PATH, 'rb') as rf:
            c_v = pickle.load(rf)
        assert type(c_v) == dict, "cluster vector pickle must contain dict"
        return c_v

    @property
    def cap_dict(self):
        return self._cap_dict

    def set_bs(self, batch_size):
        self._batch_size = batch_size
