import os
from glob import glob
import numpy as np
import cv2
import json

from utils.captions import Captions

# batch generator
class Batch_Generator():
    def __init__(self, train_dir, train_cap_json=None,
                 captions=None, batch_size=None,
                 im_shape=(299, 299), feature_dict=None,
                 get_image_ids=False, get_test_ids=False):
        """
        Args:
            train_dir: coco training images directory path
            train_cap_json: json with coco annotations
            captions : instance of Captions() class
            batch_size: batch size, can be changed later
            im_shape: desirable image shapes
            feature_dict: if given, use feauture vectors instead of images in generator
            get_image_ids: whether or not return image_id list (used for test/val)
        """
        self._batch_size = batch_size
        # TODO:  glob can be replaced with os.listdir and can be randomized
        self._iterable = glob(train_dir + '*.jpg')
        if not batch_size:
            print("use all data")
            self._batch_size = len(self._iterable)
        if len(self._iterable) == 0:
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
        # image shape, for preprocessing
        self.im_shape = im_shape
        self.feature_dict = feature_dict
        self.get_image_ids = get_image_ids

    def next_batch(self, get_image_ids = False):
        self.get_image_ids = get_image_ids
        imn_batch  = [None] * self._batch_size
        for i, item in enumerate(self._iterable):
            inx = i % self._batch_size
            imn_batch[inx] = item
            if inx == self._batch_size - 1:
                if self.feature_dict:
                    images = [self.feature_dict[imn.split('/')[-1]] for imn in imn_batch]
                    # NOTE: dont forget to squueze dimensions aquiring val/test
                    # or retrain feature vectors
                    images = np.squeeze(np.array(images), 1)
                else:
                    images = self._get_images(imn_batch)
                # concatenate to obtain [images, caption_indices, lengths]
                inp_captions, l_captions, lengths = self._form_captions_batch(imn_batch)
                if self.get_image_ids:
                    image_ids = []
                    for fn in imn_batch:
                        id_ = self.cap_instance.filename_to_imid[fn.split('/')[-1]]
                        image_ids.append(id_)
                    yield images, (inp_captions, l_captions), lengths, image_ids
                else:
                    yield images, (inp_captions, l_captions), lengths
                imn_batch = [None] * self._batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            if self.feature_dict:
                images = [self.feature_dict[imn.split('/')[-1]] for imn in imn_batch]
                images = np.squeeze(np.array(images), 1)
            else:
                images = self._get_images(imn_batch)
            inp_captions, l_captions, lengths = self._form_captions_batch(imn_batch)
            if self.get_image_ids:
                image_ids = []
                for fn in imn_batch:
                    id_ = self.cap_instance.filename_to_imid[fn.split('/')[-1]]
                    image_ids.append(id_)
                yield images, (inp_captions, l_captions), lengths, image_ids
            else:
                yield images, (inp_captions, l_captions), lengths

    def _test_images_to_imid(self):
        with open(self._train_cap_json) as rf:
            try:
                j = json.loads(rf.read())
            except FileNotFoundError as e:
                raise
        return {img['file_name']:img['id'] for img in j['images']}

    def next_train_batch(self):
        imn_batch  = [None] * self._batch_size
        for i, item in enumerate(self._iterable):
            inx = i % self._batch_size
            imn_batch[inx] = item
            if inx == self._batch_size - 1:
                if self.feature_dict:
                    images = [self.feature_dict[imn.split('/')[-1]] for imn in imn_batch]
                    images = np.squeeze(np.array(images), 1)
                else:
                    images = self._get_images(imn_batch)
                image_ids = []
                for fn in imn_batch:
                    id_ = self._fn_to_id[fn.split('/')[-1]]
                    image_ids.append(id_)
                yield images, image_ids
                imn_batch = [None] * self._batch_size
        if imn_batch[0]:
            imn_batch = [item for item in imn_batch if item]
            if self.feature_dict:
                images = [self.feature_dict[imn.split('/')[-1]] for imn in imn_batch]
                images = np.squeeze(np.array(images), 1)
            else:
                images = self._get_images(imn_batch)
            image_ids = []
            for fn in imn_batch:
                id_ = self._fn_to_id[fn.split('/')[-1]]
                image_ids.append(id_)
            yield images, image_ids

    def _get_images(self, names):
        images = []
        for name in names:
            # image preprocessing
            image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
            image = self._preprocess_image(image)
            images.append(image)
        return np.stack(images)

    def _preprocess_image(self, image):
        """
        Args:
            image: numpy array contained image
            size: tuple, desired shape
        """
        # first crop the image and resize it
        # TODO: normalization
        crop = min(image.shape[0], image.shape[1])
        h_start = image.shape[0] // 2 - crop // 2
        w_start = image.shape[1] // 2 - crop // 2
        image = image[h_start: h_start + crop, w_start: w_start + crop] / 255 - 0.5
        image = cv2.resize(image, self.im_shape)
        return image

    def _form_captions_batch(self, imn_batch):
        """
        Args:
            imn_batch: image file names in the batch
        Returns :
            list of np arrays [[batch_size, caption], [lengths]], where lengths have
            batch_size shape
        """
        # use image_names to get caption, add padding, put it into numpy array
        # calculate length of every sequence and make a list
        # randomly choose caption for the current iteration
        # use static array for efficiency
        labels_captions_list = [None] * len(imn_batch)
        input_captions_list = [None] * len(imn_batch)
        lengths = np.zeros(len(imn_batch))
        idx = 0
        for fn in imn_batch:
            # TODO: improve error handling when file is not correct
            fn = fn.split('/')[-1]
            caption = self.captions[fn][np.random.randint(len(self.captions[fn]))]
            # split into labels/inputs (encoder/decoder inputs)
            input_captions_list[idx] = caption[:-1] # <BOS>...
            labels_captions_list[idx] = caption[1:] # ...<EOS>
            lengths[idx] = len(input_captions_list[idx])
            idx += 1
        # add padding and put captions into np array of shape [batch_size, max_batch_seq_len]
        pad = len(max(input_captions_list, key=len))
        input_captions_list = np.array([cap + [0] * (pad - len(cap)) for cap in input_captions_list])
        labels_captions_list = np.array([cap + [0] * (pad - len(cap)) for cap in labels_captions_list])
        return input_captions_list, labels_captions_list, lengths

    @property
    def cap_dict(self):
        return self._cap_dict

    def set_bs(self, batch_size):
        self._batch_size = batch_size
