import tensorflow as tf
import numpy as np
import argparse
import os
import pickle

from utils.parameters import Parameters
from vae_model.decoder import Decoder
from utils.captions import Dictionary

# debuging
from tensorflow.python.tools import inspect_checkpoint as chkp

base_model = tf.contrib.keras.applications.VGG16(weights='imagenet',
                                                 include_top=True)
model = tf.contrib.keras.models.Model(inputs=base_model.input,
                                      outputs=base_model.get_layer('fc2').output)

class Generator():
    """Generate caption, given the image
    """
    def __init__(self,
                 checkpoint_path,
                 params_path, vocab_path, gen_method='greedy'):
        self.checkpoint_path = checkpoint_path
        self.params = self._load_params(params_path)
        self.gen_method = gen_method
        # load vocabulary
        try:
            os.path.exists(vocab_path)
        except:
            raise ValueError("No caption vocabulary path specified, "
                       "Usually it can be found in the ./pickles foulder "
                       "after model training")
        with open(vocab_path, 'rb') as rf:
            data_dict = pickle.load(rf)
        self.data_dict = Dictionary(data_dict)
        self.params.vocab_size =self.data_dict.vocab_size

    def _c_v_generator(self, image):
        # TODO: finish cluster vector implementation
        return None

    def _load_params(self, params_path):
        """Load serialized Parameters class, for convenience
        """
        with open(params_path, 'rb') as rf:
            params = pickle.load(rf)
        return params

    def _get_features(self, img_path):
        """Loads image, extract features using keras pretrained model
        Args:
            img_path: path to the image
        Returns:
            features: vector of shape [1, 4096]
        """
        im_shape = (224, 224) # VGG16
        img = tf.contrib.keras.preprocessing.image.load_img(img_path,
                                                            target_size=im_shape)
        x = tf.contrib.keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = tf.contrib.keras.applications.vgg16.preprocess_input(x)
        features = model.predict(x)
        return features, img

    def generate_caption(self, img_path, beam_size=2):
        """Caption generator
        Args:
            image_path: path to the image
        Returns:
            caption: caption, generated for a given image
        """
        # TODO: to avoid specify model again use frozen graph
        g = tf.Graph()
        # change some Parameters
        self.params.sample_gen = self.gen_method
        with g.as_default():
            # specify rnn_placeholders
            ann_lengths_ps = tf.placeholder(tf.int32, [None])
            images_ps = tf.placeholder(tf.float32, [None, 4096])
            captions_ps = tf.placeholder(tf.int32, [None, None])
            try:
                os.path.exists(img_path)
            except:
                raise ValueError("Image not found")

            # image fesatures [b_size + f_size(4096)] -> [b_size + embed_size]
            images_fv = tf.layers.dense(images_ps, self.params.embed_size,
                                        name='imf_emb')
            # will use methods from Decoder class
            decoder = Decoder(images_fv, captions_ps,
                              ann_lengths_ps, self.params, self.data_dict)
            # if use cluster vectors
            if self.params.use_c_v:
                # cluster vectors from "Diverse and Accurate Image Description.." paper.
                # 80 is number of classes, for now hardcoded
                # for GMM-CVAE must be specified
                c_i = tf.placeholder(tf.float32, [None, 90])
                c_i_emb = tf.layers.dense(c_i, self.params.embed_size,
                                          name='cv_emb')
                # map cluster vectors into embedding space
                decoder.c_i = c_i_emb
                decoder.c_i_ph = c_i
            # image_id
            im_id = [img_path.split('/')[-1]]
            with tf.variable_scope("decoder"):
                _, _, shpe, states = decoder.px_z_fi({})
        feature_vector, image = self._get_features(img_path)
        with g.as_default():
            # saver = tf.train.import_meta_graph(checkpoint_path + '.meta',
            #                                    clear_devices=True)
            saver = tf.train.Saver(tf.trainable_variables())
        with tf.Session(graph=g) as sess:
            saver.restore(sess, self.checkpoint_path)
            # chkp.print_tensors_in_checkpoint_file(args.checkpoint,
            #                                       tensor_name='', all_tensors=False,
            #                                       all_tensor_names=True)
            if self.params.use_c_v:
                c_v = self._c_v_generator(image)
            else:
                c_v = None
            if self.gen_method == 'beam_search':
                sent = decoder.beam_search(sess, im_id, feature_vector,
                                           images_ps, c_v,
                                           beam_size=beam_size)
            elif self.gen_method == 'greedy':
                sent, _ = decoder.online_inference(sess, im_id, feature_vector,
                                                   images_ps, c_v=c_v)
            return sent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Specify generation parameters")
    parser.add_argument('--img_path', help="Path to the image")
    parser.add_argument('--checkpoint', help="Model checkpoint path")
    parser.add_argument('--vocab_path', default='./pickles/capt_vocab.pickle',
                        help="Indices to words dictionary")
    parser.add_argument('--gpu', default='',
                        help="Specify GPU number if use GPU")
    parser.add_argument('--c_v_generator', default=None,
                        help="If use cluster vectors, specify tensorflow api model"
                        "For more information look README")
    parser.add_argument('--gen_method', default='greedy',
                        help='greedy, beam_search or sample')
    parser.add_argument('--params_path', default=None,
                        help="specify params pickle file")
    parser.add_argument('--beam_size', default=2,
                        help="If using beam_search, specify beam_size")
    args = parser.parse_args()
    # CUDA settings
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    # parameter of the model
    params = Parameters()
    generator = Generator(checkpoint_path=args.checkpoint,
                          params_path=args.params_path,
                          vocab_path=args.vocab_path,
                          gen_method=args.gen_method)
    caption = generator.generate_caption(args.img_path, args.beam_size)
    print(caption[0]['caption'])
