import numpy as np
import h5py
import pickle
import cv2
from glob import glob
import argparse

def prepare_captions(params):
    ...
def main(params):
    # specify images directory and h5 output directory
    COCO_DIR = params['coco_dir']
    h5_file = params['output_h5']
    # list valdation and train directories
    train_dir = COCO_DIR + "/images/train2014/"
    valid_dir = COCO_DIR + "/images/val2014/"
    tr_files = list(glob(train_dir + '*.jpg'))
    val_files = list(glob(valid_dir + '*jpg'))
    # conctenate train + val for more convenient futher usage
    # I dont include test as inference file load speed is ok with current processing
    imgs = tr_files + val_files
    if len(imgs) == 0:
        raise ValueError
    # create output h5 file
    N = len(imgs)
    f = h5py.File(h5_file, "w")
    dset = f.create_dataset("images", (N, 224, 224, 3), dtype='uint8') # space for resized images
    imtoi = {}
    for i, image_path in enumerate(imgs):
        # load the image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # handle grayscale input images
        if len(img.shape) == 2:
            img = img[:, :, np.newaxis]
            img = np.concatenate((img, img, img), axis=2)
        # yes, I know maybe better use indices, but I dont want to change current
        # processing too much
        imname = image_path.split('/')[-1]
        imtoi[imname] = i
        # write to h5
        dset[i] = img
        if i % 1000 == 0:
          print('processing %d/%d (%.2f%% done)' % (i, N, i*100.0/N))
    # sorry :(
    with open('./pickles/itoi.pickle', 'wb') as wf:
        pickle.dump(obj=imtoi, file=wf)
        print("Saved hdf5 imname to indices pickle")
    f.close()
    print("wrote ", h5_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_h5', default='train_val.h5',
                      help='output h5 file')
    parser.add_argument('--coco_dir', help='MSCOCO directory')
    parser.add_argument('--val_examples', default=4000,
                      help='change dataset default train/val split')
    # options
    parser.add_argument('--max_length', default=16, type=int,
                        help='max length of a caption, in number of words')
    parser.add_argument('--word_count_threshold',
                        default=5, type=int,
                        help='only words that occur more than this number of '
                        'times will be put in vocab')
    parser.add_argument('--num_test', default=0, type=int,
                        help='number of test images (to withold until very very end)')
    count_thr = params['word_count_threshold']
    args = parser.parse_args()
    params = vars(args)
    main(params)
