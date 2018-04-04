import numpy as np
import h5py
import pickle
from glob import glob
import argparse

from utils.image_utils import load_image


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
    dset = f.create_dataset("images", (N, 224, 224, 3),
                            dtype='uint8')  # space for resized images
    imtoi = {}
    for i, image_path in enumerate(imgs):
        # load the image
        img = load_image(image_path, shape=(224, 224))
        # yes, I know maybe better use indices, but I dont want to change
        # curent processing too much
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
    args = parser.parse_args()
    params = vars(args)
    main(params)
