# Use Variational Auto-Encoder to generate capions
## Overview
 Tensorflow Implementation of [Diverse and Accurate Image Description Using a Variational Auto-Encoder with an Additive Gaussian Encoding Space, (Nips)] (https://papers.nips.cc/paper/7158-diverse-and-accurate-image-description-using-a-variational-auto-encoder-with-an-additive-gaussian-encoding-space.pdf)


## Usage

Training:
Just launch the training script:
```shell=
python main.py --gpu 'your gpu'
```
### Parameters
Parameters can be set directly in in utils/parameters.py file.
(or specify through command line parameters).

### Generation
For list of required parameters:
```shell=
python gen_caption.py -h
```
For example:
```
python -i gen_caption.py --img_path ./images/COCO_val2014_000000233527.jpg --checkpoint ./checkpoints/gaussian_nocv.ckpt --params_path ./pickles/params_Normal_False_gaussian_nocv_False
```
Where:
- --params_path: saved Parameters class, can be saved by calling main.py --save_params
- --checkpoint: saved checkpoint
- --img_path: path for image
- -i: for launching python in interactive mode so captions can be generated by calling generator.generate_caption(img_path). This can be also used in ipython notebook

### Implementation progress
- LSTM baseline (implemented)
- CVAE baseline (implemented)
- cluster vectors (partially impemented, vectors for test set generated using
  tensorflow object detection API and faster-RCNN)
- beam search (implemented, but needs some corrrections)
- AG-CVAE (partially implemented)
- GMM-CVAE (in progress)
- Caption generation for new photos (partially implemented, will need to automate cluster vectors generation process)

### Specific requirements
- zhusuan - probabilistic framework https://github.com/thu-ml/zhusuan/
- tensorflow >= 1.4.1

### Other files
- prepare_cluster_vectors_train_val.ipynb - takes MSCOCO dataset json files and generates cluster vectors
- prepare_test_vectors.ipynb - gets test set cluster vector file, prepared using tf.models API and generates cluster vector