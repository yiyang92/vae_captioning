# preprocess captions and cluster vectors before feeding to the network
# only for the case, when we use multiple captions at once for one image
import numpy as np
def preprocess_captions(captions_batch, cl_batch, cv):
    """
    Args:
        captions_batch: tuple (input_captions, labels)
    [batch_size, num_captions, length_padded]
        cl_batch: caption lengths [batch_size, num_captions]
        cv: cluster vectors [batch_size, num_clusters]
    Returns:
        captions_batch: tuple (input_captions, labels), shape:
    [batch_size * num_captions, length_padded]
        cl_batch: caption lengths [batch_size * num_captions]
        cv: cluster vectors [batch_size * num_captions, num_clusters] (or 0)
    """
    inputs, labels = captions_batch
    in_shape = inputs.shape
    inputs = np.reshape(inputs, (in_shape[0] * in_shape[1], in_shape[2]))
    labels = np.reshape(labels, (in_shape[0] * in_shape[1], in_shape[2]))
    cl_batch = np.ravel(cl_batch)
    if not len(cv) == 0:
        cv_tiled = np.tile(np.expand_dims(cv, axis=1), (1, in_shape[1], 1))
        cv = np.reshape(cv_tiled, (in_shape[0] * in_shape[1], cv.shape[1]))
    return (inputs, labels), cl_batch, cv
