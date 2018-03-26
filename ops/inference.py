# inference op
import json
import os
def inference(params, decoder, val_gen, test_gen, image_f_inputs, saver, sess):
    print("Restoring from checkpoint")
    saver.restore(sess, "./checkpoints/{}.ckpt".format(
        params.checkpoint))
    # validation set
    if not params.fine_tune:
        print("Using prepared features for generation. If you want to "
              "use fine-tuned VGG16 feature extractor, need to specify "
              "--fine_tune parameter.")
    captions_gen = []
    print("Generating captions for val file")
    acc, caps = [], []
    for f_images_batch, _, _, image_ids, c_v in val_gen.next_val_batch(
        get_image_ids=True, use_obj_vectors=params.use_c_v):
        if params.use_c_v or (
            params.prior == 'GMM' or params.prior == 'AG'):
            # 0 element doesnt matter
            c_v = c_v[:, 1:]
        if params.sample_gen == 'beam_search':
            sent = decoder.beam_search(sess, image_ids, f_images_batch,
                                       image_f_inputs, c_v,
                                       beam_size=params.beam_size)
        else:
            sent, _ = decoder.online_inference(sess, image_ids,
                                               f_images_batch,
                                               image_f_inputs, c_v=c_v)
        captions_gen += sent
    print("Generated {} captions".format(len(captions_gen)))
    val_gen_file = "./val_{}.json".format(params.gen_name)
    if os.path.exists(val_gen_file):
        print("Exists {}, delete it".format(val_gen_file))
        os.remove(val_gen_file)
        print(os.listdir('.'))
    with open(val_gen_file, 'w') as wj:
        print("saving val json file into ", val_gen_file)
        json.dump(captions_gen, wj)
    # test set
    captions_gen = []
    print("Generating captions for test file")
    for f_images_batch, image_ids, c_v in test_gen.next_test_batch(
        params.use_c_v):
        if params.use_c_v:
            c_v = c_v[:, 1:]
        sent, _ = decoder.online_inference(sess, image_ids,
                                           f_images_batch,
                                           image_f_inputs, c_v=c_v)
        captions_gen += sent
    test_gen_file = "./test_{}.json".format(params.gen_name)
    if os.path.exists(test_gen_file):
        os.remove(test_gen_file)
    with open(test_gen_file, 'w') as wj:
        print("saving test json file into", test_gen_file)
        json.dump(captions_gen, wj)
