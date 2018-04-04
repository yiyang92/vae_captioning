class Parameters():
    # general parameters
    latent_size = 150
    num_clusters = 90  # 80 in mscoco, + 10 as mscoco objeect ids from 0-90
    num_epochs = 20
    learning_rate = 0.0005
    num_captions = 5  # every iteration use how many captions for one image (1-5)
    batch_size = 32  # proved to be quite good
    cnn_feature_size = 4096  # vgg16 fc2 output shape
    # for decoding
    temperature = 1.0
    # if greedy, choose word with the highest prob;
    # if sample, sample from multinullli distribution
    # sample_gen = 'greedy' # 'greedy', 'sample', 'beam_search'
    sample_gen = 'beam_search'  # set to greedy if you want faster see results
    # beam search
    # will take very long time, but was set according to papers baselines
    beam_size = 10
    # encoder
    encoder_rnn_layers = 1
    encoder_hidden = 512
    # decoder
    std = 0.1  # decoder test time N(0, std)
    decoder_hidden = 512
    decoder_rnn_layers = 1
    dec_keep_rate = 1.0  # decoder caption input dropout
    embed_size = 256
    gen_max_len = 30
    gen_z_samples = 100  # according to paper (Diverse cap)
    ann_param = 5  # KL-divergence component weight in objective function
    dec_lstm_drop = 1.0
    optimizer = 'Adam'  # SGD, Adam, Momentum
    lstm_clip_by_norm = 5.0
    # restore?
    restore = False
    # technical parameters
    LOG_DIR = './model_logs/'
    save_params = 0
    no_encoder = False
    vocab_size = None  # need to be set during data load
    coco_dir = "/home/luoyy16/datasets-large/mscoco/coco/"
    # fine-tuning
    hdf5_file = coco_dir + "train_val.hdf5"  # file contains val+train images
    use_hdf5 = True
    fine_tune = False
    fine_tune_top = True  # fine-tune CNN top layer
    fine_tune_fe = True  # fine-tune bottom layers
    cnn_lr = 0.00001
    cnn_optimizer = 'Adam'  # SGD, Adam, Momentum'
    cnn_dropout = 0.5  # cnn dropout keep_rate
    weight_decay = 0.00004  # L2-regularization for CNN parameters ||wtw||
    # inference
    gen_name = "00"  # names will be like val_<gen_name>.json
    checkpoint = "last_run"
    num_epochs_per_decay = 5
    use_c_v = False
    # preprocessing
    gen_val_captions = 4000  # set -1 to generate captions on a original dataset
    keep_words = 3  # minimal time of word occurence
    cap_max_length = 100  # maximum length of caption, more will be clipped
    prior = 'Normal'  # Normal, GMM, AG. Priors for CVAE model
    max_checkpoints_to_keep = 5
    mode = 'training'  # training or inference
    num_ex_per_epoch = 150000  # 586363 for im2txt, number examples per epoch
    image_net_weights_path = './utils/vgg16_weights.npz'
    logging = False

    def parse_args(self):
        import argparse
        import os
        parser = argparse.ArgumentParser(description="Specify training parameters, "
                                         "all parameters also can be "
                                         "directly specify in the "
                                         "Parameters class")
        parser.add_argument('--lr', default=self.learning_rate,
                            help='learning rate', dest='lr')
        parser.add_argument('--embed_dim', default=self.embed_size,
                            help='embedding size', dest='embed')
        parser.add_argument('--enc_hid', default=self.encoder_hidden,
                            help='encoder state size', dest='enc_hid')
        parser.add_argument('--dec_hid', default=self.decoder_hidden,
                            help='decoder state size', dest='dec_hid')
        parser.add_argument('--latent', default=self.latent_size,
                            help='latent space size', dest='latent')
        parser.add_argument('--restore', help='whether restore',
                            action="store_true")
        parser.add_argument('--gpu', help="specify GPU number")
        parser.add_argument('--coco_dir', default=self.coco_dir,
                            help="mscoco directory")
        parser.add_argument('--epochs', default=self.num_epochs,
                            help="number of training epochs")
        parser.add_argument('--bs', default=self.batch_size,
                            help="Batch size")
        parser.add_argument('--no_encoder',
                            help="use this if want to run baseline lstm",
                            action="store_true")
        parser.add_argument('--temperature', default=self.temperature,
                            help="set temperature parameter for generation")
        parser.add_argument('--gen_name', default=self.gen_name,
                            help="prefix of generated json nam")
        parser.add_argument('--dec_drop', default=self.dec_keep_rate,
                            help="decoder caption dropout")
        parser.add_argument('--gen_z_samples', default=self.gen_z_samples,
                            help="#z samples")
        parser.add_argument('--ann_param', default=self.ann_param,
                            help="annealing speed, more slower")
        parser.add_argument('--dec_lstm_drop', default=self.dec_lstm_drop,
                            help="decoder lstm dropout")
        parser.add_argument('--sample_gen', default=self.sample_gen,
                            help="'greedy', 'sample', 'beam_search'")
        parser.add_argument('--checkpoint', default=self.checkpoint,
                            help="specify checkpoint name, default=last_run")
        parser.add_argument('--optimizer', default=self.optimizer,
                            choices=['SGD', 'Adam', 'Momentum'],
                            help="SGD or Adam")
        parser.add_argument('--c_v', default=False,
                            help="Whether to use cluster vectors",
                            action="store_true")
        parser.add_argument('--std', default=self.std,
                            help="z~N(0, std), during the test time")
        parser.add_argument('--save_params',
                            help="save params class into pickle",
                            action="store_true")
        parser.add_argument('--prior', default=self.prior,
                            choices=['GMM', 'AG', 'Normal'],
                            help="set prior (GMM, AG, Normal)")
        parser.add_argument('--fine_tune',
                            help="fine_tune",
                            action="store_true")
        parser.add_argument('--mode', default=self.mode,
                            choices=['training', 'inference'],
                            help="specify training or inference")

        args = parser.parse_args()
        self.learning_rate = float(args.lr)
        self.embed_size = int(args.embed)
        self.encoder_hidden = int(args.enc_hid)
        self.decoder_hidden = int(args.dec_hid)
        self.latent_size = int(args.latent)
        self.restore = args.restore
        self.coco_dir = args.coco_dir
        self.num_epochs = int(args.epochs)
        self.no_encoder = args.no_encoder
        self.temperature = float(args.temperature)
        self.gen_name = args.gen_name
        self.dec_keep_rate = float(args.dec_drop)
        self.gen_z_samples = int(args.gen_z_samples)
        self.ann_param = float(args.ann_param)
        self.dec_lstm_drop = float(args.dec_lstm_drop)
        self.sample_gen = args.sample_gen
        self.checkpoint = args.checkpoint
        self.optimizer = args.optimizer
        self.use_c_v = args.c_v
        self.batch_size = int(args.bs)
        self.std = float(args.std)
        self.save_params = args.save_params
        self.prior = args.prior
        self.fine_tune = args.fine_tune
        self.mode = args.mode
        # update according to user entered parameter
        self.hdf5_file = self.coco_dir + self.hdf5_file.split('/')[-1]
        # CUDA settings
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
