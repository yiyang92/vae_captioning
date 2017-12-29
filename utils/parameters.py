class Parameters():
    # general parameters
    latent_size = 20
    num_epochs = 20
    learning_rate = 0.0001
    batch_size = 128
    # for decoding
    temperature = 0.6
    gen_length = 20
    # beam search
    beam_search = True
    beam_size = 3
    # encoder
    rnn_layers = 1
    encoder_hidden = 191
    keep_rate = 1.0
    #highway_lc = 2
    #highway_ls = 600
    # decoder
    decoder_hidden = 191
    decoder_rnn_layers = 1
    dec_keep_rate = 0.75
    embed_size = 353
    sent_max_size = 300
    debug = False
    # use pretrained w2vec embeddings
    pre_trained_embed = True
    fine_tune_embed = True
    # restore?
    restore = False
    # technical parameters
    is_training = True
    LOG_DIR = './model_logs/'
    visualise = False
    #base_cell = tf.contrib.rnn.LSTMCel
    vocab_size = 0 # need to be set during data load
    coco_dir = "/home/luoyy16/datasets-large/mscoco/coco/"
    def parse_args(self):
        import argparse
        import os
        parser = argparse.ArgumentParser(description="Specify some parameters, all parameters also can be directly specify in "
                                         "Parameters class")
        parser.add_argument('--lr', default=self.learning_rate, help='learning rate', dest='lr')
        parser.add_argument('--embed_dim', default=self.embed_size, help='embedding size', dest='embed')
        parser.add_argument('--lst_state_dim_enc', default=self.encoder_hidden, help='encoder state size', dest='enc_hid')
        parser.add_argument('--lst_state_dim_dec', default=self.decoder_hidden, help='decoder state size', dest='dec_hid')
        parser.add_argument('--latent', default=self.latent_size, help='latent space size', dest='latent')
        parser.add_argument('--dec_dropout', default=self.dec_keep_rate, help='decoder dropout keep rate', dest='dec_drop')
        parser.add_argument('--restore', default=self.restore, help='whether restore', dest='rest')
        parser.add_argument('--gpu', help="specify GPU number")
        parser.add_argument('--coco_dir', default=self.coco_dir, help="mscoco directory")
        parser.add_argument('--epochs', default=self.num_epochs, help="number of training epochs")

        args = parser.parse_args()
        self.learning_rate = args.lr
        self.embed_size = args.embed
        self.encoder_hidden = args.enc_hid
        self.decoder_hidden = args.dec_hid
        self.latent_size = args.latent
        self.dec_keep_rate = args.dec_drop
        self.restore = True if args.rest == 1 else False
        self.coco_dir = args.coco_dir
        self.num_epochs = int(args.epochs)
        # CUDA settings
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
