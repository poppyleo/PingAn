class Args:

    def __init__(self):
        self.data = '/data/none404/few/pingan/'
        self.processed_data = '/data/none404/few/pingan/processed/'
        self.addcontext = False
        self.embed_dense = True
        self.embed_dense_dim = 512
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9
        self.over_sample = True
        self.num_checkpoints = 20 * 3
        self.train_epoch = 100
        self.sequence_length = 256
        self.learning_rate = 1e-4 * 5
        self.embed_learning_rate = 3e-5 * 5
        self.batch_size = 88  # 12层64 6层88
        self.test_batch_size = 148  # 测试集batch_size
        self.decay_rate = 0.5
        self.decay_step = int(36000 / self.batch_size)
        self.embed_trainable = True
        self.as_encoder = True
        self.continue_training = False
        self.num_class = 56
        # roberta
        self.bert_file = "/data/none404/few/pingan/word_corpus/model/model.ckpt-1200000"
        self.bert_config_file = "/data/none404/few/pingan/word_corpus/model/bert_config.json"
        self.vocab_file = "/data/none404/few/pingan/char_corpus/char_vocab.txt"
        #
        # 12层15w步
        # self.bert_file = "/data/none404/few/pingan/char_corpus/model.ckpt-450000"
        # self.bert_config_file = "/data/none404/few/pingan/char_corpus/model/bert_config.json"
        # self.vocab_file = "/data/none404/few/pingan/char_corpus/char_vocab.txt"
        self.checkpoint_path = '/data/none404/few/pingan/save_model/runs_5/1592707715/model_0.4270_0.3889-25116'  # 0.2857692014591577
        self.cls_num = [11274, 9736, 7766, 2658, 2142, 1166, 965, 776, 397, 393, 264, 252, 252, 242, 208, 126, 124, 109,
                        100, 98, 93, 84, 81, 64, 60, 59, 54, 54, 49, 44, 35, 35, 31, 28, 20, 19, 18, 16, 13, 11, 11, 10,
                        10, 10, 10, 9, 9, 6, 5, 5, 3, 3, 3, 3, 3, 2]
        self.cls = True
        self.gru = False
        self.lstm =False
        self.addadv = False #扰动
        self.addfgm = True #对抗
        self.lstm_dim = 256
        self.dropout = 0.9
        self.loss_name = 'normal'
        self.gru_num = 256
        self.early_stop = 100
        self.addback = False
        self.addsep = False
        self.mask_weight = False
        self.focal_loss = False
        self.pickle_path = '/data/none404/few/pingan/get_weight/'
        self.save_model = '/data/none404/few/pingan/save_model/'
        self.pretrainning_model = 'bert'
        self.result = '/data/none404/few/pingan/keras_model/'
        self.pred = '/data/none404/few/pingan/pred/'

        # 34500/64

        self.mapping_dict = {'C': 0,
                             'III': 1,
                             'V': 2,
                             'XXXVIII': 3,
                             'XXIX': 4,
                             'LV': 5,
                             'XXXIX': 6,
                             'XIV': 7,
                             'IV': 8,
                             'XIX': 9,
                             'I': 10,
                             'XI': 11,
                             'VIII': 12,
                             'XVIII': 13,
                             'XLIX': 14,
                             'XXII': 15,
                             'XXIII': 16,
                             'XXXVII': 17,
                             'VII': 18,
                             'LIV': 19,
                             'XII': 20,
                             'X': 21,
                             'XVII': 22,
                             'XXI': 23,
                             'XLIV': 24,
                             'XXIV': 25,
                             'XXXV': 26,
                             'XXXII': 27,
                             'IX': 28,
                             'L': 29,
                             'XXXI': 30,
                             'XXXVI': 31,
                             'XIII': 32,
                             'XXVII': 33,
                             'XXXIII': 34,
                             'VI': 35,
                             'XV': 36,
                             'XLI': 37,
                             'LI': 38,
                             'XLII': 39,
                             'XXVI': 40,
                             'XLV': 41,
                             'XVI': 42,
                             'LIII': 43,
                             'XX': 44,
                             'LII': 45,
                             'XL': 46,
                             'XLIII': 47,
                             'XXX': 48,
                             'XLVII': 49,
                             'XXXIV': 50,
                             'XLVI': 51,
                             'XXV': 52,
                             'XLVIII': 53,
                             'II': 54,
                             'XXVIII': 55}

        # 加上后面一句再识别意图是否更好
