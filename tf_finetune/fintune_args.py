class Args:

    def __init__(self):

        """文件路径"""
        self.data = '/data/none404/few/pingan/data/'  # 原始数据路径
        self.processed_data = '/data/none404/few/pingan/processed/'  # 下游任务处理后的路径
        self.pickle_path = '/data/none404/few/pingan/get_weight/'  # 保存的概率文件 #路径
        self.save_model = '/data/none404/few/pingan/save_model/'  # 保存模型文件
        self.result = '/data/none404/few/pingan/result/'  # 结果文件

        """预训练模型"""

        # roberta
        self.bert_file = "/data/none404/few/pingan/word_corpus/model/model.ckpt-1200000"
        self.bert_config_file = "/data/none404/few/pingan/word_corpus/model/bert_config.json"
        self.vocab_file = "/data/none404/few/pingan/char_corpus/char_vocab.txt"

        # 12层15w步 Roberta
        # self.bert_file = "/data/none404/few/pingan/char_corpus/model.ckpt-600000"
        # self.bert_config_file = "/data/none404/few/pingan/char_corpus/model/bert_config.json"
        # self.vocab_file = "/data/none404/few/pingan/char_corpus/char_vocab.txt"

        #调节参数
        self.warmup_proportion = 0.05
        self.use_bert = True
        self.keep_prob = 0.9  # drop_out保留率
        self.train_epoch = 100  # 训练epoch轮次
        self.sequence_length = 256  # sequence_len
        self.learning_rate = 1e-4 * 5  # 下游学习率
        self.bert_learning_rate = 3e-5 * 5  # bert参数学习率
        self.batch_size = 88  # 12层48 6层88
        self.test_batch_size = 148  # 测试集batch_size
        self.decay_rate = 0.5
        self.decay_step = int(36000 / self.batch_size)
        self.embed_trainable = True
        self.as_encoder = True
        self.continue_training = False
        self.num_class = 56

        self.add_lstm = True  # 下接结构是否加bilistm
        self.lstm_dim = 256
        self.meanpool = False #Ture sequence_output+平均池化 ,False:CLS
        self.dropout = 0.8
        self.gru_num = 256
        self.early_stop = 20 #early_stop步数
        self.savemodel_num = 10 #高分时删掉前面模型
        self.addback = False  # 加意图后一句
        self.addsep = False  # 加sep分隔符
        self.mask_weight = False  # 加mask_weight(给最后一句更高权重)
        self.addcontext = False  # 是否加入上下文

        self.cls_num = [11274, 9736, 7766, 2658, 2142, 1166, 965, 776, 397, 393, 264, 252, 252, 242, 208, 126, 124, 109,
                        100, 98, 93, 84, 81, 64, 60, 59, 54, 54, 49, 44, 35, 35, 31, 28, 20, 19, 18, 16, 13, 11, 11, 10,
                        10, 10, 10, 9, 9, 6, 5, 5, 3, 3, 3, 3, 3, 2]  # for focal_loss

        # 分类映射字典
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
