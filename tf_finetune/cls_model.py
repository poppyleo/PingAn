import tensorflow as tf
from tf_finetune.fintune_args import Args
args = Args()
from tf_utils.bert_modeling import BertModel, BertConfig, get_assignment_map_from_checkpoint, get_shape_list  # BERT源码结构
from tensorflow.contrib.layers.python.layers import initializers


class Model:

    def __init__(self, args):
        self.args = args
        self.input_x_word = tf.placeholder(tf.int32, [None, None], name="input_x_word")
        self.input_x_len = tf.placeholder(tf.int32, name='input_x_len')
        self.input_mask = tf.placeholder(tf.int32, [None, None], name='input_mask')
        self.intent_label = tf.placeholder(tf.int32, [None], name='intent_label')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')
        self.is_training = tf.placeholder(tf.bool, None, name='is_training')
        self.initializer = initializers.xavier_initializer()

        self.merge_num = 56  # 分类类别个数

        sequence_output = self.bert_embed()
        if self.args.add_lstm:
            print('Bilstm')
            gru_inputs = tf.nn.dropout(sequence_output, args.dropout)
            # 前向
            GRU_cell_fw = tf.contrib.rnn.LSTMCell(args.lstm_dim)  # 参数可调试
            # 后向
            GRU_cell_bw = tf.contrib.rnn.LSTMCell(args.lstm_dim)  # 参数可调试

            output_layer_1 = tf.nn.bidirectional_dynamic_rnn(cell_fw=GRU_cell_fw,
                                                             cell_bw=GRU_cell_bw,
                                                             inputs=gru_inputs,
                                                             sequence_length=None,
                                                             dtype=tf.float32)[0]
            # 拼接双向
            output_layer_1 = tf.concat([output_layer_1[0], output_layer_1[1]], axis=-1)
            model_outputs = output_layer_1
            pool_size = args.sequence_length
            hidden_size = model_outputs.shape[-1]
        else:  # bert的输出
            model_outputs = sequence_output
            pool_size = self.args.sequence_length
            hidden_size = get_shape_list(sequence_output)[-1]
        # 池化
        if self.args.meanpool:
            print('MeanPool:', self.args.is_avg_pool)
            print(self.args)
            output_layer = model_outputs
            avpooled_out = tf.layers.average_pooling1d(output_layer, pool_size=pool_size,
                                                       strides=1)  # shape = [batch, hidden_size]
            print(avpooled_out.shape)
            avpooled_out = tf.reshape(avpooled_out, [-1, hidden_size])
        else:
            print('CLS:', True)
            avpooled_out = sequence_output[:, 0:1, :]  # pooled_output
            avpooled_out = tf.squeeze(avpooled_out, axis=1)

        def logits_and_predict(avpooled_out, num_classes, name_scope=None):
            with tf.name_scope(name_scope):
                inputs = tf.nn.dropout(avpooled_out, keep_prob=self.keep_prob)
                logits = tf.layers.dense(inputs, num_classes,
                                         kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                         name=name_scope + '_logits')
                predict = tf.round(tf.sigmoid(logits), name=name_scope + "predict")

            return logits, predict

        # 56分类
        self.merge_logits, self.merge_predict = logits_and_predict(avpooled_out, self.merge_num,
                                                                   name_scope='merge_relation')
        merge_one_hot_labels = tf.one_hot(self.intent_label, depth=self.merge_num, dtype=tf.float32)
        merge_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=merge_one_hot_labels, logits=self.merge_logits)
        merge_loss = tf.reduce_mean(tf.reduce_sum(merge_losses, axis=1))
        self.loss = merge_loss

    def bert_embed(self, bert_init=True):
        """加载bert模型结构"""
        bert_config_file = self.args.bert_config_file
        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertModel(
            config=bert_config,
            is_training=self.is_training,  # 微调
            input_ids=self.input_x_word,
            input_mask=self.input_mask,
            token_type_ids=None,
            use_one_hot_embeddings=False)

        # 加载预训练模型
        tvars = tf.trainable_variables()
        init_checkpoint = self.args.bert_file
        assignment_map, initialized_variable_names = get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if bert_init:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info("**** Trainable Variables ****")
        print('加载初始权重: {}'.format(init_checkpoint))
        # 返回sequence_output
        return model.sequence_output

