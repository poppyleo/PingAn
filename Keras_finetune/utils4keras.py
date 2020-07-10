from Keras_finetune.args4keras import Args
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.utils.np_utils import to_categorical
from bert import tokenization

args = Args()

vocab_file = args.vocab_file
do_lower_case = True
tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)


class mydata_generator(DataGenerator):
    """数据生成器"""
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text_a = text.split(';')[0]  # 机
            text_b = text.split(';')[-1]  # 人
            try:
                last_len = text_b.split(' 人 ')[-2].split(' ').__len__()
            except:
                print('SPECIL text')
                last_len = 0
            text_a = text_a.split(' ')
            text_b = text_b.split(' ')
            a_tokens = []
            b_tokens = []
            ntokens = []
            segment_ids = []
            """得到input的token-----start-------"""
            ntokens.append("[CLS]")
            segment_ids.append(0)
            """text_a"""
            # 得到问题的token
            for i, word in enumerate(text_a):
                token = tokenizer.tokenize(word)
                a_tokens.extend(token)
            # 把text_a的token加入至所有字的token中
            for i, token in enumerate(a_tokens):
                ntokens.append(token)
                segment_ids.append(0)
            if args.addsep:
                ntokens.append("[SEP]")
                segment_ids.append(1)
            # 得到text_a的token
            for i, word in enumerate(text_b):
                token = tokenizer.tokenize(word)
                b_tokens.extend(token)
            # 把text_b的token加入至所有字的token中
            for i, token in enumerate(b_tokens):
                ntokens.append(token)
                segment_ids.append(1)
            # 长于MAX LEN 则截断
            if ntokens.__len__() >= args.sequence_length - 1:
                ntokens = ntokens[:(args.sequence_length - 1)]
                segment_ids = segment_ids[:(args.sequence_length - 1)]

            ntokens.append("[SEP]")
            segment_ids.append(1)

            """得到input的token-------end--------"""
            """token2id---start---"""
            input_ids = tokenizer.convert_tokens_to_ids(ntokens)
            if args.mask_weight:
                input_mask = [1] * (len(input_ids) - last_len - 2) + [2] * last_len + [1] * 2
            else:
                input_mask = [1] * len(input_ids)
            while len(input_ids) < args.sequence_length:
                # 不足时补零
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                # we don't concerned about it!
                ntokens.append("**NULL**")
            token_ids, segment_ids = input_ids, segment_ids
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label=to_categorical(label,args.num_class).tolist() #onehot花
            batch_labels.append(label)
            if len(batch_token_ids) == self.batch_size or is_end:
                """padding"""
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)

                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []

# focal loss with multi label
def focal_loss(classes_num, gamma=2., alpha=.25, e=0.1):
    # classes_num contains sample number of each classes
    def focal_loss_fixed(target_tensor, prediction_tensor):
        '''
        prediction_tensor is the output tensor with shape [None, 100], where 100 is the number of classes
        target_tensor is the label tensor, same shape as predcition_tensor
        '''
        import tensorflow as tf
        from tensorflow.python.ops import array_ops
        from keras import backend as K

        #1# get focal loss with no balanced weight which presented in paper function (4)
        zeros = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)
        one_minus_p = array_ops.where(tf.greater(target_tensor,zeros), target_tensor - prediction_tensor, zeros)
        FT = -1 * (one_minus_p ** gamma) * tf.log(tf.clip_by_value(prediction_tensor, 1e-8, 1.0))

        #2# get balanced weight alpha
        classes_weight = array_ops.zeros_like(prediction_tensor, dtype=prediction_tensor.dtype)

        total_num = float(sum(classes_num))
        classes_w_t1 = [ total_num / ff for ff in classes_num ]
        sum_ = sum(classes_w_t1)
        classes_w_t2 = [ ff/sum_ for ff in classes_w_t1 ]   #scale
        classes_w_tensor = tf.convert_to_tensor(classes_w_t2, dtype=prediction_tensor.dtype)
        classes_weight += classes_w_tensor

        alpha = array_ops.where(tf.greater(target_tensor, zeros), classes_weight, zeros)

        #3# get balanced focal loss
        balanced_fl = alpha * FT
        balanced_fl = tf.reduce_mean(balanced_fl)

        #4# add other op to prevent overfit
        # reference : https://spaces.ac.cn/archives/4493
        nb_classes = len(classes_num)
        fianal_loss = (1-e) * balanced_fl + e * K.categorical_crossentropy(K.ones_like(prediction_tensor)/nb_classes, prediction_tensor)

        return fianal_loss
    return focal_loss_fixed