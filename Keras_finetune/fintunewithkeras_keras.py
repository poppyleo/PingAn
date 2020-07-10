import numpy as np
import pandas as pd
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense, LSTM, GRU, Bidirectional, Dropout
from keras.utils.np_utils import to_categorical
from Keras_finetune.args4keras import Args
from tf_utils.adversal import adversarial_training, loss_with_gradient_penalty
from tf_utils.focal_loss import focal_loss
import os
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import pickle
import time
from Keras_finetune.utils4keras import mydata_generator

# Keras_finetune fold5 add SEP
# /data/none404/few/pingan/keras_model/runs_0/1593086649/   addfgm
# /data/none404/few/pingan/keras_model/runs_0/1593071686/  ori+cls
# /data/none404/few/pingan/keras_model/runs_2/1593074020/  ori+cls+add_sep
# /data/none404/few/pingan/keras_model/runs_2/1593074020/  ori+cls+not_sep+focalloss

"""
/data/none404/few/pingan/keras_model/runs_4/1593174233/  0.2767 fold0
/data/none404/few/pingan/keras_model/runs_4/1593174233/  0.2713 fold1
/data/none404/few/pingan/keras_model/runs_4/1593174233/  0.2671 fold2
"""

args = Args()
gpu_id = 1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

timestamp = str(int(time.time()))
num_classes = args.num_class
maxlen = 256
batch_size = 88
epoch = 68

# 模型路径
config_path = args.bert_config_file
checkpoint_path = args.bert_file
dict_path = args.vocab_file

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)

# configs = {}
# if config_path is not None:
#     configs.update(json.load(open(config_path)))
# print(configs)

def load_data(filename):
    df = pd.read_csv(filename, encoding='utf_8_sig')
    df['text'].fillna('', inplace=True)
    D = list(zip(list(df['text']), list(df['label'])))
    return D


# 加载预训练模型
def build_model():
    bert = build_transformer_model(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        # model='albert',
        return_keras_model=False,
    )
    if args.cls:
        output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)  # CLS
    else:
        # sequence_output
        output = Lambda(lambda x: x[:, :], name='sequence-token')(bert.model.output)  #
        if args.gru:
            output = Bidirectional(GRU(256))(output)
            print(output)
        elif args.lstm:
            output = Bidirectional(LSTM(256))(output)
        else:
            pass
        # output =GlobalAvgPool1D()(output) #平均池化
    output = Dropout(0.1)(output)

    output = Dense(
        units=num_classes,
        activation='softmax',
        kernel_initializer=bert.initializer
    )(output)
    model = keras.models.Model(bert.model.input, output)
    # model.summary()
    return model


def f1_loss(y_true, y_pred):
    # 计算tp、tn、fp、fn
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    # percision与recall，这里的K.epsilon代表一个小正数，用来避免分母为零
    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())
    # 计算f1
    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)  # 其实就是把nan换成0
    return 1 - K.mean(f1)
# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。

def evaluate(data):
    true, pred = [], []
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        # print(model.predict(x_true))
        y_true = y_true.argmax(axis=1)
        true.extend(y_true)
        pred.extend(y_pred)
    F1 = f1_score(true, pred, average='macro')
    return F1


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_acc = -1.0

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if epoch == 0:
            self.best_val_acc = -1.0  # 每一折初始最有val_acc
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights(save_path + '/{}_weights'.format(fold))
        print(
            u'F1: %.5f, best_F1: %.5f\n' %
            (val_acc, self.best_val_acc)
        )


def sparse_categorical_crossentropy(y_true, y_pred):
    """自定义稀疏交叉熵
    这主要是因为keras自带的sparse_categorical_crossentropy不支持求二阶梯度。
    """
    y_true = K.reshape(y_true, K.shape(y_pred)[:-1])
    y_true = K.cast(y_true, 'int32')
    y_true = K.one_hot(y_true, K.shape(y_pred)[-1])
    return K.categorical_crossentropy(y_true, y_pred)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text_a = text.split('\t')[0]
            text_b = text.split('\t')[-1]
            token_ids, segment_ids = tokenizer.encode(text_a, text_b, max_length=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            label = to_categorical(label, args.num_class).tolist()
            batch_labels.append(label)
            # batch_labels = to_categorical(batch_labels,num_classes)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')
if __name__ == '__main__':
    # 加载数据集
    data_dir = args.processed_data
    test_data = load_data(args.processed_data + 'new_test.csv')

    # 模型存储地址
    save_path = args.result + 'runs_{}/'.format(gpu_id) + timestamp + '/'
    print(save_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)



    if not args.addsep:
        data_generator =mydata_generator

    test_generator = data_generator(test_data, batch_size)
    df_train = pd.read_csv(args.processed_data + 'all_train.csv', encoding='utf_8_sig')
    evaluator = Evaluator()
    gkf = StratifiedKFold(n_splits=5).split(X=df_train['text'],
                                            y=df_train['label'])
    test_preds = []
    for fold, (train_idx, valid_idx) in enumerate(gkf):
        print('**********************fold{}**************************'.format(fold))

        # 构建模型
        model = build_model()
        train_data = df_train.iloc[train_idx]
        valid_data = df_train.iloc[valid_idx]
        train = list(zip(list(train_data['text']), list(train_data['label'])))
        valid = list(zip(list(valid_data['text']), list(valid_data['label'])))
        train_generator = data_generator(train, batch_size)
        valid_generator = data_generator(valid, batch_size)
        if args.addadv:
            """添加扰动"""
            loss = loss_with_gradient_penalty
            mert = 'sparse_categorical_accuracy'
        else:
            loss = 'categorical_crossentropy'
            mert = 'categorical_accuracy'
        if args.focal_loss:
            loss=focal_loss(args.cls_num)
        model.compile(
            loss=loss,
            # optimizer=Adam(1e-5),  # 用足够小的学习率
            optimizer=AdamLR(lr=1e-4, lr_schedule={
                1000: 1,
                2000: 0.1
            }),
            metrics=[mert],
        )

        # 写好函数后，启用对抗训练只需要一行代码
        test = 'test.ckpt'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=test,
                                                         save_weights_only=True,
                                                         verbose=1)
        if args.addfgm:
            adversarial_training(model, 'Embedding-Token', 0.5)
        model.fit_generator(
            train_generator.forfit(),
            steps_per_epoch=len(train_generator),
            epochs=epoch,
            callbacks=[evaluator]
        )
        model.load_weights(save_path + '/{}_weights'.format(fold))
        i = 0
        for x_true, y_true in test_generator:
            y_pred = model.predict(x_true)
            if i == 0:
                t_pred = y_pred
                i = 1
            else:
                t_pred = np.concatenate([t_pred, y_pred])
        test_preds.append(t_pred)
        t = np.argmax(t_pred, axis=1)
        with open(save_path + '/{}_logits.pickle'.format(fold), 'wb') as f:
            pickle.dump(t_pred, f)

    sub = np.average(test_preds, axis=0)
    save_weight = np.array([1] * 56)
    print(save_weight)
    pred_logit_list = save_weight * np.array(sub)
    pred_label_list = np.argmax(pred_logit_list, axis=1)

    save_dict = args.mapping_dict
    mapping_dict = dict([(j, i) for i, j in save_dict.items()])
    pred_merge_list = [mapping_dict[i] for i in pred_label_list]

    test_df = pd.read_csv(args.processed_data + 'new_test.csv')
    predict_df = pd.DataFrame()
    predict_df['id'] = test_df['id']
    predict_df[''] = pred_merge_list
    save_p = save_path  +'result'
    if not os.path.exists(save_p):
        os.mkdir(save_p)
    # with open(save_p + '/20_logits.pickle', 'wb') as f:
    #     pickle.dump(pred_merge_logits, f)
    print(predict_df.shape)
    predict_df.to_csv(save_p + '/{}_{}predict.csv'.format(save_weight[0], save_path[:3]), sep='\t',
                      index=False, header=False)
