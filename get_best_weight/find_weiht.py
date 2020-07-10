import sys

sys.path.append("/home/none404/hm/PingAn")
from config import Config
import tensorflow as tf
import os
import numpy as np
from bert import tokenization
import tqdm
from tf_finetune.data_gen import DataIterator

from sklearn.metrics import f1_score, precision_score, recall_score
import pickle

result_data_dir = Config().result
gpu_id = 7
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)

class_num = 56
weights = [1.0] * class_num


def search_weight(valid_y, raw_prob, init_weight=[1.0] * class_num, step=0.001):
    """

    :param valid_y:真实类别list如[1,0,0,1,0,0,0,0]
    :param raw_prob: 预测的类别logits
    :param init_weight: 初始权重
    :param step: 随机游走的步长
    :return:找到的权重
    """
    tolarent = 0 #容忍度
    weight = init_weight.copy()
    f_best = f1_score(y_true=valid_y, y_pred=raw_prob.argmax(axis=1), average='macro')
    print('初始F1为：',f_best)
    flag_score = 0
    round_num = 1
    while (flag_score != f_best or tolarent>10):
        print("round: ", round_num)
        round_num += 1
        flag_score = f_best
        for c in range(class_num):
            for n_w in range(0, 2000, 10):
                num = n_w * step
                new_weight = weight.copy()
                new_weight[c] = num

                prob_df = raw_prob.copy()
                prob_df = prob_df * np.array(new_weight)

                f = f1_score(y_true=valid_y, y_pred=prob_df.argmax(
                    axis=1), average='macro')
                if f > f_best:
                    weight = new_weight.copy()
                    f_best = f
                    print(f)
                else:
                    tolarent+=1
    return weight


"""

"""


def softmax(x, axis=1):
    """
    自写函数定义softmax
    :param x:
    :param axis:
    :return:
    """
    # 计算每行的最大值
    row_max = x.max(axis=axis)

    # 每行元素都需要减去对应的最大值，否则求exp(x)会溢出，导致inf情况
    row_max = row_max.reshape(-1, 1)
    x = x - row_max
    # 计算e的指数次幂
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=axis, keepdims=True)
    s = x_exp / x_sum
    return s


def get_session(checkpoint_path):
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)
        with session.as_default():
            # Load the saved meta graph and restore variables
            try:
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_path))
            except OSError:
                saver = tf.train.import_meta_graph("{}.ckpt.meta".format(checkpoint_path))
            saver.restore(session, checkpoint_path)

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]

            merge_logits = graph.get_tensor_by_name('merge_relation/merge_relation_logits/BiasAdd:0')

            def run_predict(feed_dict):
                return session.run([merge_logits], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True
    pred_merge_logits = []
    pred_merge_list = []
    true_label_list = []
    predict_fun, feed_keys = get_session(model_file)
    for input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, seq_length in tqdm.tqdm(
            test_iter):
        merge_logits = predict_fun(
            dict(
                zip(feed_keys, (input_ids_list, seq_length, input_mask_list, 1, False))
            )
        )[0]
        merge_label = softmax(merge_logits)
        pred_merge_logits.extend(merge_logits)
        merge_label = np.argmax(merge_label, axis=1)
        pred_merge_list.extend(merge_label)
        true_label_list.extend(all_label_list)
    assert len(true_label_list) == len(pred_merge_list)
    pickle_path = '/home/none404/hm/PingAn/data/get_weight/'

    """写入至pickle文件"""
    with open(pickle_path + 'pred_label.pickle', 'wb') as f:
        pickle.dump(pred_merge_list, f)
    with open(pickle_path + 'pred_logits.pickle', 'wb') as f:
        pickle.dump(pred_merge_logits, f)
    with open(pickle_path + 'true_label.pickle', 'wb') as f:
        pickle.dump(true_label_list, f)


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting dev.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'new_dev.csv', config=config
                            , use_bert=config.use_bert, seq_length=config.sequence_length, is_test=True,
                            tokenizer=tokenizer)
    set_test(dev_iter, config.checkpoint_path)
    """
    读取pickle文件
    """
    pickle_path = config.pickle_path
    with open(pickle_path + 'pred_label.pickle', 'rb') as f:
        pred_merge_list = pickle.load(f)
    with open(pickle_path + 'pred_logits.pickle', 'rb') as f:
        pred_merge_logits = pickle.load(f)
    with open(pickle_path + 'true_label.pickle', 'rb') as f:
        true_label_list = pickle.load(f)
    pred_merge_logits = np.array(pred_merge_logits)

    new_weight = search_weight(true_label_list, pred_merge_logits)
    print(new_weight)
