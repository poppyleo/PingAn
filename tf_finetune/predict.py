from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
import pandas as pd
from tf_finetune.data_gen import DataIterator

result_data_dir = Config().result
gpu_id =1
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


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
    #
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
            saver.restore(session, checkpoint_path) #根据图加载权重

            _input_x = graph.get_operation_by_name("input_x_word").outputs[0]
            _input_x_len = graph.get_operation_by_name("input_x_len").outputs[0]
            _input_mask = graph.get_operation_by_name("input_mask").outputs[0]
            _keep_ratio = graph.get_operation_by_name('dropout_keep_prob').outputs[0]
            _is_training = graph.get_operation_by_name('is_training').outputs[0]
            merge_logits = graph.get_tensor_by_name('merge_relation/merge_relation_logits/BiasAdd:0')

            # merge_logits = graph.get_operation_by_name('merge_relation/Sigmoid').outputs[0]

            def run_predict(feed_dict):
                return session.run([merge_logits], feed_dict)

    print('recover from: {}'.format(checkpoint_path))
    return run_predict, (_input_x, _input_x_len, _input_mask, _keep_ratio, _is_training)


def set_test(test_iter, model_file):
    if not test_iter.is_test:
        test_iter.is_test = True
    pred_merge_logits = []
    pred_merge_list = []
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

    dym_weight = [1]*56  #根据验证集找到权重

    new_logits = np.array(pred_merge_logits) * np.array(dym_weight)
    pred_merge_list = np.argmax(new_logits, axis=1)

    save_dict = config.mapping_dict
    mapping_dict = dict([(j, i) for i, j in save_dict.items()])
    pred_merge_list = [mapping_dict[i] for i in pred_merge_list]

    test_df = pd.read_csv(config.processed_data + 'new_test.csv')
    predict_df = pd.DataFrame()
    predict_df['id'] = test_df['id']
    predict_df[''] = pred_merge_list
    model_name = config.checkpoint_path.split('/')[-1]
    print(model_name)
    save_p = result_data_dir + model_name
    if not os.path.exists(save_p):
        os.mkdir(save_p)
    model_name = config.checkpoint_path.split('/')[-1]
    print(predict_df.shape)
    predict_df.to_csv(save_p + '/{}_{}_{}predict.csv'.format(model_name, dym_weight[0], dym_weight[1]), sep='\t',
                      index=False, header=False)


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting test.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data + 'new_test.csv', config=config,
                            use_bert=config.use_bert, seq_length=config.sequence_length, is_test=True,
                            tokenizer=tokenizer)

    set_test(dev_iter, config.checkpoint_path)
