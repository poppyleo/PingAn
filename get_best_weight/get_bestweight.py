import sys
sys.path.append("/data/none404/few/pingan/")
from config import Config
import tensorflow as tf
import os
import json
import numpy as np
from bert import tokenization
import tqdm
from tf_finetune.data_gen import DataIterator

from sklearn.metrics import f1_score
from functools import partial
import scipy as sp
import pickle

result_data_dir = Config().result
gpu_id = 7
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
print('GPU ID: ', str(gpu_id))
print('Data dir: ', result_data_dir)
print('Pretrained Model Vocab: ', Config().vocab_file)
print('Model: ', Config().checkpoint_path)
"""

"""

class OptimizedF1(object):
    def __init__(self):
        self.coef_ = []

    def _kappa_loss(self, coef, X, y):
        """
        y_hat = argmax(coef*X, axis=-1)
        :param coef: (1D array) weights
        :param X: (2D array)logits
        :param y: (1D array) label
        :return: -f1
        """
        X_p = np.copy(X)
        X_p = coef * X_p


        ll = f1_score(y, np.argmax(X_p, axis=-1),average='macro')

        return 1 / ll

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        # 猜测初始权值

        initial_coef = [0.93176036, 0.82036651, 1.48035201, 0.77121235, 1.02010976, 1.03041518
        , 1.10471499, 0.83454336, 0.78324563, 1.00284514, 1.01849879, 1.04373809
        , 0.72977377, 0.97638191, 0.86863568, 1.12892304, 0.8685432, 1.20208818
        , 1.13569018, 1.13989816, 1.01456766, 1.27047542, 0.92338993, 1.0193458
        , 1.21675139, 0.81817477, 1.28745098, 0.7656725, 1.28732234, 1.22365476
        , 1.17265336, 1.03693549, 1.17868814, 0.79514414, 0.75799953, 0.68072757
        , 0.72126019, 1.15774987, 1.04624453, 1.07189072, 0.9093836, 0.50383846
        , 0.9368068, 1.29145974, 1.14709545, 0.85055542, 0.96357464, 1.26848634
        , 1.13203998, 0.75278892, 1.04386991, 1.03858905, 0.79846986, 0.87853501
        , 0.98847069, 1.33329965]
        initial_coef=[1]*56
        print(initial_coef)

        # 最小化一个或多个变量的标量函数使新的F1最小的权值
        self.coef_ = sp.optimize.basinhopping(loss_partial, initial_coef, niter=150000,
                                              # callback=print_fun,
                                              stepsize=5e-10)

    def predict(self, X, y):
        X_p = np.copy(X)
        X_p = self.coef_['x'] * X_p

        return f1_score(y, np.argmax(X_p, axis=-1),average='macro')

    def coefficients(self):
        return self.coef_['x']


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
    row_max=row_max.reshape(-1, 1)
    x = x - row_max
    #计算e的指数次幂
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
    for input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, seq_length in tqdm.tqdm(test_iter):

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
    assert len(true_label_list)==len(pred_merge_list)
    pickle_path = config.pickle_path

    """写入至pickle文件"""
    with open(pickle_path+'pred_label.pickle','wb') as f:
        pickle.dump(pred_merge_list,f)
    with open(pickle_path+'pred_logits.pickle','wb') as f:
        pickle.dump(pred_merge_logits,f)
    with open(pickle_path+'true_label.pickle','wb') as f:
        pickle.dump(true_label_list,f)




if __name__ == '__main__':
    config = Config()
    vocab_file= config.vocab_file
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    print('Predicting dev.txt..........')
    dev_iter = DataIterator(config.batch_size, data_file=config.processed_data+ 'new_dev.csv',config =config
                            ,use_bert=config.use_bert,seq_length=config.sequence_length, is_test=True, tokenizer=tokenizer)
    # set_test(dev_iter, config.checkpoint_path)
    """
    读取pickle文件
    """
    pickle_path = config.pickle_path
    with open(pickle_path+'pred_label.pickle','rb') as f:
        pred_merge_list= pickle.load(f)
    with open(pickle_path+'pred_logits.pickle','rb') as f:
        pred_merge_logits = pickle.load(f)
    with open(pickle_path+'true_label.pickle','rb') as f:
        true_label_list = pickle.load(f)

    print('改变权重前的F1为：', f1_score(true_label_list, pred_merge_list,average='macro'))
    op = OptimizedF1()

    op.fit(np.array(pred_merge_logits), true_label_list)
    print('改变权重后的F1为：', op.predict(np.array(pred_merge_logits), true_label_list))
    print(list(op.coefficients()))