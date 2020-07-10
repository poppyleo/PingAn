import os
import time
import json
import tqdm
from tf_finetune.cls_model import *
from tf_finetune.data_gen import DataIterator
from tf_finetune.fintune_args import Args
from bert_function.optimization import create_optimizer
import numpy as np
from bert import tokenization
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

gpu_id = 0 #指定gpu
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
result_data_dir = Args().processed_data




def train(train_iter, test_iter, arg):
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        session_conf.gpu_options.allow_growth = True
        session = tf.Session(config=session_conf)

        with session.as_default():
            model = Model(args)  #

            global_step = tf.Variable(0, name='step', trainable=False)
            learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, args.decay_step,
                                                       args.decay_rate, staircase=True)

            normal_optimizer = tf.train.AdamOptimizer(learning_rate)
            all_variables = graph.get_collection('trainable_variables')
            bert_var_list = [x for x in all_variables if 'bert' in x.name]  #bert参数
            normal_var_list = [x for x in all_variables if 'bert' not in x.name] #下接结构参数
            print('微调参数个数:{},微调学习率:{}'.format(len(bert_var_list),args.bert_learning_rate))
            print('下游参数个数:{},微调学习率:{}'.format(len(normal_var_list),args.learning_rate))
            normal_op = normal_optimizer.minimize(model.loss, global_step=global_step, var_list=normal_var_list)
            num_batch = int(train_iter.num_records / args.batch_size * args.train_epoch)
            word2vec_op, embed_learning_rate, embed_step = create_optimizer(
                model.loss, args.bert_learning_rate, num_train_steps=num_batch,
                num_warmup_steps=int(num_batch * 0.05), use_tpu=False, variable_list=bert_var_list
            )
            train_op = tf.group(normal_op, word2vec_op)
            timestamp = str(int(time.time())) #根据时间戳保存模型

            out_dir = os.path.abspath(
                os.path.join(args.save_model, timestamp))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            with open(out_dir + '/' + 'config.json', 'w', encoding='utf-8') as file:
                json.dump(args.__dict__, file)
            print("Writing to {}\n".format(out_dir))
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.savemodel_num) #保存最大文件数 保存新的时候删掉旧的
            if args.continue_training:
                print('recover from: {}'.format(args.checkpoint_path))
                saver.restore(session, args.checkpoint_path)
            else:
                session.run(tf.global_variables_initializer())
            cum_step = 0
            best_f1 = 0.43
            tolerate = 0
            for i in range(args.train_epoch):
                for input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, seq_length in tqdm.tqdm(
                        train_iter):
                    feed_dict = {
                        model.input_x_word: input_ids_list,
                        model.input_mask: input_mask_list,
                        model.input_x_len: seq_length,
                        model.intent_label: all_label_list,
                        model.keep_prob: args.keep_prob,
                        model.is_training: True,
                    }
                    _, step, _, loss, lr = session.run(
                        fetches=[train_op,
                                 global_step,
                                 embed_step,
                                 model.loss,
                                 learning_rate
                                 ],
                        feed_dict=feed_dict)

                    if cum_step % 300 == 0:
                        format_str = 'step {}, loss {:.4f} lr {:.5f}'
                        print(
                            format_str.format(
                                step, loss, lr)
                        )
                    cum_step += 1

                F1, P, R = set_test(model, test_iter, session)


                print('dev set : cum_step_{},P_{},R_{}'.format(cum_step, P, R))

                if F1 > best_f1 or i==69:
                    #保存特殊epoch
                    best_f1 = F1
                    saver.save(session, os.path.join(out_dir, 'model_{:.4f}_{:.4f}'.format(P, R)), global_step=step)
                elif best_f1>0.42:
                    tolerate += 1
                if tolerate >= args.early_stop:
                    print('earlystop at _epoch{}_step'.format(step,i))
                    sys.exit()

def set_test(model, test_iter, session):
    if not test_iter.is_test:
        test_iter.is_test = True
    true_intent_list = []
    pred_intent_list = []
    for input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, seq_length in tqdm.tqdm(
            test_iter):
        feed_dict = {
            model.input_x_word: input_ids_list,
            model.input_x_len: seq_length,
            model.intent_label: all_label_list,
            model.input_mask: input_mask_list,
            model.keep_prob: 1,
            model.is_training: False,
        }
        intent_logits = session.run(
            fetches=[model.intent_logits],
            feed_dict=feed_dict
        )[0]
        intent_label = softmax(intent_logits)
        intent_label = np.argmax(intent_label, axis=1)

        true_intent_list.extend(all_label_list)
        pred_intent_list.extend(intent_label)
    assert len(true_intent_list) == len(pred_intent_list)
    P = precision_score(true_intent_list, pred_intent_list, average='macro')
    R = recall_score(true_intent_list, pred_intent_list, average='macro')
    F1 = f1_score(true_intent_list, pred_intent_list, average='macro')
    print('F1 {}, P {},R {}'.format(F1, P, R))
    return F1, P, R

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
if __name__ == '__main__':
    arg = Args()
    vocab_file = args.vocab_file  # 生成的字典
    do_lower_case = False
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    #训练集数据
    train_iter = DataIterator(args.batch_size, data_file=args.processed_data + 'new_train.csv',
                              config=args,
                              use_bert=args.use_bert,
                              tokenizer=tokenizer, seq_length=args.sequence_length)
    #测试集数据
    dev_iter = DataIterator(args.batch_size, data_file=args.processed_data + 'new_dev.csv',
                            config=args,
                            use_bert=args.use_bert, tokenizer=tokenizer,
                            seq_length=args.sequence_length, is_test=True)
    train(train_iter, dev_iter, args)


