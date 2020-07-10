import numpy as np
from bert import tokenization
from tqdm import tqdm
from config import Config
import pandas as pd
import os

gpu_id = 11
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def load_data(data_file):
    """
    读取数据
    :param file:
    :return:
    """
    data_df = pd.read_csv(data_file)
    data_df.fillna('', inplace=True)
    lines = list(zip(list(data_df['text']), list(data_df['label']), list(data_df['flag'])))

    return lines


def create_example(lines):
    examples = []
    for (_i, line) in enumerate(lines):
        text = str(line[0])
        all_label = int(line[1])
        flag = int(line[2])
        examples.append(InputExample(text=text, all_label=all_label, flag=flag))
    return examples


def get_examples(data_file):
    return create_example(
        load_data(data_file)
    )


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, text, all_label, flag):
        self.text = text
        self.flag = flag
        self.all_label = all_label


class DataIterator:
    """
    数据迭代器
    """

    def __init__(self, batch_size, data_file, tokenizer, config, use_bert=False, seq_length=100, is_test=False, ):
        self.data_file = data_file
        self.data = get_examples(data_file)
        self.batch_size = batch_size
        self.use_bert = use_bert
        self.seq_length = seq_length
        self.num_records = len(self.data)
        self.all_tags = []
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test
        self.config = config

        if not self.is_test:
            self.shuffle()
        self.tokenizer = tokenizer
        print(self.num_records)

    def convert_single_example(self, example_idx):
        text = self.data[example_idx].text
        all_label = self.data[example_idx].all_label
        flag = self.data[example_idx].flag
        text_a, text_b = text.split(';')
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
            token = self.tokenizer.tokenize(word)
            a_tokens.extend(token)
        # 把text_a的token加入至所有字的token中
        for i, token in enumerate(a_tokens):
            ntokens.append(token)
            segment_ids.append(0)
        if self.config.addsep:
            ntokens.append("[SEP]")
            segment_ids.append(1)
        # 得到text_a的token
        for i, word in enumerate(text_b):
            token = self.tokenizer.tokenize(word)
            b_tokens.extend(token)
        # 把text_b的token加入至所有字的token中
        for i, token in enumerate(b_tokens):
            ntokens.append(token)
            segment_ids.append(1)

        # 长于MAX LEN 则截断
        if ntokens.__len__() >= self.seq_length - 1:
            ntokens = ntokens[:(self.seq_length - 1)]
            segment_ids = segment_ids[:(self.seq_length - 1)]
        ntokens.append("[SEP]")
        segment_ids.append(1)

        """得到input的token-------end--------"""

        """token2id---start---"""
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        if self.config.mask_weight:
            input_mask = [1] * (len(input_ids) - last_len - 2) + [2] * last_len + [1] * 2
        else:
            input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        while len(input_ids) < self.seq_length:
            # 不足时补零
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            ntokens.append("**NULL**")
            # label_mask.append(0)
        assert len(input_ids) == self.seq_length
        assert len(input_mask) == self.seq_length
        assert len(segment_ids) == self.seq_length
        """token2id ---end---"""
        return input_ids, input_mask, segment_ids, all_label, flag

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration

        input_ids_list = []
        input_mask_list = []
        segment_ids_list = []
        flag_list = []
        all_label_list = []
        num_tags = 0
        while num_tags < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)
            if res is None:
                self.idx += 1
                if self.idx >= self.num_records:
                    break
                continue
            input_ids, input_mask, segment_ids, all_label, flag = res

            # 一个batch的输入
            input_ids_list.append(input_ids)
            input_mask_list.append(input_mask)
            segment_ids_list.append(segment_ids)
            all_label_list.append(all_label)
            flag_list.append(flag)
            if self.use_bert:
                num_tags += 1

            self.idx += 1
            if self.idx >= self.num_records:
                break
        return input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, self.seq_length


if __name__ == '__main__':
    config = Config()
    vocab_file = config.vocab_file
    do_lower_case = True
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
    # print(vocab_file)
    # print(print(len(tokenizer.vocab)))

    # data_iter = DataIterator(config.batch_size, data_file= config.dir_with_mission + 'train.txt', use_bert=True,
    #                         seq_length=config.sequence_length, tokenizer=tokenizer)
    #
    # dev_iter = DataIterator(config.batch_size, data_file=config.dir_with_mission + 'dev.txt', use_bert=True,
    #                          seq_length=config.sequence_length, tokenizer=tokenizer, is_test=True)
    train_iter = DataIterator(config.batch_size,
                              data_file=config.processed_data + 'new_test.csv',
                              use_bert=config.use_bert, config=config,
                              tokenizer=tokenizer, seq_length=config.sequence_length)
    for input_ids_list, input_mask_list, segment_ids_list, flag_list, all_label_list, seq_length in tqdm(train_iter):
        # print(input_ids_list)
        # print(segment_ids_list[0])
        # print(input_mask_list[0])
        print(flag_list)
        # print(all_label_list)
        # break
