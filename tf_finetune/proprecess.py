import pandas as pd
from tf_finetune.fintune_args import Args
import ast
from tqdm import tqdm
import math


# flag_list = ['@','!']


def genertate_finetunedata(train_df, istest=False):
    ###构建训练集和测试集
    finetune_train = []
    test_flag = []
    id_list = train_df['id'].unique()
    c = 0
    for idx in tqdm(id_list):
        com_df = train_df[train_df['id'] == idx]
        text_machine = ''
        text_human = ''
        for j, category in enumerate(com_df['category']):
            try:
                next_flag = com_df['category'].iloc[j + 1]
            except:
                # 最后一个元素
                next_flag = 1
            if category == 1:
                text_machine += ' '.join(com_df['char'].iloc[j]) + ' 机 '
            else:
                testflag = 0  # 为人对话位置
                text_human += ' '.join(com_df['char'].iloc[j]) + ' 人 '
                if next_flag == 1:
                    testflag = 1  # 最后意图
                    """意图获得"""
                    if istest:
                        intent_label = -1
                    else:
                        intent_label = com_df['label'].iloc[j]
                    if args.addback:
                        """加上意图后的一句"""
                        try:
                            text_machine += ' '.join(com_df['char'].iloc[j + 1]) + ' 机 '  # 加在机器后面
                        except:
                            """没有回复了"""
                            pass

                    text = text_machine + ';' + text_human
                    finetune_dict = {'id': idx, 'text': text, 'label': intent_label}
                    finetune_train.append(finetune_dict)
                    if not args.addcontext:
                        # 只要当前对话or加入上文对话,加入下文第一句
                        text_human, text_machine = '', ''
                else:
                    try:
                        intent_label = com_df['label'].iloc[j]
                    except:
                        # 测试集
                        intent_label = 0
                    if intent_label != 0:
                        c += 1
                    if istest:
                        text = text_machine + ';' + text_human
                        finetune_dict = {'id': idx, 'text': text, 'label': intent_label}
                        finetune_train.append(finetune_dict)
                test_flag.append(testflag)
    print('非最后人说话又意图的样本有{}个'.format(c))  # 非最后人说话有意图的样本有763个/50652
    return pd.DataFrame(finetune_train), test_flag


def sp_forwardtext(x):
    if args.addback:
        slide = 3
    else:
        slide = 2
    text_a, text_b = x.split(';')
    # text_a = ' 机 '.join(text_a.split(' 机 ')[-slide:])
    # text_b = ' 人 '.join(text_b.split(' 人 ')[-slide:])
    i = 0
    while len(text_a.split(' ')) + len(text_b.split(' ')) > args.sequence_length - 1:
        i += 1
        text_a = ' 机 '.join(text_a.split(' 机 ')[1:])
        if i >= slide or text_a.count('机') == 1:
            break
    if len(text_a.split(' ')) + len(text_b.split(' ')) > args.sequence_length - 1:
        resnet = len(text_a.split(' ')) + len(text_b.split(' ')) - args.sequence_length + 1
        text_a = ' '.join(text_a.split(' ')[-math.ceil(resnet):])
        # text_b = ' '.join(text_b.split(' ')[-math.ceil(resnet / 2):])
    assert len(text_a.split(' ')) + len(text_b.split(' ')) <= args.sequence_length - 1
    assert text_b != ''
    return text_a + ';' + text_b


if __name__ == '__main__':
    args = Args()
    data_path = args.data
    mapping_dict = args.mapping_dict

    train_df = pd.read_excel(data_path + 'train.xlsx')
    test_df = pd.read_excel(data_path + 'public_test.xlsx')
    """字符串转成list"""
    train_df['char'] = train_df['char'].apply(lambda x: ast.literal_eval(x))
    train_df['word'] = train_df['word'].apply(lambda x: ast.literal_eval(x))
    test_df['char'] = test_df['char'].apply(lambda x: ast.literal_eval(x))
    test_df['word'] = test_df['word'].apply(lambda x: ast.literal_eval(x))
    # 测试集统一命名
    test_df.rename(columns={"catgory": "category"}, inplace=True)
    # 映射label
    train_df['label'] = train_df['label'].apply(lambda x: mapping_dict[x])

    # 产生所需下游数据
    new_train, train_flag = genertate_finetunedata(train_df, istest=False)
    new_test, test_flag = genertate_finetunedata(test_df, istest=True)

    new_train['text'] = new_train['text'].apply(lambda x: sp_forwardtext(x))
    new_test['text'] = new_test['text'].apply(lambda x: sp_forwardtext(x))

    new_train['sequence_len'] = new_train['text'].apply(lambda x: len(x.split(' ')))
    new_test['sequence_len'] = new_test['text'].apply(lambda x: len(x.split(' ')))
    print('训练集')
    print(new_train['sequence_len'].describe())
    print(new_train['label'].value_counts().tolist())

    print('测试集')
    print(new_test['sequence_len'].describe())
    new_train['flag'] = 1
    new_test['flag'] = test_flag  # 是否是最有一个意图
    # 切分训练集，分成训练集和验证集
    print('Train Set Size:', new_train.shape)

    # Train Set Size: (40018, 4)
    new_dev_df = new_train[32000:]
    frames = [new_train[000:24000], new_train[24000:32000]]
    new_train_df = pd.concat(frames)  # 训练集
    print(new_train_df['label'].value_counts().tolist())

    new_train_df.to_csv(args.processed_data + 'new_train.csv', index=False)
    new_train.to_csv(args.processed_data + 'all_train.csv', index=False)
    new_dev_df.to_csv(args.processed_data + 'new_dev.csv', index=False)
    new_test.to_csv(args.processed_data + 'new_test.csv', index=False)
    print('Test Set Size:', new_test.shape)
    # Test Set Size: (2990, 4)
"""
训练集
count    40018.000000
mean        65.338298
std         49.903504
min          3.000000
25%         10.000000
50%         68.000000
75%         95.000000
max        371.000000
Name: sequence_len, dtype: float64
测试集
count    2990.000000
mean       96.643478
std        55.678671
min         3.000000
25%        62.000000
50%        92.000000
75%       124.750000
max       438.000000
Name: sequence_len, dtype: float64
"""

"""
训练集
count    40018.000000
mean       166.541656
std        177.835709
min          3.000000
25%         10.000000
50%        104.000000
75%        262.000000
max       1455.000000
Name: sequence_len, dtype: float64
测试集
count    2990.000000
mean      215.465886
std       147.794702
min         3.000000
25%        98.000000
50%       168.500000
75%       311.750000
max       950.000000
Name: sequence_len, dtype: float64
Train Set Size: (40018, 5)
Test Set Size: (3233, 5) #

"""
