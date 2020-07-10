#单独预测某个模型
import numpy as np
from bert4keras.tokenizers import Tokenizer
from Keras_finetune.args4keras import Args
import os
from Keras_finetune.fintunewithkeras_keras import build_model
from Keras_finetune.fintunewithkeras_keras import  load_data
from Keras_finetune.utils4keras import mydata_generator
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.utils.np_utils import to_categorical
import pandas as pd

args = Args()
gpu_id = 4
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

batch_size = 128

model_path = "/data/none404/few/pingan/keras_model/runs_4/1593174233/4_weights" #模型路径
args_path = args.bert_config_file
checkpoint_path = args.bert_file
dict_path = args.vocab_file
class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            text_a = text.split('\t')[0]
            text_b = text.split('\t')[-1]
            token_ids, segment_ids = tokenizer.encode(text_a, text_b, max_length=args.sequence_length)
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


    if not args.addsep:
        data_generator =mydata_generator

# 加载数据集
# data_dir = args.data_process
# train_data = load_data(args.data_process + 'processed_data_train/fold{}/train.csv'.format(args.fold))
# print(train_data[:2])
# valid_data = load_data(args.data_process + 'processed_data_train/fold{}/val.csv'.format(args.fold))
test_data = load_data(args.processed_data + 'new_test.csv')
# # 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)
#
# 加载keras模型
model = build_model()
model.load_weights(model_path)

test_generator = data_generator(test_data, batch_size)
i = 0
test_preds =[]
for x_true, y_true in test_generator:
    y_pred = model.predict(x_true)
    if i == 0:
        t_pred = y_pred
        i = 1
    else:
        t_pred = np.concatenate([t_pred, y_pred])
test_preds.append(t_pred)
t = np.argmax(t_pred, axis=1)

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
save_p = args.result+ 'result'
if not os.path.exists(save_p):
    os.mkdir(save_p)
# with open(save_p + '/20_logits.pickle', 'wb') as f:
#     pickle.dump(pred_merge_logits, f)
print(predict_df.shape)
predict_df.to_csv(save_p + '/{}predict.csv'.format(save_weight[0]), sep='\t',
                  index=False, header=False)