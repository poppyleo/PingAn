import pandas as pd
import numpy as np
import os
import sys

sys.path.append('/home/none404/hm/Tencent_ads/Tencent_finetune/')
from config import Config
import pickle

config = Config()

#

dym_weight = [0.93176036, 0.82036651, 1.48035201, 0.77121235, 1.02010976, 1.03041518
    , 1.10471499, 0.83454336, 0.78324563, 1.00284514, 1.01849879, 1.04373809
    , 0.72977377, 0.97638191, 0.86863568, 1.12892304, 0.8685432, 1.20208818
    , 1.13569018, 1.13989816, 1.01456766, 1.27047542, 0.92338993, 1.0193458
    , 1.21675139, 0.81817477, 1.28745098, 0.7656725, 1.28732234, 1.22365476
    , 1.17265336, 1.03693549, 1.17868814, 0.79514414, 0.75799953, 0.68072757
    , 0.72126019, 1.15774987, 1.04624453, 1.07189072, 0.9093836, 0.50383846
    , 0.9368068, 1.29145974, 1.14709545, 0.85055542, 0.96357464, 1.26848634
    , 1.13203998, 0.75278892, 1.04386991, 1.03858905, 0.79846986, 0.87853501
    , 0.98847069, 1.33329965]  #0.2663192578632775 +0.004
"""

"""
result_dir = "/home/none404/hm/PingAn/data/result/model_0.4368_0.4368-18009/20_logits.pickle" #0.262
with open(result_dir, 'rb') as f:
    pred_gender_logits = pickle.load(f)

pred_gender = np.array(pred_gender_logits) * np.array(dym_weight)
pred_merge_list = np.argmax(pred_gender, axis=1)

save_dict = config.mapping_dict
mapping_dict = dict([(j, i) for i, j in save_dict.items()])
pred_merge_list = [mapping_dict[i] for i in pred_merge_list]

test_df = pd.read_csv(config.processed_data + 'new_test.csv')
predict_df = pd.DataFrame()
predict_df['id'] = test_df['id']
predict_df[''] = pred_merge_list
model_name = '/{}_{}predict'.format(dym_weight[0],dym_weight[1])
print(model_name)
save_p = '/'.join(result_dir.split('/')[:-1]) + model_name
if not os.path.exists(save_p):
    os.mkdir(save_p)
print(predict_df.shape)
predict_df.to_csv(save_p + '/{}_predict.csv'.format(model_name), sep='\t', index=False, header=False)
