import pickle
import os
import numpy as np
from  args import Args
import pandas as pd
from collections import Counter


#第一类 0.32576510

file_path = '/data/none404/few/pingan/keras_model/runs_4/1593174233/'
args = Args()
all_pred = []

fold=str(4)
save_dict = args.mapping_dict
mapping_dict = dict([(j, i) for i, j in save_dict.items()])

file_list = os.listdir(file_path)
for path in file_list:
    # if 'pickle' in path and fold in path:
    if 'pickle' in path :
        with open(file_path+path,'rb') as f:
            print(file_path+path)
            all_pred.append(pickle.load(f))


####概率平均
# pred_label_list = np.average(all_pred, axis=0)
# pred_label_list = np.argmax(pred_label_list,axis=1)
#
# pred_merge_list = [mapping_dict[i] for i in pred_label_list]



#####大类投票
my_best_res = pd.read_csv("/data/none404/few/pingan/keras_model/runs_4/1593174233/result/ensemble_result.csv",sep='\t',header=None)
last_res = pd.read_csv("/data/none404/few/pingan/keras_model/runs_4/1593174233/result/model_0.4626_0.4107-30682_1_1predict.csv",sep='\t',header=None)
best_res = my_best_res[1].apply(lambda x:save_dict[x]).tolist()
last_res = last_res[1].apply(lambda x:save_dict[x]).tolist()

sub_res =[]
c = 1
vote_list = list(save_dict.values())[:8] #前两类投票
for i in range(len(all_pred[0])):
    ans_list =[]
    ans_list.extend([best_res[i]] * 2)  # 最好模型两份权重,且放在前面防止平票
    # ans_list.append(last_res[i])
    for j  in range(len(all_pred)):
        logits = all_pred[j][i]
        pred= np.argmax(logits,axis=-1)
        ans_list.append(pred)

    vote_res = Counter(ans_list).most_common(1)[0][0]
    if vote_res!=best_res[i]:
        # if vote_res  in vote_list:  # 大类变成其他类
        if best_res[i]  in vote_list:  # 最好成绩之前是大类，#把其改成小类
        # if 1:
            c += 1
            # 用投票结果
            print('vote_res_{},best_res_{}'.format(vote_res,best_res[i]))
            print('投票情况为:{}'.format(ans_list))
        else:
            vote_res = best_res[i]
    sub_res.append(vote_res)

pred_merge_list = [mapping_dict[i] for i in sub_res]
change_list = [j for i,j in enumerate(best_res) if j!=sub_res[i]]
print('有{}个样本被修改'.format(len(change_list)))
# my_best_res = pd.read_csv("/data/none404/few/pingan/keras_model/runs_4/1593174233/result/ensemble_result.csv",sep='\t')
#tou

save_path = file_path+'result'
test_df = pd.read_csv(args.processed_data + 'new_test.csv')
predict_df = pd.DataFrame()
predict_df['id'] = test_df['id']
predict_df[''] = pred_merge_list
if not os.path.exists(save_path):
    os.mkdir(save_path)
# with open(save_p + '/20_logits.pickle', 'wb') as f:
#     pickle.dump(pred_merge_logits, f)
print(predict_df.shape)
# predict_df.to_csv(save_path + '/{}_{}_predict.csv'.format(fold,file_path.split('/')[-2]), sep='\t',
#                   index=False, header=False)
predict_df.to_csv(save_path + '/ensemble{}_predict.csv'.format(fold,file_path.split('/')[-2]), sep='\t',
                  index=False, header=False)