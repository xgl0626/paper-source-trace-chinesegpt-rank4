import utils
import settings
import os
from os.path import join
import json
rong1="scibert_eval1-434/test_submission_scibert.json"
rong2="scibert_eval1-429/test_submission_scibert.json"
rong3="scibert_eval1-423/test_submission_scibert.json"

data_dir="/home/best/xgl0626/paper-source-trace-main/out/kddcup/"
papers = utils.load_json(data_dir,rong1)
papers2 = utils.load_json(data_dir,rong2)
papers3 = utils.load_json(data_dir,rong3)

sub_dict = {}
for item in papers:
    score_list=[]
    for score1,score2,score3 in zip(papers[item],papers2[item],papers3[item]):
        score_list.append((score1+score2+score3)/3)
    sub_dict[item]=score_list
print(sub_dict)
utils.dump_json(sub_dict, join(data_dir, 'scibert_rong'), "test_submission_scibert_rong.json") 