import utils
import settings
import os
from os.path import join
import json
data_dir = join(settings.DATA_TRACE_DIR, "PST")
papers = utils.load_json(data_dir, "paper_source_trace_train_ans.json")
n_papers = len(papers)
papers = sorted(papers, key=lambda x: x["_id"])
n_train = int(n_papers * 8 / 10)   
papers_train = papers[:n_train]
papers_valid = papers[n_train:]
print(len(papers_train),len(papers_valid))
#dev_json = open('paper_source_trace_dev_ans.json','w')
utils.dump_json(papers_valid,'data/PST','paper_source_trace_dev2-10.json')
utils.dump_json(papers_train,'data/PST','paper_source_trace_train8-10.json')