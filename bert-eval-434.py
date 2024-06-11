import os
from os.path import join
from tqdm import tqdm
from collections import defaultdict as dd
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup,BertTokenizer,BertConfig,BertModel
from transformers.optimization import AdamW
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tqdm import trange
from sklearn.metrics import classification_report, precision_recall_fscore_support, average_precision_score
import logging
import random
import utils
import settings

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)
MAX_SEQ_LENGTH=512
gpu_id='cuda:1'
                
def prepare_bert_input():
    x_train = []
    y_train = []
    x_valid = []
    y_valid = []

    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_train7.json")
    #papers_dblp = utils.load_json(data_dir, "paper_info_hit_from_dblp.json")
    n_papers = len(papers)
    #papers = sorted(papers, key=lambda x: x["_id"])
    #n_train = int(n_papers * 2 / 3)
    n_train = n_papers
    # n_valid = n_papers - n_train

    papers_train = papers
    papers_valid = papers[n_train:]
    
    
    pids_train = [p["_id"] for p in papers_train]
    pids_valid = [p["_id"] for p in papers_valid]
    
    #print(pids_train)
    #print(pids_valid)
    
    with open(join(data_dir, "bib_train1.txt"), "w", encoding="utf-8") as f:
        for line in pids_train:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_valid1.txt"), "w", encoding="utf-8") as f:
        for line in pids_valid:
            f.write(line + "\n")
                
    in_dir = join(data_dir, "paper-xml")
    files = []
    for f in os.listdir(in_dir):
        if f.endswith(".xml"):
            files.append(f)

    pid_to_source_titles = dd(list)
    pid_to_title = {}
    for paper in tqdm(papers):
        pid = paper["_id"]
        pid_to_title[pid]=paper["title"]
        
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    # files = sorted(files)
    # for file in tqdm(files):
    # np.random.seed(seed)
    for cur_pid in tqdm(pids_train):
        # cur_pid = file.split(".")[0]
        # if cur_pid not in pids_train and cur_pid not in pids_valid:
            # continue
        f = open(join(in_dir, cur_pid + ".xml"), encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        #print(bs)
        
        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        all_bid_to_title = {}
        n_refs = 0

        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            if ref.analytic.title is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            all_bid_to_title[bid] = ref.monogr.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx
        
        flag = False

        cur_pos_bib = set()

        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    cur_pos_bib.add(bid)
        
        cur_neg_bib = set(bid_to_title.keys()) - cur_pos_bib
        if not flag:
            continue
    
        if len(cur_pos_bib) == 0 or len(cur_neg_bib) == 0:
            continue
        #print(bib_to_contexts)
        bib_to_contexts = utils.find_bib_context(xml)
        print(len(references),len(bib_to_contexts))
        n_pos = len(cur_pos_bib)
        
        n_neg = min(len(cur_neg_bib),n_pos * 10)
        #n_neg = n_pos * 10
        np.random.seed(seed)
        cur_neg_bib_list = list(cur_neg_bib)
        #if len(cur_neg_bib) >=n_neg:
        cur_neg_bib_sample = np.random.choice(cur_neg_bib_list, n_neg, replace=False)
        #else:
        #cur_neg_bib_sample = np.random.choice(cur_neg_bib_list, n_neg, replace=True)
        
        #print(cur_neg_bib_sample)
        #if cur_pid in pids_train:
        #    cur_x = x_train
        #    cur_y = y_train
        #elif cur_pid in pids_valid:
        #    cur_x = x_valid
        #    cur_y = y_valid
        #else:
        #    continue
            # raise Exception("cur_pid not in train/valid/test")
        #bid_to_title[bib]+
        
        #abstract = bs.find_all("profileDesc")[0].abstract.text.lower()
        #print(abstract[0].abstract.text.lower())
        
        for bib in cur_pos_bib:
            #print(papers_dblp[cur_pid]['authors'])
            #if cur_pid in papers_dblp.keys():
            #    cit_str = "The author of this paper is {}".format(",".join(papers_dblp[cur_pid]['authors']))
            #else:
            #    cit_str = ""
            #print(cit_str)
            #+bid_to_title[bib]
            
            cur_context = bid_to_title[bib]+" ".join(bib_to_contexts[bib])
            x_train.append(cur_context)
            y_train.append(1)
    
        for bib in cur_neg_bib_sample:
            #if cur_pid in papers_dblp.keys():
            #    cit_str = "The author of this paper is {}".format(papers_dblp[cur_pid]['authors'])
            #else:
            #    cit_str = ""
            #+bid_to_title[bib]
            cur_context = bid_to_title[bib]+" ".join(bib_to_contexts[bib])
            x_train.append(cur_context)
            y_train.append(0)
    
    print("len(x_train)", len(x_train), "len(x_valid)", len(x_valid))


    with open(join(data_dir, "bib_context_train1.txt"), "w", encoding="utf-8") as f:
        for line in x_train:
            f.write(line + "\n")
    
    with open(join(data_dir, "bib_context_train_label1.txt"), "w", encoding="utf-8") as f:
        for line in y_train:
            f.write(str(line) + "\n")
    


class BertInputItem(object):
    """An item with all the necessary attributes for finetuning BERT."""

    def __init__(self, text, input_ids, input_mask, segment_ids, label_id):
        self.text = text
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def convert_examples_to_inputs(example_texts, example_labels, max_seq_length, tokenizer, verbose=0):
    """Loads a data file into a list of `InputBatch`s."""
    
    input_items = []
    examples = zip(example_texts, example_labels)
    for (ex_index, (text, label)) in enumerate(examples):

        # Create a list of token ids
        input_ids = tokenizer.encode(f"[CLS] {text} [SEP]")
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]

        # All our tokens are in the first input segment (id 0).
        segment_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label

        input_items.append(
            BertInputItem(text=text,
                          input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
        
    return input_items


def get_data_loader(features, max_seq_length, batch_size, shuffle=True): 

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

    dataloader = DataLoader(data, shuffle=shuffle, batch_size=batch_size)
    return dataloader


def evaluate(model, dataloader, device, criterion):
    model.eval()
    
    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        with torch.no_grad():
            r = model(input_ids, attention_mask=input_mask,
                                          token_type_ids=segment_ids, labels=label_ids)
            # tmp_eval_loss = r[0]
            logits = r[1]
            # print("logits", logits)
            tmp_eval_loss = criterion(logits, label_ids)

        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        
        predicted_labels += list(outputs)
        correct_labels += list(label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    
    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
        
    return eval_loss, correct_labels, predicted_labels
import json
import time
import torch.nn as nn       
def train(year=2023, model_name="scibert"):
    train_texts = []
    dev_texts = []
    train_labels = []
    dev_labels = []
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    print("data_year_dir", data_year_dir)

    with open(join(data_year_dir, "bib_context_train1-439.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_texts.append(line.strip())

    with open(join(data_year_dir, "bib_context_train_label1-439.txt"), "r", encoding="utf-8") as f:
        for line in f:
            train_labels.append(int(line.strip()))

    print("Train size:", len(train_texts))
    print("Dev size:", len(dev_texts))

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")

    class_weight = len(train_labels) / (2 * np.bincount(train_labels))
    class_weight = torch.Tensor(class_weight).to(device)
    print("Class weight:", class_weight)

    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "./scibert_scivocab_uncased/" #"./scibert_scivocab_uncased/"
    else:
        raise NotImplementedError
    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL) 
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    #model_para=torch.load('./out/kddcup/pretrain-450/pytorch_model.bin')
    #model.load_state_dict(model_para, strict=False) 
    model.to(device)
     

    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    train_features = convert_examples_to_inputs(train_texts, train_labels, MAX_SEQ_LENGTH, tokenizer, verbose=0)
    BATCH_SIZE = 16 #32
    train_dataloader = get_data_loader(train_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=True)

    GRADIENT_ACCUMULATION_STEPS = 1
    NUM_TRAIN_EPOCHS = 10
    LEARNING_RATE = 1e-5
    WARMUP_PROPORTION = 0.1
    MAX_GRAD_NORM = 10 #5 429 #10 434 #10 random2021 423

    num_train_steps = int(len(train_dataloader.dataset) / BATCH_SIZE / GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)
    num_warmup_steps = int(WARMUP_PROPORTION * num_train_steps)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    
    OUTPUT_DIR = join(settings.OUT_DIR, "kddcup", model_name+'_eval1-434')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file = open('./out/kddcup/scibert_eval1-434/'+str(time.time())+'.txt','w')
    MODEL_FILE_NAME = "pytorch_model.bin"
    PATIENCE = 3
                    
    loss_history = []
    no_improvement = 0
    for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Training iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            outputs = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)
            logits = outputs[1]
            loss = criterion(logits, label_ids)

            if GRADIENT_ACCUMULATION_STEPS > 1:
                loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            loss.backward()
            print("loss",loss.item())    
            tr_loss += loss.item()
                            
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)  
                
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                
        dev_loss = eval_test_papers_bert(model)
        model.train()
        log_file.write(str(dev_loss)+'\n')
        print("Loss history:", loss_history)
        print("Dev loss:", dev_loss)
        print("Dev loss:", dev_loss)        
        if len(loss_history) == 0 or dev_loss > max(loss_history):
            no_improvement = 0
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
            torch.save(model_to_save.state_dict(), output_model_file)
        else:
            no_improvement += 1
        
        if no_improvement >= PATIENCE: 
            print("No improvement on development set. Finish training.")
            break   
        loss_history.append(dev_loss)
    log_file.close()

import torch.nn.functional as F
def eval_test_papers_bert(model):
    data_year_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers_test = utils.load_json(data_year_dir, "paper_source_trace_dev7.json")
   # papers_dblp = utils.load_json(data_year_dir, "paper_info_hit_from_dblp.json")
    pids_test = {p["_id"] for p in papers_test}

    in_dir = join(settings.DATA_TRACE_DIR, "PST/paper-xml")
    files = []
    for f in os.listdir(in_dir):
        cur_pid = f.split(".")[0]
        if f.endswith(".xml") and cur_pid in pids_test:
            files.append(f)

    truths = papers_test
    pid_to_source_titles = dd(list)
    pid_to_title={}
    for paper in tqdm(truths):
        pid = paper["_id"]
        pid_to_title[pid] = paper["title"]
        for ref in paper["refs_trace"]:
            pid_to_source_titles[pid].append(ref["title"].lower())

    BERT_MODEL = "./scibert_scivocab_uncased/"
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    print("device", device)
    model.eval()

    BATCH_SIZE = 16
    metrics = []
    f_idx = 0

    xml_dir = join(settings.DATA_TRACE_DIR, "PST/paper-xml")

    for paper in tqdm(papers_test):
        cur_pid = paper["_id"]
        
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')

        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()

        source_titles = pid_to_source_titles[cur_pid]
        if len(source_titles) == 0:
            continue

        references = bs.find_all("biblStruct")
        bid_to_title = {}
        all_bid_to_title = {}
        n_refs = 0
        for ref in references:       
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            if ref.analytic.title is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            all_bid_to_title[bid] = ref.monogr.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        # dev source < reference
        bib_to_contexts = utils.find_bib_context(xml) 
        # 
        bib_sorted = sorted(bib_to_contexts.keys())
        print(len(bib_sorted))
        print(len(bid_to_title.keys()))
        for bib in bib_sorted:
            cur_bib_idx = int(bib[1:])
            if cur_bib_idx + 1 > n_refs:
                n_refs = cur_bib_idx + 1
        
        y_true = [0] * n_refs
        y_score = [0] * n_refs

        flag = False
        # pos choose ref.analytic.title
        for bid in bid_to_title:
            cur_ref_title = bid_to_title[bid]
            for label_title in source_titles:
                if fuzz.ratio(cur_ref_title, label_title) >= 80:
                    flag = True
                    b_idx = int(bid[1:])
                    y_true[b_idx] = 1
        
        if not flag:
            continue
        #print(cur_pid)
        #if cur_pid in papers_dblp.keys():
        #    cit_str = "The author of this paper is {}".format(",".join(papers_dblp[cur_pid]['authors']))
        #else:
        #    cit_str = ""
        
        #+bid_to_title[bib]
        #abstract = bs.find_all("profileDesc")[0].abstract.text.lower()
        contexts_sorted = [bid_to_title[bib]+" ".join(bib_to_contexts[bib]) for bib in bib_sorted]

        test_features = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)

                logits = outputs[1]
                   
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        
        try:
            for ii in range(len(predicted_scores)):
                bib_idx = int(bib_sorted[ii][1:])
                # print("bib_idx", bib_idx)
                ####
                y_score[bib_idx] = float(utils.sigmoid(predicted_scores[ii]))
        except IndexError as e:
            metrics.append(0)
            continue
        
        cur_map = average_precision_score(y_true, y_score)
        metrics.append(cur_map)
        f_idx += 1
        if f_idx % 20 == 0:
            print("map until now", np.mean(metrics), len(metrics), cur_map)

    print("bert average map", np.mean(metrics), len(metrics))
    return np.mean(metrics)
    
def gen_kddcup_valid_submission_bert(model_name="scibert"):
    print("model name", model_name)
    data_dir = join(settings.DATA_TRACE_DIR, "PST")
    papers = utils.load_json(data_dir, "paper_source_trace_test_wo_ans.json")
    #papers_dblp = utils.load_json(data_dir, "paper_info_hit_from_dblp.json")
    if model_name == "bert":
        BERT_MODEL = "bert-base-uncased"
    elif model_name == "scibert":
        BERT_MODEL = "./scibert_scivocab_uncased/"
    else:
        raise NotImplementedError
    
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    sub_example_dict = utils.load_json(data_dir, "submission_example_test.json")

    device = torch.device(gpu_id if torch.cuda.is_available() else "cpu")
    print("device", device)
     
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    model.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name+'_eval1-434', "pytorch_model.bin")))
    model.to(device)
    model.eval()
    
    #model1 = BertForSequenceClassification.from_pretrained(BERT_MODEL, num_labels = 2)
    #model1.load_state_dict(torch.load(join(settings.OUT_DIR, "kddcup", model_name+'_eval1', "pytorch_model.bin")))
    #model1.to(device)
    #model1.eval()
    
    BATCH_SIZE = 16
    # metrics = []
    # f_idx = 0

    xml_dir = join(data_dir, "paper-xml")
    sub_dict = {}

    for paper in tqdm(papers):
        cur_pid = paper["_id"]
        file = join(xml_dir, cur_pid + ".xml")
        f = open(file, encoding='utf-8')
        xml = f.read()
        bs = BeautifulSoup(xml, "xml")
        f.close()
        #print(bs)
        references = bs.find_all("biblStruct")
        bid_to_title = {}
        all_bid_to_title = {}
        n_refs = 0
        for ref in references:
            if "xml:id" not in ref.attrs:
                continue
            bid = ref.attrs["xml:id"]
            if ref.analytic is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            if ref.analytic.title is None:
                if ref.monogr is not None:
                    #print(ref.monogr.idno,ref.monogr.title.text.lower())
                    all_bid_to_title[bid] = ""
                    bid_to_title[bid]=""
                continue
            bid_to_title[bid] = ref.analytic.title.text.lower()
            all_bid_to_title[bid] = ref.monogr.title.text.lower()
            b_idx = int(bid[1:]) + 1
            if b_idx > n_refs:
                n_refs = b_idx

        bib_to_contexts = utils.find_bib_context(xml)
        # choose last ref.analytic.title
        bib_sorted = ["b" + str(ii) for ii in range(n_refs)]
        
        y_score = [0] * n_refs
        print(len(sub_example_dict[cur_pid]),n_refs)
        assert len(sub_example_dict[cur_pid]) == n_refs
        # continue
        #if cur_pid in papers_dblp.keys():
        #    cit_str = "The author of this paper is {}".format(",".join(papers_dblp[cur_pid]['authors']))
        #else:
        #    cit_str = ""
        #bid_to_title[bib]+
        #abstract = bs.find_all("profileDesc")[0].abstract.text.lower() 
        contexts_sorted = [bid_to_title[bib]+" ".join(bib_to_contexts[bib]) for bib in bib_sorted]

        test_features = convert_examples_to_inputs(contexts_sorted, y_score, MAX_SEQ_LENGTH, tokenizer)
        test_dataloader = get_data_loader(test_features, MAX_SEQ_LENGTH, BATCH_SIZE, shuffle=False)

        predicted_scores = []
        for step, batch in enumerate(test_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)
                #outputs1 = model1(input_ids, attention_mask=input_mask,token_type_ids=segment_ids, labels=label_ids)
                                           
                #tmp_eval_loss = outputs[0]
                #logits = (outputs[1]+outputs1[1])/2
                logits = outputs[1]

                
            cur_pred_scores = logits[:, 1].to('cpu').numpy()
            predicted_scores.extend(cur_pred_scores)
        
        for ii in range(len(predicted_scores)):
            bib_idx = int(bib_sorted[ii][1:])
            # print("bib_idx", bib_idx)
            y_score[bib_idx] = float(utils.sigmoid(predicted_scores[ii]))
        
        sub_dict[cur_pid] = y_score
    
    utils.dump_json(sub_dict, join(settings.OUT_DIR, "kddcup", model_name+'_eval1-434'), "test_submission_scibert.json")

def setup_seed(seed):
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.	
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



if __name__ == "__main__":
    seed=2023
    setup_seed(seed)
    #prepare_bert_input()
    train(model_name="scibert")
    gen_kddcup_valid_submission_bert(model_name="scibert")
