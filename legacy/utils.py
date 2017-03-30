# -*- coding: utf-8 -*-
"""
Created on Sun Sep 25 23:16:57 2016

@author: hhj
"""

#from pycorenlp import StanfordCoreNLP
import pickle as pkl
import numpy as np
import os
import pickle
import json
from collections import Counter
import string
import re
import argparse
import sys
import random
import logging
import codecs
from nltk.tokenize import word_tokenize

# from corenlp_pywrap import pywrap 

class log:
    
    @staticmethod
    def init(filename):
        # set up logging to file - see previous section for more details
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s %(levelname)-8s %(message)s',
                            datefmt='%m-%d %H:%M',
                            filename=filename,
                            filemode='w')
        # define a Handler which writes INFO messages or higher to the sys.stderr
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        # set a format which is simpler for console use
        formatter = logging.Formatter('%(levelname)-8s %(message)s')
        # tell the handler to use this format
        console.setFormatter(formatter)
        # add the handler to the root logger
        logging.getLogger('').addHandler(console)
        
    @staticmethod
    def info(x):
        logging.info(x)
    @staticmethod
    def debug(x):
        logging.debug(x)
    @staticmethod
    def warning(x):
        logging.warning(x)
    @staticmethod
    def error(x):
        logging.error(x)

class Loader:
    # url="http://localhost:9000"
    training_data_path="train-v1.1.json"
    training_pkl_path="train-v1.1.pkl"
    dev_data_path="dev-v1.1.json"
    dev_pkl_path="dev-v1.1.pkl"
    word_emb_path="glove.840B.300d.txt"
    word_emb_pkl_path="glove.840B.300d.pkl"
    word_emb_for_train_pkl_path="glove.840B.300d.train.pkl"
    
    vocab = "vocab.pkl"
    ivovab = ""
    
    # nlp = None        
        
    # @staticmethod
    # def sent_tokenize_simple(doc):
    #     if Loader.nlp==None:
    #         pywrap.root.setLevel(logging.WARNING)
    #         Loader.nlp=pywrap.CoreNLP(url='http://localhost:9000', annotator_list=['tokenize','ssplit'])
    #     return Loader.nlp.basic(doc, out_format='json').json()

    # @staticmethod
    # def word_tokenize(doc):
    #     doc=doc.encode('ascii', 'mine').decode("ascii")
    #     #print(doc)
    #     ret=[]
    #     out=Loader.sent_tokenize_simple(doc)
    #     #print(out)
    #     #if isinstance(out,str):
    #     #    out=json.loads(out)
    #     for sentence in out["sentences"]:
    #         for token in sentence["tokens"]:
    #             ret.append((token["originalText"],
    #             token["characterOffsetBegin"],
    #             token["characterOffsetEnd"]-1))
    #     return ret
                
    @staticmethod
    def locate(tokens,pos):
        for i,v in enumerate(tokens):
            begin=v[1]
            end=v[2]
            if pos>=begin and pos<=end:
                return i
        raise Exception         


    @staticmethod
    def load_dev_data():
        with open(Loader.dev_data_path,"r") as f:
            dataset_json=json.load(f)
        return dataset_json["data"]
        
    @staticmethod
    def load_dev_data_fine():
        if os.path.isfile(Loader.dev_pkl_path):
            with open(Loader.dev_pkl_path,"rb") as f:
                ret=pkl.load(f)
        else:
            # with open(Loader.dev_data_path,"r") as f:
            #     dataset_json=json.load(f)
            # ret=[]
            # #print(dataset)
            # dataset=dataset_json["data"]
            # for article in dataset:
            #     for paragraph in article['paragraphs']:
            #         #try:
            #         paragraph_token=Loader.word_tokenize(paragraph["context"])
            #         #except:
            #         #    print(paragraph["context"])
            #         for qa in paragraph['qas']:
            #             question_token=Loader.word_tokenize(qa["question"])
            #             question_token.append(('NULL', question_token[len(question_token)-1][2]+2, question_token[len(question_token)-1][2]+3))
            #             ret.append({"id":qa["id"],"paragraph":paragraph_token,"question":question_token})
            with open(Loader.dev_data_path,"r") as f:
                dataset_json=json.load(f)
            ret=[]
            dataset=dataset_json["data"]
            for c in dataset:
                for p in c["paragraphs"]:
                    context = p["context"].split(' ')
                    for qa in p["qas"]:
                        question = word_tokenize(qa["question"])
                        ret.append({"id":qa["id"],"paragraph":word_tokenize(p["context"]),"question":question})
                      
            with open(Loader.dev_pkl_path,"wb") as f:
                pkl.dump(ret,f,pickle.HIGHEST_PROTOCOL)

        return ret
        
    @staticmethod
    def load_training_data():
        if os.path.isfile(Loader.training_pkl_path):
            with open(Loader.training_pkl_path,"rb") as f:
                ret=pkl.load(f)
        else:
            # with open(Loader.training_data_path,"r") as f:
            #     dataset_json=json.load(f)
            # ret=[]

            # dataset=dataset_json["data"]
            # for article in dataset:
            #     for paragraph in article['paragraphs']:
            #         #try:
            #         paragraph_token=Loader.word_tokenize(paragraph["context"])
            #         #except:
            #         #    print(paragraph["context"])
            #         for qa in paragraph['qas']:
            #             question_token=Loader.word_tokenize(qa["question"])
            #             question_token.append(('NULL', question_token[len(question_token)-1][2]+2, question_token[len(question_token)-1][2]+3))
            #             answer=qa["answers"][0]
            #             answer_start=answer["answer_start"]
            #             answer_text=answer["text"]
            #             start_token=Loader.locate(paragraph_token,answer_start)
            #             try:
            #                 end_token=Loader.locate(paragraph_token,answer_start+len(answer_text)-1)
            #             except:
            #                 print(paragraph["context"])
            #                 print(answer_start)                        
            #                 print(answer_text)
            #                 raise
            #             ret.append({"id":qa["id"],"paragraph":paragraph_token,"question":question_token,
            #                         "answer_start":start_token,"answer_end":end_token})
            #             return ret
            
            with open(Loader.training_data_path,"r") as f:
                dataset_json=json.load(f)
            ret=[]
            dataset=dataset_json["data"]
            for c in dataset:
                for p in c["paragraphs"]:
                    context = p["context"].split(' ')
                    for qa in p["qas"]:
                        question = word_tokenize(qa["question"])
                        for a in qa["answers"]:
                            answer = a["text"].strip()
                            answer_start = int(a["answer_start"])
                        answer_words = word_tokenize(answer + '.')
                        if answer_words[-1] == '.':
                            answer_words = answer_words[:-1]
                        else:
                            answer_words = word_tokenize(answer)
                        
                        prev_context_words = word_tokenize(p["context"][0:answer_start])
                        left_context_words = word_tokenize(p["context"][answer_start:])
                        answer_reproduce = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                w = left_context_words[i]
                                answer_reproduce.append(w)
                        
                        join_a = ' '.join(answer_words)
                        join_ar = ' '.join(answer_reproduce)

						#if not ((join_ar in join_a) or (join_a in join_ar)):
                        # if join_a != join_ar:
                            # print (join_ar)
                            # print (join_a)
                            # print ('answer:'+answer)
                            # count += 1
                        paragraph_token = ' '.join(prev_context_words+left_context_words)
                        question_token = ' '.join(question)
                        pos_list = []
                        for i in range(len(answer_words)):
                            if i < len(left_context_words):
                                pos_list.append(str(len(prev_context_words)+i))
                        if len(pos_list) == 0:
                            print (join_ar)
                            print (join_a)
                            print ('answer:'+answer)
                        assert(len(pos_list) > 0)
                        ret.append({"id":qa["id"],"paragraph":prev_context_words+left_context_words,"question":question,
                                    "answer_start":pos_list[0],"answer_end":pos_list[-1]})


            with open(Loader.training_pkl_path,"wb") as f:
                pkl.dump(ret,f,pickle.HIGHEST_PROTOCOL)              
               #print(ret)
                #raise
        return ret
    
    @staticmethod
    def load_word_emb():
        if os.path.isfile(Loader.word_emb_pkl_path):
            with open(Loader.word_emb_pkl_path,"rb") as f:
                ret=pkl.load(f)
        else:
            i_line=0
            embl=[]
            word2idx={}
            idx2word=[]
            with open(Loader.word_emb_path,"r") as f:
                for line in f:
                    values=line.split()
                    word=''.join(values[0:len(values)-300])
                    coefs=np.asarray(values[len(values)-300:],dtype='float32')
                    word2idx[word]=i_line
                    idx2word.append(word)
                    embl.append(coefs)
                    #print(ret[word])
                    i_line+=1
                    if i_line%100000==0: print(i_line)
            print(i_line)
            emb=np.vstack(embl)
            ret={"emb":emb,"word2idx":word2idx,"idx2word":idx2word}
            with open(Loader.word_emb_pkl_path,"wb") as f:
                pkl.dump(ret,f)
        return ret
        
    @staticmethod
    def load_word_emb_for_train():
        if os.path.isfile(Loader.word_emb_for_train_pkl_path):
            with open(Loader.word_emb_for_train_pkl_path,"rb") as f:
                ret=pkl.load(f)
                print (len(ret['word2idx'].keys()))
            with open(Loader.vocab,"rb") as f:
                ret1=pkl.load(f)
            return ret, ret1
        else:
            embl=[]
            word2idx={}
            idx2word=[]
            emb=Loader.load_word_emb()
            train_data=Loader.load_training_data()
            dev_data=Loader.load_dev_data_fine()
            words=set()
            for v in train_data:
                for word in v["paragraph"]:
                    words.add(word)
                for word in v["question"]:
                    words.add(word)
           
            for v in dev_data:
                for word in v["paragraph"]:
                    words.add(word)
                for word in v["question"]:
                    words.add(word)
                
            print("nb of words",len(words))
            vocab_dict = {}
            num = 0
            i_line=0
            for key,value in enumerate(emb["idx2word"]):
                # print (value)
                if value not in words:
                    continue
                vocab_dict[value] = num
                num += 1
                word2idx[value]=i_line
                idx2word.append(value)
                embl.append(emb["emb"][key])
                
                #print(key)
                #raise
                
                #assert np.shape(emb["emb"][key])==(300,)                
                
                #if i_line==12901:
                #    assert key==14014
                #
                #assert np.array_equal(embl[word2idx[value]],emb["emb"][key])                
                
                #print(ret[word])
                i_line+=1
                if i_line%1000==0: print(i_line)
            print(i_line)
            print (num)
            embr=np.vstack(embl)
            ret={"emb":embr,"word2idx":word2idx,"idx2word":idx2word}
            print (len(ret['word2idx'].keys()))
            print (len(vocab_dict.keys()))
            for w in words:
                if w in vocab_dict.keys(): continue
                vocab_dict[w] = num
                num += 1
            print (num)
            with open(Loader.word_emb_for_train_pkl_path,"wb") as f:
                pkl.dump(ret,f,pickle.HIGHEST_PROTOCOL)
            with open(Loader.vocab,"wb") as f:
                pkl.dump(vocab_dict,f,pickle.HIGHEST_PROTOCOL)
        return ret
        
class Evaluate:
    @staticmethod
    def normalize_answer(s):
        """Lower text and remove punctuation, articles and extra whitespace."""
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
    
        def white_space_fix(text):
            return ' '.join(text.split())
    
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
    
        def lower(text):
            return text.lower()
    
        return white_space_fix(remove_articles(remove_punc(lower(s))))
    
    
    @staticmethod
    def f1_score(prediction, ground_truth):
        prediction_tokens = Evaluate.normalize_answer(prediction).split()
        ground_truth_tokens = Evaluate.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1
    
    @staticmethod   
    def precision(prediction, ground_truth):
        prediction_tokens = Evaluate.normalize_answer(prediction).split()
        ground_truth_tokens = Evaluate.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        return precision
        
    @staticmethod
    def recall(prediction, ground_truth):
        prediction_tokens = Evaluate.normalize_answer(prediction).split()
        ground_truth_tokens = Evaluate.normalize_answer(ground_truth).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        recall = 1.0 * num_same / len(ground_truth_tokens)
        return recall
    
    
    @staticmethod
    def exact_match_score(prediction, ground_truth):
        return (Evaluate.normalize_answer(prediction) == Evaluate.normalize_answer(ground_truth))
    
    
    @staticmethod
    def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)
    
    
    @staticmethod
    def evaluate(predictions):
        dataset=Loader.load_dev_data()
        f1 = exact_match = precision=recall=total = 0
        for article in dataset:
            for paragraph in article['paragraphs']:
                context = paragraph['context']
                for qa in paragraph['qas']:
                    total += 1
                    if qa['id'] not in predictions:
                        message = 'Unanswered question ' + qa['id'] + \
                                  ' will receive score 0.'
                        print(message, file=sys.stderr)
                        continue
                    ground_truths = list(map(lambda x: x['text'], qa['answers']))
                    prediction = predictions[qa['id']]
                    prediction = prediction.replace("``", "\"")
                    prediction = prediction.replace("''", "\"")

                    words = prediction.split()
                    after = []
                    preword = ''
                    for w in words:
                        if preword!='':
                            if after[-1]+w in context:
                                after[-1] = after[-1] + w
                            else:
                                after.append(w)
                        else:
                            after.append(w)
                        preword = w
                    prediction = ' '.join(after)

                    
                    exact_match += Evaluate.metric_max_over_ground_truths(
                        Evaluate.exact_match_score, prediction, ground_truths)
                    f1 += Evaluate.metric_max_over_ground_truths(
                        Evaluate.f1_score, prediction, ground_truths)
                    precision += Evaluate.metric_max_over_ground_truths(
                        Evaluate.precision, prediction, ground_truths)
                    recall += Evaluate.metric_max_over_ground_truths(
                        Evaluate.recall, prediction, ground_truths)
    
        exact_match = exact_match / total
        f1 = f1 / total
        precision=precision/total
        recall=recall/total
    
        return {'exact_match': exact_match, 'f1': f1, 'precision':precision,'recall':recall}
        
        
class TrainIterator:
    def __init__(self, word_emb, vocab, batch_size):
        self.ret = Loader.load_training_data()
        		
        self.word_dict=word_emb["word2idx"]
        self.vocab = vocab
        self.batch_size = batch_size
        
        self.reset()

    def __iter__(self):
        return self
		
    def reset(self):
        self.index = 0
        random.shuffle(self.ret)

    def __next__(self):
        para = []
        question = []
        answer = []
        tpara=[]
        while True:
            if self.index >= len(self.ret):
                self.reset()
            tmpret = self.ret[self.index]
            tmppara = tmpret["paragraph"]
            tmpques = tmpret["question"]
            tmpstart = int(tmpret["answer_start"])
            tmpend = int(tmpret["answer_end"])

            
                    				
            tmppara = [self.vocab[w] for w in tmppara]
            tmpques = [self.vocab[w] for w in tmpques]
            # tmpques.append(self.word_dict[b'NULL'])      
            #print(tmpques)                      
                      
            tpara.append(tmpret["paragraph"])
            para.append(tmppara)
            question.append(tmpques)
            tmpans = [tmpstart,tmpend]
            #for i in range(tmpstart, tmpend+1):
            #    tmpans.append(i)
            
            answer.append(tmpans)
            self.index += 1
            if len(para) >= self.batch_size:
                break
				
        return para,question,answer,tpara
        
class DevIterator:
    def __init__(self, word_emb, vocab):
        self.ret = Loader.load_dev_data_fine()
        		
        self.word_dict=word_emb["word2idx"]
        self.vocab = vocab
        self.index=0
    
    def __iter__(self):
        return self
		

    def __next__(self):
        if self.index >= len(self.ret):
            raise StopIteration
        tmpret = self.ret[self.index]
    				
        idx=tmpret["id"]            
        
        tmppara = tmpret["paragraph"]
        tmpques = tmpret["question"]

        
                				
        para = [self.vocab[w] for w in tmppara]
        ques = [self.vocab[w] for w in tmpques]
        # ques.append(self.word_dict[b'NULL']) 
        #for i in range(tmpstart, tmpend+1):
        #    tmpans.append(i)
        
        self.index += 1
				
        return idx,para,ques,tmppara
        
        
def debug():
    #a=Loader.load_word_emb()
    #print(a["idx2word"][a["word2idx"][b"."]])
    #print(np.shape(a["emb"]))


    a=Loader.load_training_data()[1]
    print(a["paragraph"][a["answer_start"]:a["answer_end"]+1])
    
    
def test_new_word_emb():
    word=b"zoo"
    emb=Loader.load_word_emb()
    new_emb=Loader.load_word_emb_for_train()
    old_id=emb["word2idx"][word]
    new_id=new_emb["word2idx"][word]
    print(old_id,new_id)
    print(np.array_equal(emb["emb"][old_id],new_emb["emb"][new_id]))
   

def handler(e): 
    return (u'-' * (e.end-e.start), e.end) 

# codecs.register_error('mine',handler)    

def write_into_pkl():
    train_file = 'train.txt'
    dev_file = 'dev.txt'
    ftr = open(train_file, 'r')
    lines = ftr.readlines()
    ftr.close()
    ret = []
    for line in lines:
        divs = line.split('\t')
        paragraph_token = ''
        ret.append({"paragraph":paragraph_token,"question":question_token,
                                    "answer_start":start_token,"answer_end":end_token})
    
if __name__=="__main__":
    # test_new_word_emb()
    # log.init('result')
    # ret = Loader.load_training_data()
    # ret = Loader.load_dev_data_fine()
    ret1 = Loader.load_word_emb_for_train()
    # for i in range(len(ret)):
    #     for w in ret[i]['paragraph']:
    #         if w in ret1:
    #             print (w)
    
        