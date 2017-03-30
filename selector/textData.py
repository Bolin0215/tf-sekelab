import os
import json
import numpy as np
import pickle as pkl

class pair:
    def __init__(self):
        self.id = None
        self.p = None
        self.q = None
        self.ans = None

class Batch:
    def __init__(self):
        self.id = None
        self.p = None
        self.p_mask = None
        self.q = None
        self.q_mask = None
        self.y = None
        self.y_mask = None

        self.pad_q_mask = None
        self.pad_p_mask = None

class textData:
    def __init__(self, args):
        self.args = args
        self.word2idx = {}
        self.idx2word = {}
        self.val_text = {}

        self.unk_word = 'UNK'
        self.pad_word = 'PAD'
        self.word2idx[self.pad_word] = 0
        self.idx2word[0] = self.pad_word
        self.word2idx[self.unk_word] = 1
        self.idx2word[1] = self.unk_word

        # load pretrained embedding
        self.gen_pretrain_word_vec()
        self.untrainableCnt = len(self.word2idx)
        print('{} words embedding are untrainable'.format(self.untrainableCnt))

        self.total_idx = self.untrainableCnt

        if self.args.mode == 'test':
            loadfrom = os.path.join(self.args.load_path, 'best')
            with open('{}_word2idx.pkl'.format(loadfrom), 'rb') as f:
                self.word2idx = pkl.load(f)
            with open('{}_idx2word.pkl'.format(loadfrom), 'rb') as f:
                self.idx2word = pkl.load(f)
            self.total_idx = len(self.word2idx)
        self.read_all_data()
        self.vocabSize = len(self.word2idx)
        assert self.total_idx == self.vocabSize


        self.val_predict = {}
        # self.train_batches = self.gen_Batches('train')
        # self.val_batches = self.gen_Batches('test')

    def gen_pretrain_word_vec(self):
        vec_path = os.path.join(self.args.emb_dir, "glove_train_s.json")
        with open(vec_path, 'r') as f:
            vecs = json.load(f)
        vectors = []
        self.word_emb = np.zeros((2, self.args.embeddingSize)).astype(np.float32)
        for idx, word in enumerate(vecs):
            self.word2idx[word] = idx + 2
            self.idx2word[idx + 2] = word
            vectors.append(vecs[word])
        self.word_emb = np.concatenate([self.word_emb, np.array(vectors).astype(np.float32)])

    def _read_data(self, data_type):
        data_path = os.path.join(self.args.data_dir, "data_{}_s.json".format(data_type))
        with open(data_path, 'r') as f:
            datas = json.load(f)

        num_examples = len(datas)
        # if data_type == 'train':
        #     self.trainCnt = num_examples
        print ('{} total {} examples... '.format(data_type, num_examples))

        new_datas = []

        for data in datas:
            npair = pair()
            npair.id = data["id"]
            sent_ids = []
            for sent in data["sents"]:
                word_ids = []
                for w in sent:
                    w = w.lower()
                    if w in self.word2idx:
                        word_ids.append(self.word2idx[w])
                    else:
                        self.word2idx[w] = self.total_idx
                        self.idx2word[self.total_idx] = w
                        word_ids.append(self.total_idx)
                        self.total_idx += 1
                sent_ids.append(word_ids)
            if not data_type == 'train':
                self.val_text[npair.id] = [data["sents"], data["qa"]]
            npair.p = sent_ids
            qas_ids = []
            qas = data["qa"]
            for w in qas:
                w = w.lower()
                if w in self.word2idx:
                    qas_ids.append(self.word2idx[w])
                else:
                    self.word2idx[w] = self.total_idx
                    self.idx2word[self.total_idx] = w
                    qas_ids.append(self.total_idx)
                    self.total_idx += 1
            npair.q = qas_ids
            npair.ans = [data["ans_id"][0][0], data["ans_id"][1][0]]
            new_datas.append(npair)

        np.random.shuffle(new_datas)
        return new_datas

    def read_all_data(self):
        self.train_data = self._read_data('train')
        self.val_data = self._read_data('dev')

    def gen_Batches(self, data_type):
        batches_sample = []
        samples = self.train_data if data_type == 'train' else self.val_data
        numSamples = len(samples)
        for i in range(0, numSamples, self.args.batchSize):
            batch_sample = samples[i: min(i + self.args.batchSize, numSamples)]
            if len(batch_sample) != self.args.batchSize:
                break
            max_sent_num, max_pword_num, max_qword_num = -1, -1, -1
            for pair in batch_sample:
                max_sent_num = len(pair.p) if max_sent_num < len(pair.p) else max_sent_num
                for sent in pair.p:
                    max_pword_num = len(sent) if max_pword_num < len(sent) else max_pword_num
                max_qword_num = len(pair.q) if max_qword_num < len(pair.q) else max_qword_num

            new_p = np.zeros((len(batch_sample), max_sent_num, max_pword_num)).astype(np.int32)
            new_pad_q_mask = np.zeros((len(batch_sample), max_sent_num, max_qword_num)).astype(np.float32)
            new_pad_p_mask = np.zeros((len(batch_sample), max_sent_num, max_pword_num)).astype(np.float32)
            new_p_mask = np.zeros((len(batch_sample), max_sent_num, max_pword_num)).astype(bool)
            new_q = np.zeros((len(batch_sample), max_sent_num, max_qword_num)).astype(np.int32)
            new_q_mask = np.zeros((len(batch_sample), max_sent_num, max_qword_num)).astype(bool)
            new_y = np.zeros((len(batch_sample),)).astype(np.int32)
            new_y_mask = np.zeros((len(batch_sample), max_sent_num)).astype(np.int64)
            new_ids = []
            for idx, pair in enumerate(batch_sample):
                new_ids.append(pair.id)
                for jdx, sent in enumerate(pair.p):
                    new_p[idx][jdx][:len(sent)] = sent
                    new_pad_q_mask[idx][jdx][:len(pair.q)] = 1.
                    new_pad_p_mask[idx][jdx][:len(sent)] = 1.
                    new_p_mask[idx][jdx][:len(sent)] = True
                    new_q[idx][jdx][:len(pair.q)] = pair.q
                    new_q_mask[idx][jdx][:len(pair.q)] = True
                for jdx in range(len(pair.p), max_sent_num):
                    new_pad_q_mask[idx][jdx][:len(pair.q)] = 1.
                    new_pad_p_mask[idx][jdx][:len(pair.p[-1])] = 1.
                new_y[idx] = pair.ans[0]
                new_y_mask[idx][:len(pair.p)] = 1.

            new_batch = Batch()
            new_batch.id = new_ids
            new_batch.p = new_p
            new_batch.p_mask = new_p_mask
            new_batch.q = new_q
            new_batch.q_mask = new_q_mask
            new_batch.y = new_y
            new_batch.y_mask = new_y_mask
            new_batch.pad_q_mask = new_pad_q_mask
            new_batch.pad_p_mask = new_pad_p_mask
            batches_sample.append(new_batch)

        return batches_sample

    def write_Result(self, ids, pred, truth):
        for id, p, t in zip(ids, pred, truth):
            truth_text = ' '.join(self.val_text[id][0][t])
            pred_text = ' '.join(self.val_text[id][0][p])
            self.val_predict[id] = [' '.join(self.val_text[id][1]), truth_text, pred_text]

