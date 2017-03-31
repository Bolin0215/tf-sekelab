import argparse
import json
import os
import nltk
from squad.utils import *
from tqdm import tqdm
from collections import defaultdict

word_counter = {}

def main():
    args = get_args()
    prepro(args)


def get_args():
    parser = argparse.ArgumentParser()
    source_dir = os.path.join('.', 'data', 'squad')
    target_dir = os.path.join('.', 'data', 'squad')
    glove_dir = os.path.join('.', 'data', 'glove')

    parser.add_argument('-s', "--source_dir", default=source_dir)
    parser.add_argument('-t', "--target_dir", default=target_dir)
    parser.add_argument('--glove_dir', default=glove_dir)
    parser.add_argument('--glove_corpus', default='6B')
    parser.add_argument("--glove_vec_size", default=100, type=int)

    return parser.parse_args()


def prepro(args):
    prepro_each_select(args, 'train', out_name='train_s')
    prepro_each_select(args, 'dev', out_name='dev_s')
    word2vec_dict = get_word2vec(args, word_counter)
    save_vec(args, word2vec_dict, 'train_s')


def save(args, data, data_type):
    data_path = os.path.join(args.target_dir, "data_{}.json".format(data_type))
    json.dump(data, open(data_path, 'w'))


def prepro_each_select(args, data_type, out_name='default', num=1):
    sent_tokenizer = nltk.sent_tokenize

    def word_tokenizer(tokens):
        return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

    with open(os.path.join(args.source_dir, '{}-v1.1.json'.format(data_type)), 'r') as f:
        dataset = json.load(f)["data"]

    processed = []
    cnt = 0
    ret = False
    wrongcnt = 0

    max_sent_num = -1
    max_pword_num = -1
    max_qword_num = -1
    for data in dataset:
        for para in data["paragraphs"]:
            if ret: break
            context = para["context"].replace("''", '" ').replace("``", '" ')
            sents = list(map(word_tokenizer, sent_tokenizer(context)))
            sents = [process_token(sent) for sent in sents]
            max_sent_num = max_sent_num if max_sent_num >= len(sents) else len(sents)
            for sent in sents:
                max_pword_num = max_pword_num if max_pword_num >= len(sent) else len(sent)
                for word in sent:
                    word_counter.setdefault(word.lower(), 0)
                    word_counter[word.lower()] += 1
            for qa in para["qas"]:
                if ret: break
                question = word_tokenizer(qa["question"])
                max_qword_num = max_qword_num if max_qword_num >= len(question) else len(question)
                for word in question:
                    word_counter.setdefault(word.lower(), 0)
                    word_counter[word.lower()] += 1
                ans_start, ans_stop = -1, -1
                ans_text = None
                ans_cnt = defaultdict(int)
                for ans in qa["answers"]:
                    ans_text = ans['text']
                    ans_start = int(ans['answer_start'])
                    ans_stop = ans_start + len(ans_text)
                    ans_tuple = (ans_start, ans_stop, ans_text)
                    ans_cnt[ans_tuple] += 1
                max_cnt = -111
                for ans in ans_cnt:
                    if ans_cnt[ans] > max_cnt:
                        max_cnt = ans_cnt[ans]
                        ans_start, ans_stop = ans[0], ans[1]
                        ans_text = ans[2]

                yi0, yi1 = get_word_span(context, sents, ans_start, ans_stop)
                y = [yi0, yi1]
                cnt += 1
                if yi0[0] != yi1[0]:
                    wrongcnt += 1
                    continue
                    # print (ans_text)
                processed.append({"id": qa["id"], "sents": sents, "qa": question, "ans_id": y, "context":context, "ans_text":ans_text})

    print('saving {}...'.format(out_name))
    print('datatype: {} {} {}'.format(cnt, wrongcnt, float(wrongcnt) / cnt))
    print('max sentence num is {}, max pword num is {}, max qword num is {}'.format(max_sent_num, max_pword_num,
                                                                                    max_qword_num))
    # data = {'data':processed, 'msn':max_sent_num, 'mqn':max_qword_num, 'mpn':max_pword_num}
    save(args, processed, out_name)


def save_vec(args, word2vec_dict, type):
    print('saving word vectors...')
    data_path = os.path.join(args.target_dir, "glove_{}.json".format(type))
    json.dump(word2vec_dict, open(data_path, 'w'))


def get_word2vec(args, word_counter):
    glove_path = os.path.join(args.glove_dir, 'glove.{}.{}d.txt'.format(args.glove_corpus, args.glove_vec_size))
    sizes = {'6B': int(4e5), '42B': int(1.9e6), '840B': int(2.2e6), '2B': int(1.2e6)}
    total = sizes[args.glove_corpus]
    word2vec_dict = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, total=total):
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            vector = list(map(float, array[1:]))
            if word in word_counter:
                word2vec_dict[word] = vector
    print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter),
                                                                        glove_path))
    return word2vec_dict


if __name__ == "__main__":
    main()
