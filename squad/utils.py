import re

def get_2d_spans(text, tokenss):
    spanss = []
    cur_idx = 0
    for tokens in tokenss:
        spans = []
        for token in tokens:
            if text.find(token, cur_idx) < 0:
                print ('something wrong with processing span...')
                print ('{} {} {}'.format(token, cur_idx, text))
                raise Exception()
            cur_idx = text.find(token, cur_idx)
            spans.append((cur_idx, cur_idx + len(token)))
            cur_idx += len(token)
        spanss.append(spans)
    return spanss

def get_word_span(context, wordss, start, stop):
    spanss = get_2d_spans(context, wordss)
    idx = []
    for sent_idx, spans in enumerate(spanss):
        for word_idx, span in enumerate(spans):
            if stop > span[0] and start < span[1]:
                idx.append((sent_idx, word_idx))
    return idx[0], idx[-1]

def get_best_span(yp, yp2):
    max_val = 0
    best_s, best_e = -1, -1
    for i in range(len(yp)):
        for j in range(i + 1, len(yp2)):
            if max_val < yp[i] * yp2[j]:
                max_val = yp[i] * yp2[j]
                best_s = i
                best_e = j
    return (best_s, best_e)

def get_phrase(context, wordss, span):
    char_start, char_stop = None, None
    char_idx = 0
    words = sum(wordss, [])
    for word_id, word in enumerate(words):
        char_idx = context.find(word, char_idx)
        assert char_idx >= 0
        if word_id == span[0]:
            char_start = char_idx
        char_idx += len(word)
        if word_id == span[1]:
            char_stop = char_idx
    assert char_start is not None
    assert char_stop is not None
    return context[char_start:char_stop]

def process_token(temp_tokens):
    tokens = []
    for token in temp_tokens:
        l = ("-", "\u2212", "\u2014", "\u2013", "/", "~", '"', "'", "\u201C", "\u2019", "\u201D", "\u2018", "\u00B0")
        tokens.extend(re.split("([{}])".format("".join(l)), token))
    return tokens

