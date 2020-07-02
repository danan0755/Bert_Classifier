#!/usr/bin/env python
# -*- coding: utf-8 -*-


# author chenyongsheng
# date 20200608

import json
import os
import pickle

import numpy as np
import requests
import tensorflow as tf

from Business.Intent.Out_Call import tokenization


class InputExample(object):

    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


def _create_examples(lines, label2id_file):
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            label_list = [key for key in label2id.keys()]
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % ('test', i)
        text_a = tokenization.convert_to_unicode(line)
        text_b = None
        label = label_list[0]
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class PaddingInputExample(object):
    """
    """


class InputFeatures(object):

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_id = label_map[example.label]
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        is_real_example=True)
    return feature


def request_from_raw_text(vocab_file, label2id_file, query, model_key):
    """

    :return:
    """
    text_list = [query]
    data_list = []
    label_list = []
    if os.path.exists(label2id_file):
        with open(label2id_file, 'rb') as rf:
            label2id = pickle.load(rf)
            id2label = {value: key for key, value in label2id.items()}
            label_list = [key for key in label2id.keys()]

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=True)

    predict_examples = _create_examples(text_list, label2id_file)
    for (ex_index, example) in enumerate(predict_examples):
        feature = convert_single_example(ex_index, example, label_list, 128,
                                         tokenizer)  # ex_index, example, label_list, max_seq_length,tokenizer

        features = {}
        features["input_ids"] = feature.input_ids
        features["input_mask"] = feature.input_mask
        features["segment_ids"] = feature.segment_ids

        features["label_ids"] = feature.label_id

        data_list.append(features)

    data = json.dumps({"signature_name": "serving_default", "instances": data_list})
    headers = {"content-type": "application/json"}
    json_response = requests.post('http://localhost:8501/v1/models/{}:predict'.format(model_key), data=data,
                                  headers=headers)
    predictions = json.loads(json_response.text)
    p_list = predictions.get('predictions')[0]
    label_index = np.argmax(p_list)
    label = id2label.get(label_index)
    pred_score = max(p_list)
    return pred_score, label


if __name__ == '__main__':
    request_from_raw_text()
