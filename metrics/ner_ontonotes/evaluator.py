# encoding=utf8
from collections import defaultdict

import numpy as np
from metrics.ner_ontonotes.ner_metrics import span_f1

# this function is adapt from tapex
'''def evaluate_example(_predict_str: str, _ground_str: str):
    target_delimiter = ', '
    _predict_spans = _predict_str.split(target_delimiter)
    _ground_spans = _ground_str.split(target_delimiter)
    _predict_values = defaultdict(lambda: 0)
    _ground_values = defaultdict(lambda: 0)
    for span in _predict_spans:
        # try:
        #     _predict_values[float(span)] += 1
        # except ValueError:
            _predict_values[span.strip()] += 1
    for span in _ground_spans:
        # try:
        #     _ground_values[float(span)] += 1
        # except ValueError:
            _ground_values[span.strip()] += 1
    _is_correct = _predict_values == _ground_values

    return _is_correct'''


class EvaluateTool(object):

    def __init__(self, args):
        self.args = args

    def evaluate(self, preds, golds, section):
        gold_inferreds = [item["seq_out"] for item in golds]
        results = span_f1(preds, gold_inferreds)
        return results
