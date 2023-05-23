# Copyright 2021 The mT5 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of various metrics to be used with the T5 library.
"""


from __future__ import print_function

import collections
import re
import string
import sys
import unicodedata


def normalize_mlqa(s, lang, punct):
  """Lower text and remove punctuation, articles and extra whitespace.

  Based on third_party/py/xtreme/third_party/evaluate_mlqa.py
  Args:
    s: string, typically the answer span predicted by a QA model.
    lang: ISO code of language.
    punct: set of punctuation characters.

  Returns:
    string, after applying normalization rules.
  """

  whitespace_langs = ['en', 'es', 'hi', 'vi', 'de', 'ar']
  mixed_segmentation_langs = ['zh']

  def whitespace_tokenize(text):
    return text.split()

  def mixed_segmentation(text):
    segs_out = []
    temp_str = ''
    for char in text:
      if re.search(r'[\u4e00-\u9fa5]', char) or char in punct:
        if temp_str != '':
          ss = whitespace_tokenize(temp_str)
          segs_out.extend(ss)
          temp_str = ''
        segs_out.append(char)
      else:
        temp_str += char
    if temp_str != '':
      ss = whitespace_tokenize(temp_str)
      segs_out.extend(ss)
    return segs_out

  def drop_articles(text, lang):
    if lang == 'en':
      return re.sub(r'\b(a|an|the)\b', ' ', text)
    elif lang == 'es':
      return re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
    elif lang == 'hi':
      return text
    elif lang == 'vi':
      return re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
    elif lang == 'de':
      return re.sub(
          r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b',
          ' ', text)
    elif lang == 'ar':
      return re.sub('\sال^|ال', ' ', text)
    elif lang == 'zh':
      return text

  def white_space_fix(text, lang):
    if lang in whitespace_langs:
      tokens = whitespace_tokenize(text)
    elif lang in mixed_segmentation_langs:
      tokens = mixed_segmentation(text)
    return ' '.join([t for t in tokens if t.strip()])

  def drop_punc(text):
    return ''.join(c for c in text if c not in punct)

  s = s.lower()
  s = drop_punc(s)
  s = drop_articles(s, lang)
  s = white_space_fix(s, lang)
  return s

def flatten(listoflists):
  return [item for listy in listoflists for item in listy]

def span_f1(targets, predictions, data_format='paolini_copy_brackets'):
  #print("data_format: {}".format(data_format))
  """Computes Span based F1 score.

  Args:
    targets: list of strings or list of list of strings if multiple references
      are present.
    predictions: list of strings

  Returns:
    span f1 across all targets and predictions (Based on CoNLL script)
  """
  true_positives = collections.defaultdict(int)
  false_positives = collections.defaultdict(int)
  false_negatives = collections.defaultdict(int)

  def tags_to_spans(tag_sequence, delimiters=[' ## ', ' \$\$ ']):
    if tag_sequence in ['Null', 'None', 'null']:
      return []
    """Extract spans from IOB1 or BIO tags."""
    tag_sequence_split = [x.strip() for x in re.split('|'.join(delimiters), tag_sequence)]
    tags_entities = []
    for tag_entity in tag_sequence_split:
      tag_entity_split = tag_entity.split(': ')
      ## If the entity cannot be split, first try without the space
      if len(tag_entity_split) != 2:
        tag_entity_split = tag_entity.split(':')
      ## If still not split, this is an error, don't make it easier for model!
      if len(tag_entity_split) != 2:
        #print("Bad entity! {}".format(tag_entity))
        tag = ""
        entity = tag_entity
      else:
        tag = tag_entity_split[0].strip()
        entity = tag_entity_split[1].strip()
      tags_entities.append((tag, entity))
    return tags_entities

  def extract_sentinel_to_spans(sequence):
    errors = 0
    if sequence in ['Null', 'None', 'null']:
      return [], errors
    tags_entities = []
    sequence_split = sequence.split()
    cur_idx = 0
    cur_entity, cur_tag = [-1, -1], ''
    while cur_idx < len(sequence_split):
      ## Checks the id is an action sentinel token
      try:
        sentinel_id = int(sequence_split[cur_idx].split('_')[-1][:-1])
      except:
        sentinel_id = ''
      ## Checks the tag exists
      try:
        tag = sequence_split[cur_idx+1]
      except:
        tag = ''

      if tag == 'I':
        ## Continues current entity
        cur_entity[1] = sentinel_id
      else:
        ## Adds completed entity to list
        if cur_entity != [-1, -1] or cur_tag != '':
          tags_entities.append((cur_tag, cur_entity))
          if cur_entity == [-1, -1] or cur_tag == '':
            errors += 1
        ## Starts new entity
        cur_entity = [sentinel_id, sentinel_id]
        cur_tag = tag
      ## Iterates by 2
      cur_idx += 2

    ## There should always be a last entity
    tags_entities.append((cur_tag, cur_entity))
    if cur_entity == [-1, -1] or cur_tag == '':
      errors += 1
    return tags_entities, errors

  def liu_to_spans(sequence):
    errors = 0
    if '[' not in sequence:
      return [], errors
    spans = [s.split("]")[0] for s in sequence.split("[")[1:]]
    tags_entities = []
    for span in spans:
      tag = span.split(' ')[0]
      entity = ' '.join(span.split(' ')[1:])
      if entity == '':
        errors += 1
      tags_entities.append((tag, entity))
    return tags_entities, errors

  def paolini_to_spans(sequence):
    errors = 0
    if '[' not in sequence:
      return [], errors
    spans = [s.split("]")[0] for s in sequence.split("[")[1:]]
    tags_entities = []
    for span in spans:
      try:
        tag = span.split(' | ')[1]
      except IndexError:
        tag = ''
        errors += 1
      entity = span.split(' | ')[0]
      tags_entities.append((tag, entity))
    return tags_entities, errors

  def compute_f1_metrics(true_positives, false_positives, false_negatives):
    precision = float(true_positives) / float(true_positives + false_positives +
                                              1e-13)
    recall = float(true_positives) / float(true_positives + false_negatives +
                                           1e-13)
    f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
    return precision, recall, f1_measure

  all_gold_spans, all_pred_spans = 0, 0
  all_gold_errors, all_pred_errors = 0, 0
  for target, pred in zip(targets, predictions):
    if 'original' in data_format or data_format == 'sentinel_tag_indices':
      gold_spans = tags_to_spans(target)
      predicted_spans = tags_to_spans(pred)
    elif data_format == 'extractive_sentinel_tag':
      gold_spans, gold_errors = extract_sentinel_to_spans(target)
      predicted_spans, pred_errors = extract_sentinel_to_spans(pred)
      all_gold_spans += len(gold_spans)
      all_pred_spans += len(predicted_spans)
      all_gold_errors += gold_errors
      all_pred_errors += pred_errors
    elif data_format == 'liu_copy_brackets':
      gold_spans, gold_errors = liu_to_spans(target)
      predicted_spans, pred_errors = liu_to_spans(pred)
      all_gold_spans += len(gold_spans)
      all_pred_spans += len(predicted_spans)
      all_gold_errors += gold_errors
      all_pred_errors += pred_errors
    elif data_format == 'paolini_copy_brackets':
      gold_spans, gold_errors = paolini_to_spans(target)
      predicted_spans, pred_errors = paolini_to_spans(pred)
      all_gold_spans += len(gold_spans)
      all_pred_spans += len(predicted_spans)
      all_gold_errors += gold_errors
      all_pred_errors += pred_errors
    else:
      gold_spans = tags_to_spans(target)
      predicted_spans = tags_to_spans(pred)
    #print("\nTarget: {}\nGold Spans: {}".format(target, gold_spans))
    #print("\nPred: {}\nPred spans: {}\n".format(pred, predicted_spans))

    for span in predicted_spans:
      if span in gold_spans:
        true_positives[span[0]] += 1
        gold_spans.remove(span)
      else:
        false_positives[span[0]] += 1
    # These spans weren't predicted.
    for span in gold_spans:
      false_negatives[span[0]] += 1

    #print("\n\ntp: {}\nfp: {}\nfn: {}".format(true_positives, false_positives, false_negatives))
    #import pdb; pdb.set_trace()

  if 'copy_bracket' in data_format or data_format == 'extractive_sentinel_tag':
    print("With {}:\nGold errors: {} out of {} spans, Pred errors: {} out of {} spans".format(data_format, all_gold_errors, all_gold_spans, all_pred_errors, all_pred_spans))
  precision, recall, f1_measure = compute_f1_metrics(
      sum(true_positives.values()), sum(false_positives.values()),
      sum(false_negatives.values()))

  result = {'span_f1': f1_measure, 'span_recall': recall, 'span_precision': precision}
  return result
