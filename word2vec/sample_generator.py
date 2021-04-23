import random
import tensorflow as tf
import numpy as np
from itertools import chain

from vocabulary import Vocabulary

class SampleGenerator():
  def __init__(self, corpus, vocabulary, window_size = 5, negative_sample_size = 7):
    self.corpus = corpus
    self.vocabulary = vocabulary
    
    self.window_size = window_size
    self.negative_sample_size = negative_sample_size
    self.C = int(self.window_size / 2)

    self.window_index = 0
    self.neg_sample_index = 0

    # WRS helper
    self.sample_box = vocabulary.make_sample_box()

    self.i = 0
    
  def __call__(self, *args, **kwargs):
    return self

  def __iter__(self):
    neg_samples = [0] * self.negative_sample_size

    for line in open(self.corpus):
      self.i += 1
      if self.i % 10000 == 0:
        print(self.i)
      line = line.strip()
      tokens = [int(t) for t in line.split(',')]
      for i in range(len(tokens)):
        word = tokens[i]
        #TODO do subsampling here or before dataset?

        left_start = max(0, i - self.C)
        left_end = i
        right_start = min(len(tokens), i+1)
        right_end = min(len(tokens), i+1+self.C)

        left_range = range(left_start, left_end)
        right_range = range(right_start, right_end)

        window = chain(left_range, right_range)
        context_words = [tokens[w] for w in window]

        if len(context_words) == 0:
          continue # skip this word


        for context_word in context_words:
          yield (word, context_word), 1

          self.negative_sampling([context_word], neg_samples)
          for neg_sample in neg_samples:
            yield (word, neg_sample), 0

        # negative sampling each context window
        # self.negative_sampling(context_words, neg_samples)
        # for neg_sample in neg_samples:
        #   yield (word, neg_sample), 0

  def negative_sampling(self, excludes, neg_samples):
    sampled = 0
    exclude_map = {k:1 for k in excludes}

    while sampled != self.negative_sample_size:
      r = random.randint(0, len(self.sample_box) - 1)
      cand = self.sample_box[r]

      if cand in exclude_map:
        continue
 
      neg_samples[sampled] = cand
      exclude_map[cand] = 1
      sampled += 1

if __name__ == '__main__':
  vocab = Vocabulary('../data/vocab.txt')
  vocab.load()

  gen = SampleGenerator('../data/3b_token.txt.final', vocab)

  i  = 0
  for s in gen:
    i += 1
    if i % 100000 == 0:
      print(s)
      print(i)
