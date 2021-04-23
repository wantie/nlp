import pandas as pd
import re
import jieba
import math
import numpy as np

class CorpusProcess:
  def __init__(self, data_path, vocab_file = None, token_file = None, stop_file = None):
    self.data_path = data_path
    self.vocab_file = vocab_file
    self.token_file = token_file
    self.stop_file = stop_file
    self.vocabulary = {}
    self.token_id_map = {}
    self.id_token_map = {}

    self.token_count = 0
    self.min_count = 5 
    self.t = 0.1

  def process(self):
    f = open(self.token_file, 'w')
    f_id = open(self.token_file + '.id', 'w')
    for line in open(self.data_path):
      #parts = line.split('_!_')
      #paragraph = parts[3].strip() + '。' + parts[4].strip()
      paragraph = line.strip()
      if len(paragraph) == 0:
        continue
      sentenses = self.cut_sent(paragraph)
      for s in sentenses :
        tokens = self.tokenize(s)
        #tokens = self.del_stop_words(tokens)
        if len(tokens) != 0 :
          self.to_vocabulary(tokens)
          f.write(','.join(tokens))
          f.write('\n')
          f_id.write(','.join([str(self.token_id_map[t]) for t in tokens]))
          f_id.write('\n')
    f.close()
    f_id.close()

    self.min_count_filter()
    self.subsampling()

  def tokenize(self, sentense):
    return jieba.lcut(sentense)

  def to_vocabulary(self, tokens):
    for t in tokens :
      if t not in self.vocabulary.keys():
        self.vocabulary[t] = 1
        self.token_id_map[t] = self.token_count
        self.token_count += 1
      else :
        self.vocabulary[t] += 1

  def new_vocabulary(self):
    self.vocabulary = {}
    self.token_id_map = {}
    self.token_count = 0

  def min_count_filter(self):
    f_min_count = open(self.token_file + '.min_count', 'w')
    for line in open(self.token_file, 'r'):
      line = line.strip()
      ss = line.split(',')
      if len(ss) == 0:
        continue
      
      print('line:', line)
      print('ss:', ss)
      filtered = [s for s in ss if s != '' and self.vocabulary[s] > self.min_count]
      if len(filtered) == 0:
        continue
      f_min_count.write(','.join(filtered))
      f_min_count.write('\n')

    f_min_count.close()
    self.new_vocabulary()

    f_min_count_id = open(self.token_file + '.min_count.id', 'w')
    for line in open(self.token_file + '.min_count', 'r'):
      tokens = line.strip().split(',')
      if len(tokens) != 0:
        self.to_vocabulary(tokens)
        f_min_count_id.write(','.join([str(self.token_id_map[t]) for t in tokens]))
        f_min_count_id.write('\n')

    f_min_count_id.close()

  def subsampling(self):
    sample_rate = [0.0]*len(self.vocabulary)
    total_word_count = sum(self.vocabulary.values())

    for k,v in self.vocabulary.items():
      token_id = self.token_id_map[k]
      f_w = v / total_word_count
      sample_rate[token_id] = 1.0 - math.sqrt(self.t / f_w)

    f_final = open(self.token_file + '.final', 'w')
    for line in open(self.token_file + '.min_count.id', 'r'):
      tokens = [int(t) for t in line.strip().split(',')]
      rands = np.random.uniform(0, 1, len(tokens))
      sampled = [tokens[i] for i in range(len(tokens)) if sample_rate[tokens[i]] < rands[i]]
      if len(sampled) != 0:
        f_final.write(','.join([str(t) for t in sampled]))
        f_final.write('\n')

    f_final.close()
      
  def del_stop_words(self, segs):
    segs = [w for w in segs if w not in self.stop_words]
    return segs

  def save_vocabulary(self):
    f = open(self.vocab_file, 'w')
    for k, v in self.vocabulary.items():
      f.write('{0},{1},{2}'.format(k, v, self.token_id_map[k]))
      f.write('\n')
      
    f.close()

  def save_segs(self, path):
    pass

  def load_stop_words(self):
    for line in open(self.stop_file):
      self.stop_words.append(line.strip())

    self.stop_words.append(' ')

    print(len(self.stop_words))
    

  def cut_sent(self, para):
    para = re.sub('([，。！？\?])([^”’])', r"\1\n\2", para)  # 单字符断句符
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)  # 英文省略号
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)  # 中文省略号
    para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    # 如果双引号前有终止符，那么双引号才是句子的终点，把分句符\n放到双引号后，注意前面的几句都小心保留了双引号
    para = para.rstrip()  # 段尾如果有多余的\n就去掉它
    # 很多规则中会考虑分号;，但是这里我把它忽略不计，破折号、英文双引号等同样忽略，需要的再做些简单调整即可。
    return para.split("\n")

if __name__ == '__main__':
  p = CorpusProcess('../data/3b.txt',
                    token_file = '../data/3b_token.txt',
                    vocab_file = '../data/vocab.txt',
                    stop_file = '../data/stop_words.txt')

  p.load_stop_words()
  p.process()
  p.save_vocabulary()

