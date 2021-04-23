import random

class Vocabulary():
  def __init__(self, path):
    self.path = path
    self.size = 0
    self.token_id_map = {}
    self.id_token_map = []
    self.token_freq_map = {} 
    self.total_count = 0

  def load(self):
    for line in open(self.path):
      ss = line.split(',')
      token = ss[0]
      token_id = int(ss[2])
      freq = int(ss[1])

      if token not in self.token_id_map.keys():
        self.token_id_map[token] = token_id
        self.token_freq_map[token] = freq
        self.total_count += freq

    
    self.size = len(self.token_id_map)

    self.id_token_map = ['']*self.size
    self.id_freq_map = [0]*self.size
    for token, token_id in self.token_id_map.items():
      self.id_token_map[token_id] = token
      self.id_freq_map[token_id] = self.token_freq_map[token]

  def get_size(self):
    return self.size

  def token(self, token_id):
    return self.id_token_map[token_id]

  def token_id(self, token):
    if token in self.token_id_map.keys():
      return self.token_id_map[token]

    return False

  def make_sample_box(self):
    box = [0]*self.total_count
    box_pos = 0
    for token, token_id in self.token_id_map.items():
      freq = self.token_freq_map[token]
      for i in range(box_pos, box_pos + freq):
        box[i] = token_id
      box_pos += freq

    random.shuffle(box)
    return box

