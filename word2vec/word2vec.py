import tensorflow as tf
from tensorflow import keras
from sample_generator import SampleGenerator

import numpy as np
import pandas as pd
import sys
from vocabulary import Vocabulary
import sys

class Word2Vec():
  def __init__(self):
    self.model = None
    self.emb_layer_name = 'emb'
    self.emb_matrix = None

  def build_model(self, V, K):
    word_input = keras.Input(shape = ())
    context_input = keras.Input(shape = ())

    word_embedding = keras.layers.Embedding(V, K, input_length = 1, name = self.emb_layer_name)
    context_embedding = keras.layers.Embedding(V, K, input_length = 1)
    dot_layer = keras.layers.Dot(axes = 1)

    h = word_embedding(word_input)
    h = tf.reshape(h, shape = (-1, K))
    wo = context_embedding(context_input)
    wo = tf.reshape(wo, shape = (-1, K))
    logits = dot_layer([wo, h])
    output = tf.sigmoid(logits)

    self.model = keras.Model(inputs = [word_input, context_input], outputs = output)
    self.model.summary()

  def extract_embedding_matrix(self):
    word_emb = self.model.get_layer(name = self.emb_layer_name)
    self.emb_matrix = word_emb.trainable_weights[0].numpy()

  def train(self, dataset, epoch = 1):
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)
    self.model.compile(optimizer = opt,
                       loss = 'binary_crossentropy')

    tfbd = tf.keras.callbacks.TensorBoard(log_dir = './logs/word2vec', histogram_freq = 1)

    # no simple way to do evaluate
    self.model.fit(dataset, epochs = epoch, callbacks = [tfbd])

  def save(self, path):
    self.model.save(path)
    
  def load(self, path):
    self.model = keras.models.load_model(path)

  def most_similar(self, word, k):
    # cosine distance : dot(a,b) / norm(a) * norm(b)

    if self.emb_matrix is None :
      self.extract_embedding_matrix()

    m = self.emb_matrix
    
    norms = np.sqrt(np.sum(np.square(m), axis = 1))
    word_norm = norms[word]
    norms = norms*word_norm

    v = m[word]
    dots = np.dot(m, v)
    sims = dots / norms

    sim_df = pd.DataFrame(sims, columns = ['cosine_sim'])
    sim_df = sim_df[sim_df.index != word]
    sim_df.sort_values(by='cosine_sim', ascending = False, inplace = True)
    return sim_df.head(k)


if __name__ == '__main__':

  if len(sys.argv) != 2:
    print('word2vec.py train & sim')
    sys.exit(0)

  vocab = Vocabulary('../data/vocab.txt')
  vocab.load()

  word2vec = Word2Vec()

  what= sys.argv[1]
  if what == 'train':
    generator = SampleGenerator('../data/3b_token.txt.final', vocab)
    dataset = tf.data.Dataset.from_generator(generator, 
                 output_types = ((tf.int32, tf.int32), tf.float32),
                 output_shapes = (((), ()),())
                 #((tf.int32, tf.int32), tf.int32))
                 )

    batch_size = 1024
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size = 5 * batch_size)

    word2vec.build_model(vocab.get_size(), 200)
    word2vec.train(dataset, epoch = 1)
    word2vec.save('../data/word2vec.h5')

  if what == 'sim':
    word2vec.load('../data/word2vec.h5')


  tokens = ['地球','太阳','外星','文明','王子', '史强','你', '惊恐','欧洲','似乎']
  for t in tokens :
    sims = word2vec.most_similar(vocab.token_id(t), 20)
    sims['token'] = [vocab.token(i) for i in sims.index]
    print(t)
    print(sims[['token', 'cosine_sim']].to_string(index = False, header = False))
    print('\n')


