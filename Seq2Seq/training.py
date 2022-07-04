# -*- litte_bot -*-
# -*- A.Pappa-*-
# -*- coding: utf-8 -*-


from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import re
import pandas as pd
import pickle
import datetime




# longueur max de la phrase
MAX_LENGTH = 120
#tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()

# instantiate a distribution strategy
# strategy = tf.distribute.experimental.TPUStrategy(tpu)
# utilisation du GPU pour tensorflow
strategy = tf.distribute.get_strategy()

# parametres du tf.data.Dataset
BATCH_SIZE = int(64 * strategy.num_replicas_in_sync)
BUFFER_SIZE = 20000

# parametres du model Transformer
NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.1

EPOCHS = 300 #on a baissé à 940 pour le lancer en TPU, on avait 2000, 1500 et 1000 pour le GPU

tf.random.set_seed(1234)
AUTO = tf.data.experimental.AUTOTUNE



# fonc de preprocessing du text, pour enlever les accent et les répétitions
def textPreprocess(input_text):

  def removeAccents(input_text):
      strange='ąćęłńóśżź'
      ascii_replacements='acelnoszz'
      translator=str.maketrans(strange,ascii_replacements)
      return input_text.translate(translator)


  def removeTriplicated(input_text):
      return re.compile(r'(.)\1{2,}', re.IGNORECASE).sub(r'\1', input_text)

  return removeTriplicated(removeAccents(input_text.lower()))



    
# fonc qui construit un tokenizer pour les questions\réponses, il : Tokenise, filtre et pad (égalise la taille)  des phrases
def tokenizeAndFilter(inputs, outputs):
    
    #tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(					#attention changer selon TPU, GPU
    tokenizer = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    inputs + outputs, target_vocab_size=2**13)
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    
    # Definir le token de debut et de fin pour indiquer le debut et la fin d'une phrase
    START_TOKEN, END_TOKEN = [tokenizer.vocab_size], [tokenizer.vocab_size + 1]
    # la taille du vocabulaire (+ le token de debut et de fin)
    VOCAB_SIZE = tokenizer.vocab_size + 2
    
    tokenized_inputs, tokenized_outputs = [], []
      
    for (sentence1, sentence2) in zip(inputs, outputs):
        # tokenise la phrase
        sentence1 = START_TOKEN + tokenizer.encode(sentence1) + END_TOKEN
        sentence2 = START_TOKEN + tokenizer.encode(sentence2) + END_TOKEN
        # check la taille max de la phrase tokenisée
        if len(sentence1) <= MAX_LENGTH and len(sentence2) <= MAX_LENGTH:
          tokenized_inputs.append(sentence1)
          tokenized_outputs.append(sentence2)
  
  # pad les phrases tokenisées
    tokenized_inputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_inputs, maxlen=MAX_LENGTH, padding='post')
    tokenized_outputs = tf.keras.preprocessing.sequence.pad_sequences(
      tokenized_outputs, maxlen=MAX_LENGTH, padding='post')
  
    return tokenized_inputs, tokenized_outputs, VOCAB_SIZE



# création d'un tensorflow dataset pour simplifier le training plus tard 
def createDataset(inputs, outputs):
    dataset = tf.data.Dataset.from_tensor_slices((
        {
            'inputs': inputs,
            'dec_inputs': outputs[:, :-1]
        },
        {
            'outputs': outputs[:, 1:]
        },
    ))
    return dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)



# fonc qui calcule les poids de l'attention
def scaledDotProductAttention(query, key, value, mask):
  matmul_qk = tf.matmul(query, key, transpose_b=True)

  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)

  # ajout du masque pour annuler le padding de tokens
  if mask is not None:
    logits += (mask * -1e9)

  # la softmax est normalisée au dérnier axe
  attention_weights = tf.nn.softmax(logits, axis=-1)

  output = tf.matmul(attention_weights, value)

  return output




class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)

    self.dense = tf.keras.layers.Dense(units=d_model)

  def get_config(self):
      cfg = super().get_config()
      return cfg    

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(
        inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs[
        'value'], inputs['mask']
    batch_size = tf.shape(query)[0]

    # couche linéaire
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)

    # spliter les tetes (splite heads)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)

    # produit scalaire mis à l'échelle de l'attention
    scaled_attention = scaledDotProductAttention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

    # concaténation des tetes
    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))

    # couche linéaire finale
    outputs = self.dense(concat_attention)

    return outputs




def createPaddingMask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]




def createLookAheadMask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = createPaddingMask(x)
  return tf.maximum(look_ahead_mask, padding_mask)




class PositionalEncoding(tf.keras.layers.Layer):

  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model)
    # appliquer le sin afin de rendre les indices du tableau pairs
    sines = tf.math.sin(angle_rads[:, 0::2])
    # applquer le cos afin de rendre les indices du tableau impairs
    cosines = tf.math.cos(angle_rads[:, 1::2])

    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]




def encoderLayer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })
  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(inputs + attention)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention + outputs)

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)




def encoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
  for i in range(int(num_layers)):
    outputs = encoderLayer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, padding_mask], outputs=outputs, name=name)




def decoderLayer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')

  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })
  attention1 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention1 + inputs)

  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })
  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(
      epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)




def decoder(vocab_size,
            num_layers,
            units,
            d_model,
            num_heads,
            dropout,
            name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(
      shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)

  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(int(num_layers)):
    outputs = decoderLayer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(
      inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask],
      outputs=outputs,
      name=name)




def transformer(vocab_size,
                num_layers,
                units,
                d_model,
                num_heads,
                dropout,
                name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(
      createPaddingMask, output_shape=(1, 1, None),
      name='enc_padding_mask')(inputs)
  # masquer les futures tokens des inputs du décodeur au 1er bloque d'attention
  look_ahead_mask = tf.keras.layers.Lambda(
      createLookAheadMask,
      output_shape=(1, None, None),
      name='look_ahead_mask')(dec_inputs)
  # masque les outputs de l'encodeur du 2ème bloque d'attention
  dec_padding_mask = tf.keras.layers.Lambda(
      createPaddingMask, output_shape=(1, 1, None),
      name='dec_padding_mask')(inputs)

  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)

  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)




def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction='none')(y_true, y_pred)

  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)

  return tf.reduce_mean(loss)




def accuracy(y_true, y_pred):
  # s'assure que les labels ont la forme suivante (batch_size, MAX_LENGTH - 1)
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)




class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(tf.cast(step, tf.float32))
    arg2 = step * (self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)




def main():
    print("Pré-processing du texte")
    df = pd.read_csv('./Data/68k.csv')
    df['Input'] = df['Input'].apply(lambda x: textPreprocess(str(x)))
    df['Target'] = df['Target'].apply(lambda x: textPreprocess(str(x)))
    questions, answers = df['Input'].tolist(), df['Target'].tolist()

    
    print("\Constructiion du tokenizer")
    questions, answers, VOCAB_SIZE = tokenizeAndFilter(questions, answers)

    print('Taille du vocabulaire: {}'.format(VOCAB_SIZE))
    print("Nombre d'échantillon: {}".format(len(questions)))
    
    print("\nCreation de la dataset")
    dataset = createDataset(questions, answers)
    
    # nettoyage du backend tf
    tf.keras.backend.clear_session()
    learning_rate = CustomSchedule(D_MODEL)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    
    print("\nInitialisation et compilation du modèle dans la stratégie")
    
    with strategy.scope():
      model = transformer(
          vocab_size=VOCAB_SIZE,
          num_layers=NUM_LAYERS,
          units=UNITS,
          d_model=D_MODEL,
          num_heads=NUM_HEADS,
          dropout=DROPOUT)    
      model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
    print(model.summary())
    print('\n')
    print("Training du modèle")
    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)

    model.fit(dataset, epochs=EPOCHS)
    model.save_weights('./saved_weights68k-940ep.h5')

    print("\nPoids du modèle sauvgardés !")
    


if __name__ == '__main__':
    main()
