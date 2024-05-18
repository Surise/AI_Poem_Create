import math
import re
import numpy as np
import tensorflow as tf
from collections import Counter
from surise.surise import *
DATA_PATH = './data/poems.txt'
MAX_LEN = 64
DISALLOWED_WORDS = ['（', '）', '(', ')', '__', '《', '》', '【', '】', '[', ']']
poetry = []
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    lines = f.readlines()  
for line in lines:
    fields = re.split(r"[:：]", line)
    if len(fields) != 2:
        continue
    content = fields[1]
    if len(content) > MAX_LEN - 2:
        continue
    if any(word in content for word in DISALLOWED_WORDS):
        continue
    poetry.append(content.replace('\n', ''))
MIN_WORD_FREQUENCY = 8
counter = Counter()
for line in poetry:
    counter.update(line)
tokens = [token for token, count in counter.items() if count >= MIN_WORD_FREQUENCY]
tokens = ["[PAD]", "[NONE]", "[START]", "[END]"] + tokens
tokenizer = Tokenizer(tokens)
word_idx = {}
idx_word = {}
for idx, word in enumerate(tokens):
    word_idx[word] = idx
    idx_word[idx] = word
BATCH_SIZE = 32
dataset = PoetryDataSet(poetry, tokenizer, BATCH_SIZE)
#model = tf.keras.models.load_model('./model/poemsagain9.keras')

model = tf.keras.Sequential([tf.keras.layers.Embedding(input_dim=tokenizer.dict_size, output_dim=150),
    # 第一个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    # 第二个LSTM层
    tf.keras.layers.LSTM(150, dropout=0.5, return_sequences=True),
    tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(tokenizer.dict_size, activation='softmax')),])
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.sparse_categorical_crossentropy)

for i in range(10):
    model.fit(dataset.generator(), steps_per_epoch=dataset.steps,verbose=2,epochs=1)
    model.save(f'./model/poemsmodel{i}.keras')
