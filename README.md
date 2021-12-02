# Bi-Directional LSTM RNN - Domain Prediction in Healthcare Text

## Overview
This project uses a bi-directional RNN with LSTM to predict domain relevance of healthcare related comments from social media websites. Accuracy of c. 99% was achieved on test data indicating a very high ability for the RNN to learn the underlying relationships.

## Data
The dataset is a tagged 0 = irrelevant, 1 = low sentiment, 5 = high sentiment. There are two domains 'effective treatment' and 'attention to physical & environmental needs'. In this case we are interested only in domain relevance, so we map all non-zero integers to 1 (becomes a binary classification problem).

## Preprocessing
Basic preprocessing steps were implied, which includes lowercasing, and removal of '#' and '@'. The dataset was shuffled, and split into train/val/test: 80/10/10. The maximum sequence length was capped at 100 (90% of sequences less than this). The vocabulary size was not capped. The train/val/test datasets were batched to size 64. 

## Bidirectional RNN
A bi-directional RNN was created as shown below. Embeddings were created of size (vocab, max_seqlen) which are fully trainable. The embedding length was chosen as the same length as the sequence size, but any number could have been chosen. It is possible to insert pre-trained embeddings but given the high level of performance (and healthcare specific words) it was deemed unnecessary. The output from the RNN is fed into a densely connected layer with 100 units and relu activation. This is fed into a final sigmoid output layer that maps the probability of being irrelevant / relevant between 0 and 1.
```
class TreatmentClassifier(tf.keras.Model):
    def __init__(self, vocab_size, max_seqlen, **kwargs):
        super(TreatmentClassifier, self).__init__(**kwargs)
        self.embedding = tf.keras.layers.Embedding(vocab_size, max_seqlen)
        self.bilstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_seqlen))
        self.dense = tf.keras.layers.Dense(100, activation = 'relu')
        self.out = tf.keras.layers.Dense(1, activation = 'sigmoid')

    def call(self, x):
        x = self.embedding(x)
        x = self.bilstm(x)
        x = self.dense(x)
        x = self.out(x)
        return x
 ```

## Results
**Accuracy was c. 99% for both 'effective treatment' and 'attention to physical & environmental needs'. Some of the false positives and false negatives appear to have been misclassified. Training was smooth and showed consistent improvements in performance.
