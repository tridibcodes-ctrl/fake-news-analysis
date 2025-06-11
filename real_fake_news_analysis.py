import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from wordcloud import WordCloud

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPool1D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import preprocess_kgptalkie as ps
import gensim
import pickle

fake= pd.read_csv("https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/refs/heads/main/data/Fake.csv")
fake.head()

fake.columns

fake['subject'].value_counts()

plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=fake)

fake_text = ' '.join(fake['text'].tolist())

wordcloud = WordCloud(width=1920,height=1080).generate(fake_text)
fig= plt.figure(figsize=(10,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

real = pd.read_csv("https://raw.githubusercontent.com/laxmimerit/fake-real-news-dataset/refs/heads/main/data/True.csv")
real.head()

real.columns

real['subject'].value_counts
plt.figure(figsize=(10,6))
sns.countplot(x='subject',data=real)

real_text = ' '.join(real['text'].tolist())
wordcloud = WordCloud(width=1920,height=1080).generate(real_text)
fig= plt.figure(figsize=(10,20))
plt.imshow(wordcloud)
plt.axis('off')
plt.tight_layout(pad=0)
plt.show()

unknown_publisher = []
for index, row in enumerate(real.text.values):
  try:
    record=row.split('-',maxsplit=1)
    record[1]
    assert(len(record[0])<120)
  except:
    unknown_publisher.append(index)

len(unknown_publisher)

real.iloc[unknown_publisher].text

publisher=[]
temp_text=[]
for index, row in enumerate(real.text.values):
  if index in unknown_publisher:
    temp_text.append(row)
    publisher.append('unknown')
    continue
  else:
    record=row.split('-',maxsplit=1)
    publisher.append(record[0].strip())
    temp_text.append(record[1].strip())

real['publisher']=publisher
real['text']=temp_text

real.head()

real.shape

empty_fake_index= [index for index, text in enumerate(fake.text.tolist()) if str(text).strip()==""]

fake.iloc[empty_fake_index]

# Ensure 'title' column exists in both datasets before concatenation
if 'title' in real.columns and 'title' in fake.columns:
    real['text'] = real['title'] + " " + real['text']
    fake['text'] = fake['title'] + " " + fake['text']
else:
    print("Error: 'title' column is missing in one of the datasets.")
    exit()

real['text'] = real['text'].apply(lambda x: str(x).lower())
fake['text'] = fake['text'].apply(lambda x: str(x).lower())

real['class']=1
fake['class']=0

real= real[['text','class']]
fake= fake[['text','class']]

data = pd.concat([real, fake], ignore_index=True)

data.sample(5)

data['text'].apply(lambda x: ps.remove_special_chars(x))

data.head()

y=data['class'].values

X=[d.split() for d in data['text'].tolist()]

type(X[0])

print(X[0])

DIM=100
w2v_model= gensim.models.Word2Vec(sentences=X, vector_size=DIM, window=5, min_count=1)

len(w2v_model.wv.key_to_index)

w2v_model.wv['love']

w2v_model.wv.most_similar('india')

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)

X = tokenizer.texts_to_sequences(X)

plt.hist([len(x) for x in X], bins=700)
plt.show()

nos = np.array([len(x) for x in X])
maxlen = 1000
X = pad_sequences(X, maxlen=maxlen)

len(X[101])

vocab_size = len(tokenizer.word_index) + 1
vocab = tokenizer.word_index

def get_weight_matrix(model):
    weight_matrix = np.zeros((vocab_size, DIM))

    for word, i in vocab.items():
        weight_matrix[i] = model.wv[word]
    return weight_matrix

embedding_vectors = get_weight_matrix(w2v_model)

embedding_vectors.shape

model = Sequential()
model.add(Embedding(vocab_size, output_dim=DIM, weights=[embedding_vectors], input_length=maxlen, trainable=False))
model.add(LSTM(units=128))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

model.build(input_shape=(None, maxlen))
model.summary()

X_train, X_test, y_train, y_test = train_test_split(X, y)

model.fit(X_train, y_train, validation_split=0.3, epochs=6)

y_pred= (model.predict(X_test)>=0.5).astype(int)

accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred))

x=['nasa confirms earth will have 6 days of total darkness in june 2025']
x=tokenizer.texts_to_sequences(x)
x=pad_sequences(x, maxlen=maxlen)

(model.predict(x)>=0.5).astype(int)

x=['india successfully lands chandrayaan-3 on the moon\'s south pole']
x=tokenizer.texts_to_sequences(x)
x=pad_sequences(x, maxlen=maxlen)
(model.predict(x)>=0.5).astype(int)

# prompt: I want to make an API out of this model, what should I do?

# Save the model
model.save('fake_news_model.h5')  # Save the model in HDF5 format

# Save the tokenizer as a .pkl file
with open('tokenizer.pkl', 'wb') as handle:  # Ensure consistent naming
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer have been saved successfully!")

