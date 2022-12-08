#!/usr/bin/env python
# coding: utf-8

# # Sentiment Analysis and Topic Identification
# ##Imports

# In[ ]:


pip install textblob


# In[2]:


from sklearn import svm, datasets
from sklearn import preprocessing 

import pandas as pd
#Metrics libraries
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc

#Visualization libraries
import matplotlib.pyplot as plt 
from matplotlib import rcParams
import seaborn as sns
from textblob import TextBlob
from plotly import tools
import plotly.graph_objs as go
from plotly.offline import iplot
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import re


# In[4]:


pip install transformers


# In[35]:


data_df = pd.read_json('xaa', lines=True)


# In[ ]:


data_df = pd.read_json('/content/xaa', lines=True)


# In[ ]:


get_ipython().system('pip install --upgrade openai wandb')


# ## Sentiment Analysis
# Processing the ratings on Amazon, and associating a positive or negative sentiment based on the review text. 
# Fasttext and Glove are used to obtain vector representations for words.
# Numerous models (SVM, Random Forest, Logistic Regression, LSTM) are used, in addition to a transformer (BERT). A deep neural network is to be implemented to recognize underlying relationships between the vector representation. 
# In section 2, two techniques, frequency-based and GTP-3 based topic modelling is used on positive and negative reviews to capture the source of satisfaction/dissatisfaction (Mailing issue, customer support, quality of product...)

# In[36]:


#Creating a copy
process_reviews=data_df.copy()

#Checking for null values
process_reviews.isnull().sum()


# # Let us remove the Nan values

# In[37]:


data_df.size


# In[38]:


data_df['overall'].value_counts().plot(kind = 'pie')
plt.title('Distribution of ratings pre-processing')


# In[39]:


data_df['reviewText'].isnull().sum()


# In[40]:


data_df = data_df.dropna(subset=['reviewText'])
data_df.size


# In[41]:


data_df = data_df[data_df.overall != int(3)]
data_df.size


# In[42]:


data_df['reviewText'].isnull().sum()


# In[43]:


# Distribution of ratings
data_df.head()


# We only care about overall, reviewText and summary for the time being. 

# In[20]:


data_df.describe()


# In[21]:


data_df['reviewText'][0]


# In[14]:


data_df['overall'].value_counts().plot(kind = 'pie')


# In[15]:


# A quick look at the type of text we are working with. 


# In[16]:


# Plot distribution of review length
#final_words = [word for word in s.split()]

#c = len(final_words)
review_texts_size = data_df['reviewText'].map(lambda x: len([word for word in x.split()]))
plt.figure(figsize=(22,12))
review_texts_size.loc[review_texts_size < 800].hist()
plt.title("A look at the lengths of various reviews")
plt.xlabel('Number of words')
plt.ylabel('Number of Reviews')


# In[17]:


max(review_texts_size)


# ## Let us capture the sentiment and topic in the first 200 words (4400 is somewhat overkill). 

# In[44]:


import numpy as np
data_df['sentiment'] = np.select(
        condlist = [data_df.overall > 3, data_df.overall < 3],
        choicelist = [1, 0])


# Positive sentiment: assigned 1 for ratings >3
# Negative sentiment: assigned 0 for ratings <3

# In[45]:


# Assign 1 if positive, 0 if negative sentiment 
data_df['sentiment']


# In[46]:


data_df['sentiment'].value_counts()


# In[25]:


data_df['sentiment'].value_counts().plot(autopct='%1.1f%%', kind = 'pie', ylabel = 'Total Reviews: 1,800,331')
plt.title('Sentiment Distribution Post-processing')


# ## Topic Classification

# ## Frequency Based
# 

# In[ ]:


# Some pre-processing: we join summary and review texts, remove punctuation, stopwords etc and tokenize. 


# In[47]:


review_aggregated = data_df['summary'].fillna(' ') + " " + data_df['reviewText'] 
sentiments_agg = data_df['sentiment']


# In[48]:


sentiments_agg = data_df['sentiment']


# In[49]:


data_df['summary'] = review_aggregated


# In[66]:


data_df['summary'] 


# In[51]:


data_df =  data_df[['summary', 'sentiment']].copy()


# In[78]:


data_df


# In[67]:


data_df['summary'] = review_aggregated.str.replace("[^a-zA-Z#]", " ")


# In[27]:


#review = data_df['reviewText'].str.replace("[^a-zA-Z#]", " ")


# In[73]:


import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


# In[ ]:


def remove_numbers(text):
    return re.sub(r'\b\d+\b', '', text)


# In[ ]:


review_list = list(review)


# In[ ]:


# list_final = []
# counter = 0
# for list_i in review_list: 
#   if len(list_i)>= 1:
#     list_final.append([])
#     for word in list_i: 
#       if word not in stop_words:
#         list_final[counter].append(word)
#   else:
#         list_final.append(list)
#   counter+=1


# In[29]:


data_df['review_agg'] = review


# In[30]:


review_3 = data_df['review_agg'].str.replace('\d+', '')


# In[31]:


data_df['review_agg'] = review_3


# In[32]:


data_df['review_agg']


# In[69]:


data_df['preprocessed'] = data_df['summary'] .apply(lambda x: ' '.join([word.lower() for word in x.split()]))


# In[74]:


data_df['preprocessed'] = data_df['preprocessed'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# In[77]:


data_df['preprocessed'][0]


# In[36]:


import csv
col1 = data_df['preprocessed'].values.tolist()
col2 = data_df['sentiment'].values.tolist()
with open('cleaned_full_sentences_reviews.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(col1, col2))


# In[ ]:


list_to_octis = data_df['preprocessed'].tolist()


# In[76]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.tokenize import word_tokenize


# In[ ]:


## To save dataframe
#data_df.to_csv('out.zip', index=False)


# ## LDA

# In[ ]:


import nltk
nltk.download('punkt')


# In[ ]:


import csv
with open('/content/some_file_(1).csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


# In[ ]:


negative = [i[0] for i in data if i[1] == '0']


# In[ ]:


len(negative)


# In[ ]:


positive = [i[0] for i in data if i[1] == '1']


# In[ ]:


data_df['preprocessed']


# In[ ]:


data_df[['preprocessed', 'sentiment']]


# In[ ]:


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def lemmatize_word(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(sentence):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
output_lemmatized = []


# In[ ]:


import csv
col1 = data_df['preprocessed'].values.tolist()
col2 = data_df['sentiment'].values.tolist()
with open('some_file_.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(col1, col2))


# In[ ]:


for sentence in data_df['preprocessed']:
  output_lemmatized.append(' '.join(lemmatize_word(sentence)))


# In[ ]:


with open('some_file_2_2.csv', 'w') as f2:
    writer = csv.writer(f2)
    writer.writerows(zip(output_lemmatized, col2))


# In[ ]:


output_lemmatized[:5]


# In[ ]:


from gensim import corpora, models
from gensim.models import LdaModel


# In[ ]:


data_df['preprocessed'] = output_lemmatized


# In[ ]:


positive = data_df['preprocessed'].where(data_df['sentiment']==1).dropna()
negative = data_df['preprocessed'].where(data_df['sentiment']==0).dropna()


# In[ ]:


import csv

with open('some_file_.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

output_lematized_loaded = data


# In[ ]:


sentences_clean = [i[0] for i in output_lematized_loaded]
sentiments_clean = [i[1] for i in output_lematized_loaded]


# In[ ]:


import pandas as pd 
dic = {'sentences': sentences_clean, 'sentiment': sentiments_clean}
data_df_clean = pd.DataFrame(dic)


# In[ ]:


negative = data_df_clean[data_df_clean['sentiment'] == '0']


# In[ ]:


negative


# In[ ]:


from nltk.tokenize import word_tokenize

negative_tokenized = negative['sentences'].apply(lambda x: word_tokenize(x))


# In[ ]:


data_df_clean['sentences'] = negative_tokenized


# In[ ]:


negative_listtt = negative['sentences']


# In[ ]:


negative_listttt = str(negative_listtt)


# In[ ]:


negative_listtt


# ## Clustering to find number of topics

# In[ ]:


# For negative


# In[ ]:


import ast
for x in range(len(negative)): 
  negative[x] = ast.literal_eval(negative[x])


# In[ ]:


from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


lmtzr = WordNetLemmatizer()
for i in range(len(negative)):
    new_list = []
    for token, tag in pos_tag(negative[i]):
        lemma = lmtzr.lemmatize(token, tag_map[tag[0]])
        new_list.append(lemma)
    negative[i] = new_list
    if i < 10:
      print(negative[i])


# In[ ]:


jupyter --NotebookApp.iopub_data_rate_limit=1.0e10


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
X = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.001, max_df=0.9).fit_transform(negative_listt)
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(X)


# In[ ]:


X.shape


# In[ ]:


tfidf.shape


# In[ ]:


from sklearn.cluster import KMeans
SSE = []
for k_num in range(5, 55, 5):
    kmeans = KMeans(n_clusters = k_num, max_iter = 100)
    kmeans.fit(tfidf)
    SSE.append(kmeans.inertia_)


# In[ ]:


plt.plot(range(5, 55, 5), SSE)
plt.xlabel("Number of topics")
plt.ylabel("SSE")
plt.title("Elbow method for Num of Topics using KMeans Clustering")


# In[ ]:


from gensim import corpora
dictionary = corpora.Dictionary(negative)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
dictionary.filter_extremes(no_below=0, no_above=0.80)


# In[ ]:


from gensim.models import LdaModel
doc_term_matrix = [dictionary.doc2bow(rev) for rev in negative]
LDA_model = LdaModel(doc_term_matrix, num_topics=35, id2word=dictionary, passes=5)


# In[ ]:


dict_of_topics = {}
for idx, topic in LDA_model.print_topics(-1):
    #print('Topic: {} \nWords: {}'.format(idx, topic))
    dict_of_topics[idx] = topic


# In[ ]:


LDA_frequency = []
for key in dict_of_topics:
  values = dict_of_topics[key] 
  top3 = re.findall('"(.+?)"', values)
  print(top3[:3])
  LDA_frequency.append(top3[:3])


# In[ ]:


dict_of_topics_top3 = [['door', 'wood', 'shelf'],
['color', 'sound', 'light'],
['work', 'well', 'stop'],
['small', 'size', 'fit'],
['pan', 'stick', 'use'],
['star', 'one', 'two'],
['one', 'year', 'old'],
['unit', 'review', 'product'],
['use', 'get', 'food'],
['cooker', 'paper', 'cook'],
['useless', 'cup', 'tray'],
['smell', 'burn', 'like'],
['wash', 'rust', 'use'],
['apart', 'knife', 'cut'],
['money', 'waste', 'worth'],
['coffee', 'water', 'clock'],
['rip', 'head', 'weigh'],
['quality', 'product', 'make'],
['filter', 'bag', 'image'],
['table', 'can', 'hook'],
['vacuum', 'shoe', 'floor'],
['iron', 'board', 'heavy'],
['piece', 'bad', 'junk'],
['like', 'look', 'get'],
['machine', 'clean', 'get'],
['oven', 'rod', 'cube'],
['fan', 'air', 'room'],
['glass', 'heat', 'panasonic'],
['lid', 'salt', 'top'],
['customer', 'service', 'call'],
['ice', 'candle', 'blender'],
['together', 'screw', 'hole'],
['return', 'box', 'item'],
['month', 'use', 'work'],
['break', 'plastic', 'cheap']]


# In[ ]:


dict_of_topics_top3


# In[ ]:


for key in dict_of_topics:
  values = dict_of_topics[key] 
  top3 = re.findall('"(.+?)"', values)
  print(top3)


# In[ ]:


with open('neative_topic_dictionary.csv','w') as f:
    w = csv.writer(f)
    w.writerows(dict_of_topics.items())


# In[ ]:


dict_of_topics = {}
for idx, topic in LDA_model.print_topics(-1):
    #print('Topic: {} \nWords: {}'.format(idx, topic))
    dict_of_topics[idx] = topic


# In[ ]:


Topics = ['customer service', 'follow', 'difficult set', 'useless', 'damage', 'food', 'smell', 'plastic', 'rust', 'split', 'color', 'noise', 'waste', 'problems', 'dust', 'broken', 'coffee', 'size', 'review', 'make', 'design', 'electric', 'bad product', 'stop' 'print', 'time', 'heat', 'normal', 'quality', 'dent', 'flimsy', 'last', 'steel', 'keep']


# In[ ]:


len(Topics)


# In[ ]:


dict_of_topics


# In[ ]:


dict_of_topics = {}
for idx, topic in LDA_model.print_topics(-1):
    #print('Topic: {} \nWords: {}'.format(idx, topic))
    dict_of_topics[idx] = topic


# In[ ]:


topics_list_3 = []
for x in dict_of_topics:
  topics_list_3.append(re.findall('"(.+?)"', dict_of_topics[x]))


# In[ ]:


topic_list_4 = []
i = 0
for x in topics_list_3: 
  topic_list_4.append([])
  j = 0
  for word in x: 
    if j<3:
      if 'review' not in word and 'star' not in word and 'one' not in word:
        topic_list_4[i].append(word)
        j+=1
  j+=1
  i+=1


# In[ ]:


topic_list_4


# In[ ]:


max = 0
for (i, score) in LDA_model[doc_term_matrix[0]]:
  if score>max:
    max = score
    topic_l = i
topic = topics_list[topic_l]
print(topic)


# In[ ]:


LDA_model[doc_term_matrix[0]]


# In[ ]:


for index, score in sorted(LDA_model[doc_term_matrix[0]]):
    print("Topic: {}".format(LDA_model.print_topic(index, 1)))


# In[ ]:


dict_of_topics_2 = {}
counter = 0
for topic_name in topic: 
    dict_of_topics_2[topic_name] = dict_of_topics[counter]
    counter +=1


# ## Same for Positive Reviews

# In[ ]:


positive = data_df_clean[data_df_clean['sentiment'] == '1']


# In[ ]:


positive_listt = positive['sentences'].tolist()


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import CountVectorizer
Y = CountVectorizer(tokenizer=lambda doc: doc, lowercase=False, min_df=0.01, max_df=0.8).fit_transform(positive_listt)
transformer = TfidfTransformer(smooth_idf=False)
tfidf = transformer.fit_transform(Y)


# In[ ]:


from sklearn.cluster import KMeans
SSE = []
for k_num in range(15, 55, 5):
    kmeans = KMeans(n_clusters = k_num, max_iter = 100)
    kmeans.fit(tfidf)
    print(k_num, " the SSE is ", kmeans.inertia_)
    SSE.append(kmeans.inertia_)


# In[ ]:


plt.plot(range(15, 55, 5), SSE)
plt.xlabel("Number of topics")
plt.ylabel("SSE")
plt.title("Elbow method for Num of Topics using KMeans Clustering")


# In[ ]:


## We find 30 topics


# In[ ]:


data_df['preprocessed']


# In[ ]:


with open('preprocessed_full_data', 'w') as f:
      
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(zip(data_df['preprocessed'], data_df['sentiment']))


# In[ ]:


data_df['sentiment']


# In[ ]:


copy_df = data_df_clean[['sentences', 'sentiment']]


# In[ ]:


negative_BERTopic = data_df[data_df['sentiment'] == 0]


# In[ ]:


list_nn = negative_BERTopic['preprocessed'].tolist()


# In[ ]:


list_nn_2 = [[i] for i in list_nn]


# In[ ]:


with open('list_nn_full.csv','w') as result_file:
    wr = csv.writer(result_file, dialect='excel')
    wr.writerows(list_nn_2)


# ## BERTopic comparison

# In[173]:


import csv

with open('list_nn_full.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

output_lematized_loaded = data


# In[174]:


output_lematized_loaded = [x for [x] in output_lematized_loaded]


# In[181]:


output_lematized_loaded_shrunk = output_lematized_loaded[:50000]


# In[178]:


#data_df['preprocessed'] = data_df['summary'].fillna(' ') + " " + data_df['reviewText'] 
sentiments_agg = data_df['sentiment']
review_3 = review_aggregated.str.replace('\d+', '')


# In[177]:


data_df


# In[ ]:


data_df['preprocessed'] = review_3


# In[ ]:


data_df['preprocessed']


# In[ ]:


data_df['preprocessed'] = data_df['preprocessed'].dropna(inplace=True)


# In[ ]:


data_df['preprocessed'] = data_df['preprocessed'].str.replace('\d+', '')


# In[ ]:


data_df['preprocessed']


# In[ ]:


data_df['preprocessed'] = data_df['preprocessed'].apply(lambda x: ' '.join([word.lower() for word in x.split()]))
data_df['preprocessed'] = data_df['preprocessed'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))


# In[ ]:


data_df['preprocessed'] = data_df['preprocessed'].where(data_df['sentiment']==0).dropna()


# In[ ]:


negative_BERTOPIC = data_df['preprocessed'].tolist()


# In[ ]:


negative_BERTOPIC_3 = negative_BERTOPIC
negative_BERTOPIC = negative_BERTOPIC_2


# In[ ]:


import string
negative_BERTOPIC_2 = ["".join( j for j in i if j not in string.punctuation) for i in  negative_BERTOPIC]


# In[ ]:


sentences_clean = [i[0] for i in output_lematized_loaded]


# In[ ]:


import spacy


# In[ ]:


for x in sentences_clean:
   print(x.tolist())


# In[ ]:


sentences_clean


# In[179]:


from bertopic import BERTopic
#output_lemmatized


# In[ ]:


len(output_lematized_loaded)


# In[183]:


topic_model = BERTopic(language="english", calculate_probabilities=True, verbose=True)


# In[184]:


topics, probs = topic_model.fit_transform(output_lematized_loaded_shrunk)


# In[ ]:


freq = topic_model.get_topic_info(); 
freq.head(20)


# In[ ]:


topic_model.get_topics()


# In[ ]:


topic_model.save("negative_BERTOPIC_model")


# In[ ]:


topic_model.generate_topic_labels()


# In[ ]:


topic_model.visualize_topics()


# In[ ]:


freq = topic_model.get_topic_info(); 
freq.head(20)


# In[ ]:


topic_model.visualize_hierarchy()


# In[ ]:


len(list_nn)


# In[ ]:


doc_term_matrix[10]


# In[ ]:





# # GTP-3

# In[ ]:


review_2 = []
for x in review: 
  review_2.append(word_tokenize(x))


# In[ ]:


get_ipython().run_line_magic('env', 'OPENAI_API_KEY=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')
import os
import openai
import wandb


openai.api_key = os.getenv("OPENAI_API_KEY")


# In[ ]:


run = wandb.init(project='GPT-3 in Python')
prediction_table = wandb.Table(columns=["prompt", "completion"])


# In[ ]:


review_2[1]


# In[ ]:


result_AI_open = []

for x in range(len(dict_of_topics)): 
  gpt_prompt = "Find a topic for this text:\n\n" + str(dict_of_topics[x])
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  print(response)
  result_AI_open.append(response)


# In[ ]:


import time
result_AI_open_top3 = []

for x in range(len(dict_of_topics_top3)): 
  gpt_prompt = "Find a 3 word topic for this text:\n\n" + str(dict_of_topics_top3[x])
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  print(response)
  result_AI_open_top3.append(response)
  time.sleep(3)


# In[ ]:


text_output_openAI = []
i = 0
for x in result_AI_open_top3:
  #text_output_openAI.append([])
  text_output_openAI.append(x["choices"][0]["text"].split("\n\n",1)[1])
  i+=1


# In[ ]:


text_output_openAI


# In[ ]:


import time
result_AI_open_2 = []

for x in range(len(dict_of_topics)): 
  gpt_prompt = "Find a one word topic for this text:\n\n" + str(dict_of_topics[x])
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  print(response)
  result_AI_open_2.append(response)
  time.sleep(3)


# In[ ]:


import time
result_AI_open_3 = []

for x in range(len(topic_list_4)): 
  gpt_prompt = "Find a hypernym for this text:\n\n" + str(topic_list_4[x])
  response = openai.Completion.create(
    engine="text-davinci-002",
    prompt=gpt_prompt,
    temperature=0.5,
    max_tokens=256,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
  )
  print(response)
  result_AI_open_3.append(response)
  time.sleep(3)


# # Now let us apply LDA
# 
# 

# In[ ]:


from transformers import pipeline
from nltk.tokenize import word_tokenize
#NLTK word tokenize
# def word_tokenize_wrapper(text):
#  return word_tokenize(text)
# negative_2 = negative.apply(word_tokenize_wrapper)
# negative_2.head()

classifier = pipeline('sentiment-analysis')


# In[ ]:


with open('some_file_.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)

output_lematized_loaded = data


# # Trying the classifier from the transformer pipeline
# It is very slow as it ranks each word as positive/negative then classifies based on majority

# In[ ]:


from ast import literal_eval
for i in range(len(output_lematized_loaded)):
    output_lematized_loaded[i] = literal_eval(output_lematized_loaded[i][0])


# In[ ]:


output_lematized_loaded


# In[ ]:


[i for [i[0:]] in output_lematized_loaded]


# In[ ]:


negative_c = 0
positive_c = 0
sentiments = []
for i in range(len(output_lematized_loaded)): 
    primal_result = classifier(output_lematized_loaded[i])
    for x in primal_result:
        if x['label'] == 'POSITIVE':
            positive_c +=1
        else: 
            negative_c +=1
    if positive_c>negative_c:
        result = 1
    else:
        result = 0
    sentiments.append(result)


# In[ ]:


sentiments


# In[ ]:


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def lemmatize_word(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(sentence):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
output = []
for sentence in review_2:
  output.append(' '.join(lemmatize_word(sentence)))


# In[ ]:


# LDA 
dictionary = corpora.Dictionary(output)
doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]
LDA = gensim.models.ldamodel.LdaModel


# In[ ]:


review_2[1]


# In[ ]:


review_2[1]


# In[ ]:


#review = review.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))


# In[ ]:


review_2


# In[ ]:


from nltk.probability import FreqDist


# In[ ]:


result_2 = open("review_tokenized.txt", "a")
result_2.write(str(review))


# In[ ]:


all_words = ' '.join([text for text in review])


# In[ ]:





# In[ ]:


#all_words = ' '.join([text for text in review])
all_words = all_words.split()
fdist = FreqDist(all_words)


# In[ ]:


to_oder = fdist.most_common(30)


# In[ ]:


df = pd.DataFrame(to_oder, columns=['word', 'frequency'])
df.plot(kind='bar', x='word', figsize=(12,12))


# In[ ]:


data_df['reviewText'].size


# In[ ]:


positive = data_df.loc[data_df['sentiment'] == 1, 'reviewText']


# In[ ]:


negative = data_df.loc[data_df['sentiment'] == 0, 'reviewText']


# In[ ]:


positive.size


# In[ ]:


positive[:2]


# In[ ]:


positive = positive.reset_index(drop = True).str.replace("[^a-zA-Z#]", " ")


# In[ ]:


positive = positive.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))


# In[ ]:


negative = negative.reset_index(drop = True).str.replace("[^a-zA-Z#]", " ")
negative = negative.apply(lambda x: ' '.join([w for w in x.split() if w not in stop_words]))


# In[ ]:


negative = negative.apply(lambda x: [w.lower() for w in x.split()])


# In[ ]:





# In[ ]:


get_ipython().system('pip install --upgrade spacy')


# In[ ]:





# In[ ]:


def lemmatization(texts, tags=['NOUN', 'ADJ']): # filter noun and adjective
       output = []
       for sent in texts:
             doc = nlp(" ".join(sent)) 
             output.append([token.lemma_ for token in doc if token.pos_ in tags])
       return output


# In[ ]:


tokenized_reviews = negative.apply(lambda x: lemmatization(x))


# In[ ]:





# In[ ]:


from nltk.tokenize import word_tokenize
for x in negative:
  for i in range(len(x)): 
    x[i] = word_tokenize(x[i])
negative_2 = word_tokenize(negative)


# In[ ]:


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def lemmatize_word(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
output = []
for sentence in tokenized_reviews:
  output.append(' '.join(lemmatize_word(' '.join(sentence))))


# Per the 

# In[ ]:


from transformers import pipeline
from nltk.tokenize import word_tokenize
#NLTK word tokenize
def word_tokenize_wrapper(text):
 return word_tokenize(text)
negative_2 = negative.apply(word_tokenize_wrapper)
negative_2.head()

classifier = pipeline('sentiment-analysis')
primal_result = classifier(negative_2)


# In[ ]:


negative.shape


# In[ ]:


tokenized_reviews_2 = lemmatization(tokenized_reviews)


# In[ ]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')


# In[ ]:


from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
def lemmatize_word(sentence):
    wnl = WordNetLemmatizer()
    for word, tag in pos_tag(word_tokenize(sentence)):
        if tag.startswith("NN"):
            yield wnl.lemmatize(word, pos='n')
        elif tag.startswith('VB'):
            yield wnl.lemmatize(word, pos='v')
        elif tag.startswith('JJ'):
            yield wnl.lemmatize(word, pos='a')
output = []
for sentence in tokenized_reviews:
  output.append(' '.join(lemmatize_word(' '.join(sentence))))


# In[ ]:


result_3 = open("outputt_tokenized.txt", "a")
result_3.write(str(output))


# In[ ]:


output[0]


# In[ ]:


all_words_negative = ' '.join([text for text in output])
all_words_negative =all_words_negative.split()
fdist_2 = FreqDist(all_words_negative)


# In[ ]:


df = pd.DataFrame(fdist_2.most_common(50), columns=['word', 'frequency'])
df.plot(kind='bar', x='word', y = 'count', figsize=(12,12))


# In[ ]:


# Build LDA model for topic 


# In[ ]:


dictionary = corpora.Dictionary(output)


# In[ ]:


doc_term_matrix = [dictionary.doc2bow(rev) for rev in reviews_2]


# In[ ]:


LDA = gensim.models.ldamodel.LdaModel


# ## Evaluate the topic models using Topic Quality, Diversity and Coherence

# In[ ]:


from gensim.models import CoherenceModel

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence Score: ', coherence_lda)


# In[ ]:


pip install octis


# In[ ]:


from octis.evaluation_metrics.diversity_metrics import TopicDiversity

metric = TopicDiversity(topk=10) # Initialize metric
topic_diversity_score = metric.score(model_output) # Compute score of the metric


# In[ ]:



import csv 
with open('tsv_file_for_octis.tsv', 'w', newline='') as f_output:
    tsv_output = csv.writer(f_output, delimiter='\t')
    tsv_output.writerow(list_to_octis)


# In[ ]:


import string
from octis.preprocessing.preprocessing import Preprocessing


# In[ ]:


spacy.load('en_core_web_sm')


# In[ ]:


preprocessor = Preprocessing(vocabulary=None, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=1, min_words_docs=0)
# preprocess
dataset = preprocessor.preprocess_dataset(documents_path=r'tsv_file_for_octis.tsv')


# In[ ]:


conda install -c conda-forge spacy


# In[ ]:


get_ipython().system('pip install spacy')


# In[ ]:


get_ipython().system('python -m spacy download en_core_web_sm')


# # Prepare for Fasttext

# In[ ]:


data_df.sentiment= data_df.sentiment.apply(lambda x: '__label__' + str(x))
data_df.size


# In[ ]:


data_df.sentiment


# In[ ]:


get_ipython().system('wget https://github.com/facebookresearch/fastText/archive/0.2.0.zip')
get_ipython().system('unzip 0.2.0.zip')
get_ipython().run_line_magic('cd', 'fastText-0.2.0')
get_ipython().system('make')


# In[ ]:


import os
import string
from octis.preprocessing.preprocessing import Preprocessing
os.chdir(os.path.pardir)
data_
# Initialize preprocessing
preprocessor = Preprocessing(vocabulary=None, max_features=None,
                             remove_punctuation=True, punctuation=string.punctuation,
                             lemmatize=True, stopword_list='english',
                             min_chars=1, min_words_docs=0)
# preprocess
dataset = preprocessor.preprocess_dataset(documents_path=r'..\corpus.txt', labels_path=r'..\labels.txt')

# save the preprocessed dataset
dataset.save('hello_dataset')


# In[ ]:


data_df.head()


# In[ ]:


data_df['sentences'] = data_df['summary'].fillna(' ') + " " + data_df['reviewText'] 


# In[ ]:


train=data_df.sample(frac=0.8,random_state=200)
test=data_df.drop(train.index)


# In[ ]:


train_list = train[['sentiment', 'sentences']].values.tolist()


# In[ ]:


df = pd.DataFrame(train_list, columns = ['sentiment', 'sentences'])


# In[ ]:


df.head()


# In[ ]:


test_list = test[['sentiment', 'sentences']].values.tolist()
df_2 = pd.DataFrame(test_list, columns = ['sentiment', 'sentences'])


# In[ ]:


df_2.head()


# In[ ]:


import csv
df.to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")

df_2.to_csv('test.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")


# Training the fastText classifier
model = fasttext.train_supervised('train.txt', wordNgrams = 2)

# Evaluating performance on the entire test file
model.test('test.txt')                      

# Predicting on a single input
model.predict(ds.iloc[2, 0])

# Save the trained model
#model.save_model('model.bin')


# In[ ]:


import csv
df.to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")

df_2.to_csv('test.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")


# In[ ]:


import csv
train[['sentiment', 'sentences']].to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")

test[['sentiment', 'sentences']].to_csv('test.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")



# In[ ]:


get_ipython().system('./fasttext supervised -input train.txt -output model -dim 2')


# In[ ]:


model = get_ipython().getoutput('./fasttext.load_model("/content/fastText-0.2.0/model.bin")')


# In[ ]:


model


# In[ ]:


get_ipython().system('./fasttext test /content/fastText-0.2.0/model.bin test.txt')


# In[ ]:


labels_test


# In[ ]:


f1_at_k = 2 * (0.961* 0.961) / (0.961 + 0.961)
f1_at_k


# In[ ]:



test[['sentences']].to_csv('test_sentences.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")


# In[ ]:


predictions = get_ipython().getoutput('./fasttext predict model.bin test_sentences.txt')


# In[ ]:


sum(x == y for x, y in zip(predictions, test['sentences'].tolist()))


# In[ ]:


true_labels_test = test['sentiment'].values.tolist()


# In[ ]:


accuracy = sum(x == y for x, y in zip(predictions,true_labels_test))/len(true_labels_test)


# In[ ]:


accuracy_fasttext_2gram = accuracy


# In[ ]:


accuracy_fasttext_2gram


# In[ ]:


from sklearn.metrics import classification_report
print(classification_report(true_labels_test, predictions[:360066]))


# In[ ]:


true_test_label = test['sentiment'].tolist()


# In[ ]:


len(predictions)


# In[ ]:


test['sentences'].to_csv('/content/fastText-0.2.0/test.txt', 
                                     index = False, 
                                     sep = ' ',
                                     header = None, 
                                     quoting = csv.QUOTE_NONE, 
                                     quotechar = "", 
                                     escapechar = " ")


# In[ ]:


testList=[]
with open("/content/fastText-0.2.0/test.txt") as f:
        testList = f.readlines()


# In[ ]:


testList


# In[ ]:


predictions_labels = get_ipython().getoutput('./fasttext predict model.bin test_predictions.txt')


# In[ ]:


len(predictions_labels)


# In[ ]:


list_to_test = test['sentences'].tolist()


# In[ ]:


np.savetxt("GFG.txt", 
           list_to_test,
           delimiter =", ", 
           fmt ='% s')


# In[ ]:


sum(1 for i, j in zip(predictions, true_test_label) if i != j)


# In[ ]:


num_records, precision_at_k, recall_at_k = model.test('/content/fastText-0.2.0/test.txt')
f1_at_k = 2 * (0.961* 0.961) / (0.961 + 0.961)
print("records\t{}".format(num_records))
print("Precision@{}\t{:.3f}".format(precision_at_k))


# In[ ]:





# In[ ]:


# Training the fastText classifier
model = fasttext.train_supervised('train.txt', wordNgrams = 2)

# Evaluating performance on the entire test file
model.test('test.txt')                      

# Predicting on a single input
model.predict(ds.iloc[2, 0])

# Save the trained model
#model.save_model('model.bin')


# Tokenize and remove punctations, to vectorize for our logistic regression and SVM models. 

# In[ ]:


import nltk
#nltk.download('punkt')
#tokenizedText=[nltk.word_tokenize(item) for item in data_df.reviewText]


# ## Tokenize the review Texts

# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_df['preprocessed'],data_df['sentiment'] , test_size=0.2)


# In[84]:


data_df.head()


# In[85]:


from tensorflow.keras.preprocessing.text import Tokenizer
vocab_size=10000
oov_token='OOV'
tokenizer=Tokenizer(num_words=vocab_size,oov_token=oov_token)


# In[86]:


X_train = list(X_train.astype(str))


# In[87]:


X_train


# In[88]:


tokenizer.fit_on_texts(X_train)
word_index=tokenizer.word_index
#print(word_index)


# In[89]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
train_sequences=tokenizer.texts_to_sequences(X_train)

# PADDING AND TRUNCATING
length_of_review=[len(i) for i in train_sequences] # get len of all rows
# max([len(x) for x in train_sequences]) ## 4425
print(np.median(length_of_review)) # to get no. of max length for padding
#print(train_sequences.shape)


# In[ ]:


x = np.arange(0, 1000, 100)
fig, ax = plt.subplots(figsize =(12, 12))
ax.hist(length_of_review, bins = x)
ax.set_xlabel('words in review')
ax.set_ylabel('Count of reviews with x words')


# ### Let us focus on 200 words. The reasoning is two fold: A higher word count word padding is too slow and crashes the kernel down the line, and let's face it: if someone cannot capture a product's merit in 200 words (excluding stop words and punctuation!), then we will focus on the first 200 to capture the sentiment. 

# In[90]:


max_len=200

padded_review=pad_sequences(train_sequences,maxlen=max_len,truncating='post',padding='post')
#print(padded_review.shape)


# In[91]:


# #f = open("myfile.txt", "x")
# f = open("myfile.txt", "w+")
# f.write(padded_review.astype(str))


# In[92]:


padded_review.shape


# In[ ]:





# ## Change back to 300d preferably. 

# In[93]:


embeddings_dictionary = dict()
glove_file = open('glove.6B.200d.txt', encoding='UTF-8')
for line in glove_file:
    #records = line.split()
    records = line.split()
    word = records[0]
    vector_dimensions = np.asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()


# In[94]:


word_index=tokenizer.word_index
len(tokenizer.word_index.items())


# In[96]:


embedding_matrix = np.zeros((len(tokenizer.word_index), 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
      embedding_matrix[index] = embedding_vector


# In[97]:


embedding_matrix


# Alternative Model: bag of words

# In[65]:


import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download("stopwords") # for removing stop words
nltk.download("wordnet") # for using wordnet lemmatizer
nltk.download('punkt') # for using nltk.tokenize.word_tokenize
nltk.download('averaged_perceptron_tagger') # for using nltk.pos_tag
nltk.download('omw-1.4')


# In[ ]:


wordnet = WordNetLemmatizer()
stopwords_set = set(stopwords.words('english'))
def get_wordnet_pos(pos_tag):

    if pos_tag.startswith('J'):  
        return 'a' # adjective
    elif pos_tag.startswith('V'): 
        return 'v' # verb
    elif pos_tag.startswith('R'):
        return 'r' # adverb
    else:           
        return 'n' # fallback to noun

def lemmatize_text(text): 
    # Text input is string, returns lowercased strings.
    return [wordnet.lemmatize(word.lower(), pos=get_wordnet_pos(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    
    return [word for word in lemmatize_text(text)
            if word not in stopwords_set
            and word.isalpha()]  # english words only (not punctuations nor numbers)


# In[ ]:


data_df.head()


# # LSTM model using embedding

# In[115]:


pip install numpy~=1.19.5


# In[131]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Flatten, Dense   

model1 = Embedding(len(tokenizer.word_index), 200,  weights=[embedding_matrix], input_length=200, trainable=False)
model = Sequential([
    Embedding(len(tokenizer.word_index), 200,  weights=[embedding_matrix], input_length=200, trainable=False),
    LSTM(128),
    Dense(10, activation = 'sigmoid')])
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])


# In[ ]:


history = model.fit(padded_review, y_train, batch_size = 128, epochs = 20, validation_split = 0.20, verbose = 1)


# In[126]:


#embedding_matrix = embedding_matrix.tolist()
embedding_matrix = np.array(embedding_matrix)


# In[112]:


import numpy as np 
accuracy = [0.9258, 0.9618, 0.9676, 0.9684, 0.9689, 0.9690, 0.9692, 0.9686, 0.9688, 0.9688]
#loss_counter = [0.1117 , 0.0966,0.0919, 0.0870, 0.0880, 0.0643]
epochs_counter = np.arange(1, len(accuracy)+1, 1)
plt.plot(epochs_counter, accuracy, '-*')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.figure(2)
plt.plot(epochs_counter, loss_counter, 'r', '-*')
plt.xlabel("Epoch")
plt.ylabel("Loss")


# In[ ]:


## Increasing epochs per feedback


# In[ ]:


from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, LSTM, Flatten, Dense     # import some layers for training.
model = Sequential([
    Embedding(len(tokenizer.word_index), 200,  weights=[embedding_matrix], input_length=200, trainable=False),
    LSTM(128),
    Dense(1, activation = 'sigmoid')
])

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', 'precision', 'recall'])


# In[146]:


X_test = X_test.astype(str)
X_test = list(X_test)
X_test = tokenizer.texts_to_sequences(X_test)


# In[147]:


padded_review_2=pad_sequences(X_test,maxlen=max_len,truncating='post',padding='post')


# In[ ]:


y_test = np.array(y_test)


# In[ ]:


loss, accuracy = model.evaluate(padded_review_2, y_test)
print('Test accuracy :', accuracy)


# ## Random Forest-- One of my favorites

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1)
rf_model = rf.fit(padded_review, y_train)


# In[ ]:


X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test,maxlen=max_len,truncating='post',padding='post')


# In[ ]:


predictions = rf.predict(X_test)


# In[ ]:


rf.score(X_test, y_test)


# In[ ]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(padded_review,data_df['sentiment'],test_size=0.2,random_state=123)


# # Logistic Regression

# In[132]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(solver='lbfgs', max_iter=1000)


# In[133]:


clf.fit(padded_review, y_train)


# In[145]:


padded_review


# In[150]:


predictions_ctv = clf.predict(padded_review_2)


# In[ ]:


print("The accuracy of logistic regression, is ", 100*get_accuracy(clf, predictions_ctv, y_test),'%')


# In[153]:


print("The accuracy of logistic regression, is ", 100*get_accuracy(clf, predictions_ctv, y_test),'%')


# In[ ]:


score = clf.score(x_test, y_test)
print(score)


# In[154]:


import sklearn.metrics 
sklearn.metrics.precision_score(y_test, predictions_ctv)


# In[156]:


sklearn.metrics.f1_score(y_test, predictions_ctv)


# In[ ]:


# Let's use a simple accuracy measure to compare the different models. 


# In[152]:


def get_accuracy(clf, predictions, yvalid):
    return np.mean(predictions == yvalid)


# In[ ]:


data_df['sentiment']


# In[ ]:





# # SVM

# In[ ]:


#data_df.iloc[10] = padded_review
#sample = data_df.sample(frac=0.05)
len(train_sequences)==len(data_df.sentiment)


# In[ ]:


from sklearn import svm
cl_v = svm.SVC(kernel='rbf')
cl_v.fit(x_train, y_train)


# In[ ]:


cl_v.predict


# In[ ]:


# TF-IDF 


# In[ ]:


punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~'''
tokenizedText= [[word for word in data_df.reviewText if word not in punc] for review in tokenizedText]


# In[ ]:


tfv = TfidfVectorizer(min_df=3,  max_features=None, 
            strip_accents='unicode', analyzer='word',token_pattern=r'\w{1,}',
            ngram_range=(1, 3), use_idf=1,smooth_idf=1,sublinear_tf=1,
            stop_words = 'english')
tfv.fit(list(X_train[:,-1]))
xtrain_tfv =  tfv.transform(X_train) 
xtest_tfv = tfv.transform(X_test[:,-1])


# In[ ]:


clf = LogisticRegression()
clf.fit(xtrain_tfv, y_train)
predictions = clf.predict_proba(xtest_tfv)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_df['reviewText'],data_df['sentiment'] , test_size=0.2)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_df['reviewText'], data_df['sentiment'], test_size = 0.20, random_state = 100)


# In[ ]:


# import csv 
# [y_train, X_train].to_csv('train.txt', 
#                                           index = False, 
#                                           sep = ' ',
#                                           header = None, 
#                                           quoting = csv.QUOTE_NONE, 
#                                           quotechar = "", 
#                                           escapechar = " ")


# In[ ]:


training_set = pd.concat([y_train, X_train], axis=1)
training_set.to_csv('training.csv', index=False)
test_set = pd.concat([y_test, X_test], axis=1)
test_set.to_csv('test.csv', index=False)


# In[ ]:


# data_df['split'] = np.random.randn(data_df.shape[0], 1)
# data_df = data_df[['sentiment'], ['reviewText']]
# msk = np.random.rand(len(data_df)) <= 0.7

# train = data_df[msk]
# test = data_df[~msk]
# train.to_csv('train.csv', index=False)


# In[ ]:


import fasttext
model = fasttext.train_supervised('training.csv', wordNgrams = 2)


# In[ ]:


test_set


# In[ ]:


test_set = pd.concat([y_test, X_test], axis=1)
test_set.size
test_set = test_set.sample(frac=0.1)
test_set.size
test_set.to_csv('test.csv', index=False)


# In[ ]:


test_set.size


# In[ ]:


model.test('test.csv')


# In[ ]:


# Precision at one = 0.37907022638211496
# Recall at one = 0.37907022638211496
# We can also obtain the top five labels for any prediction. 


# In[ ]:





# In[ ]:


model.predict("Why clothes?", k=5)


# In[ ]:


model.save_model("model_fasttext.bin")


# In[ ]:


from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import collections
tokenizer = ToktokTokenizer()#tokenize positive reviews
tokens = tokenizer.tokenize(data_df['reviewText'])#print ten words
print(tokens[:20])


# In[ ]:


tokens = [word for word in tokens if not word in stopwords.words()]
word_counts = collections.Counter(tokens)


# In[ ]:


word_counts


# In[ ]:


import tensorflow.keras


# In[ ]:


max_fatures = 2000
tokenizer = Tokenizer(num_words = max_fatures, split = ' ')
tokenizer.fit_on_texts(df['Text'].values)


# In[ ]:


from keras.preprocessing.text import Tokenizer
vocab_size = 10000
oov_token = "<OOV>"
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(X_train)


# In[ ]:


X_test = tokenizer.texts_to_sequences(X_test)
X_test = pad_sequences(X_test,maxlen=max_len,truncating='post',padding='post')


# In[ ]:


# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
nb=MultinomialNB()
nb.fit(padded_review,y_train)


# In[ ]:


nb.score(X_test, y_test)


# In[ ]:


X_test


# In[ ]:


# Logistic Regression 
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
lr.fit(train_data_features,train.Score)
target_pred_Log_Reg=lr.predict(test_dtm)
print metrics.accuracy_score(test.Score,target_pred_Log_Reg)
print metrics.confusion_matrix(test.Score,target_pred_Log_Reg)
print metrics.classification_report(test.Score,target_pred_Log_Reg)


# In[ ]:


from gensim.utils import simple_preprocess
data_df.iloc[:, 7] = data_df.iloc[:, 7].apply(lambda x: ' '.join(simple_preprocess(x)))


# In[ ]:


import csv 
data_df[['overall', 'reviewText']].to_csv('train_NLP.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")


# In[ ]:


from gensim.utils import simple_preprocess
rating = rating.apply(lambda x: '__label__' + str(x))


# In[ ]:


data_df = data_df.apply (pd.to_numeric, errors='coerce')


# In[ ]:


data_df.head()


# In[ ]:


dataset[['rating', 'text']].to_csv('train.txt', 
                                          index = False, 
                                          sep = ' ',
                                          header = None, 
                                          quoting = csv.QUOTE_NONE, 
                                          quotechar = "", 
                                          escapechar = " ")


# In[ ]:


import fasttext

model = fasttext.train_supervised(input="cooking.train")


# In[ ]:


data_df.iloc[:,7]


# In[ ]:


from gensim.utils import simple_preprocess


# ## BERT

# In[ ]:


from torch.utils.data import RandomSampler
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer

from dataset import SentimentDataset
from model import SentimentBERT

BERT_MODEL = 'bert-base-uncased'
NUM_LABELS = 2  # negative and positive reviews


# In[ ]:





# ## CTM

# In[4]:


get_ipython().run_cell_magic('capture', '', '!pip install contextualized-topic-models==2.3.0')


# In[5]:


get_ipython().run_cell_magic('capture', '', '!pip install pyldavis')


# In[6]:


from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
import nltk


# In[7]:


import csv

with open('/content/neative_topic_dictionary.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


# In[ ]:


lemma_list = data


# In[ ]:


lemma_list


# In[ ]:


flattened = [sublist[1] for sublist in lemma_list]


# In[ ]:


flattened


# In[ ]:


from nltk.corpus import stopwords as stop_words

nltk.download('stopwords')

#documents = [line.strip() for line in open(text_file, encoding="utf-8").readlines()[0:2000]]

stopwords = list(stop_words.words("english"))

sp = WhiteSpacePreprocessingStopwords(flattened, stopwords_list=stopwords)
preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()


# In[162]:


import csv

with open('negative_lemmatized.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)


# In[163]:


negtive_list = data


# In[164]:


negtive_list 


# In[165]:


from gensim import corpora
dictionary = corpora.Dictionary(negtive_list)
# Filter out words that occur less than 20 documents, or more than 50% of the documents.
#dictionary.filter_extremes(no_below=0, no_above=0.80)


# In[166]:


from gensim.models import LdaModel
doc_term_matrix = [dictionary.doc2bow(rev) for rev in negtive_list]
LDA_model = LdaModel(doc_term_matrix, num_topics=35, id2word=dictionary, passes=1)


# In[167]:


dict_of_topics = {}
for idx, topic in LDA_model.print_topics(-1):
    #print('Topic: {} \nWords: {}'.format(idx, topic))
    dict_of_topics[idx] = topic


# In[168]:


dict_of_topics


# In[169]:


from gensim.models import CoherenceModel # Compute Coherence Score

coherence_model_lda = CoherenceModel(model=LDA_model, texts=negtive_list, dictionary = dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# In[50]:


dictionary = corpora.Dictionary(negtive_list)


# In[172]:


BERTopic.load("topic_model")


# In[ ]:


from gensim.models import CoherenceModel # Compute Coherence Score

coherence_model_lda = CoherenceModel(model=LDA_model, texts=negtive_list, dictionary = dictionary, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

