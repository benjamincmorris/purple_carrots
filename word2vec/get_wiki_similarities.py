from __future__ import absolute_import, division, print_function, unicode_literals
import io
import os
from gensim import utils
import gensim.models
import gensim.models.word2vec
from gensim.test.utils import datapath
from gensim.models import KeyedVectors
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)             
import numpy as np   
import csv
from matplotlib import pyplot
from numpy import cov
import seaborn
import itertools

model = gensim.models.Word2Vec.load("ldp_adult_word2vec.model")

sim_judgments = {}
with open('../data/judgments_session.csv', mode='r') as csv_file:
    readCSV = csv.DictReader(csv_file, delimiter=',')
    for row in readCSV:
        if row['adj'] not in sim_judgments:
            sim_judgments[row['adj']] = {}
        sim_judgments[row['adj']][row['noun']] = row['mean_typ']


wiki_model = KeyedVectors.load_word2vec_format('../data/wiki-news-300d-1M.vec')

wiki_sim_judgments = []
ldp_sim_judgments =[]
human_judgments = []
word_pairs = []

for adj in sim_judgments:
    for noun in sim_judgments[adj]:
        if adj in wiki_model.vocab and noun in wiki_model.vocab and adj in model.wv.vocab and noun in model.wv.vocab:
            human_judgments.append(float(sim_judgments[adj][noun]))
            wiki_sim_judgments.append(float(wiki_model.similarity(adj, noun)))
            ldp_sim_judgments.append(float(model.similarity(adj, noun)))
            word_pairs.append((adj,noun))


seaborn.stripplot(ldp_sim_judgments, wiki_sim_judgments, s = 2, jitter = 3)
covariance = np.corrcoef(ldp_sim_judgments, wiki_sim_judgments)
print(covariance)

'''
with open('wiki_sim_judgments.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in wiki_sim_judgments and (adj,noun) in word_pairs:
        writer.writerow(adj + ',' + noun + ',' + str(row))

with open('human_judgments.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=' ',
        quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for row in human_judgments and (adj,noun) in word_pairs:
        writer.writerow(adj + ',' + noun + ',' + str(row))
'''

all_judgments = zip(word_pairs, human_judgments, wiki_sim_judgments, ldp_sim_judgments)

with open('all_judgments.csv', 'w') as csvfile:
    writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(["noun","adj","turker_judgment","wiki_similarity","ldp_similarity"])
    for (words, human, wiki, ldp) in all_judgments:
        writer.writerow([words[1],words[0],str(human),str(wiki),str(ldp)])


wiki_model.wv.evaluate_word_pairs(datapath('simlex999.txt'))


