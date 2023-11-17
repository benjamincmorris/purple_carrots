import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import math
import time
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
import scipy
from scipy.stats import entropy

epsilon = 0.000000000001

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

df = pd.read_csv("../data/final_pairs_ldp_cabnc.csv")
alt_adjs = pd.read_csv("../data/250_common_adjs.csv")
alternative_adjs = np.unique(alt_adjs.adjective).tolist()

def bert_completions(text, model, tokenizer):

  text = '[CLS] ' + text + ' [SEP]'
  tokenized_text = tokenizer.tokenize(text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  masked_index = tokenized_text.index('[MASK]')  

  # Create the segments tensors.
  segments_ids = [0] * len(tokenized_text)

  # Convert inputs to PyTorch tensors
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_tensors = torch.tensor([segments_ids])

  # Predict all tokens
  with torch.no_grad():
      predictions = model(tokens_tensor, segments_tensors)
 
  probs = softmax(predictions[0, masked_index].data.numpy())
  words = tokenizer.convert_ids_to_tokens(range(len(probs)))
  word_predictions  = pd.DataFrame({'prob': probs, 'word':words})
  word_predictions = word_predictions.sort_values(by='prob', ascending=False)    
  word_predictions['rank'] = range(word_predictions.shape[0])
  return(word_predictions)
  
def compare_completions(context, candidates, bertMaskedLM, tokenizer):
  continuations = bert_completions(context, bertMaskedLM, tokenizer)
  return(continuations.loc[continuations.word.isin(candidates)])


def bert_score(lst, w):     
  if lst[lst['word'] == w.lower()].empty:
  	print(w + ' is not in BERT vocab')
  	return(epsilon) # word is not in the BERT vocab    
  score = lst[lst['word'] == w.lower()]['prob'].iloc[0]
  return(score)

def get_adj_prob_full(adjective, noun):
	begin = time.time()
	sentence = 'The ' + noun + ' is [MASK]'
	all_probs = bert_completions(sentence, model, tokenizer)
	all_adj_probs = pd.DataFrame(columns = ['adj','prob'])
	adjs = np.unique(np.append(alternative_adjs, adjective))
	for adj_candidate in adjs:
		adj_tokenized = tokenizer.tokenize(adj_candidate)
		if len(adj_tokenized) == 1:
			adj_prob = bert_score(all_probs, adj_candidate)
			row = {'adj': adj_candidate, 'prob': adj_prob}
			all_adj_probs = all_adj_probs.append(row, ignore_index = True)
		else:
			adj_prob = 1
			token_sentence = 'The ' + noun + ' is [MASK]'
			for i in range(0, len(adj_tokenized)):
				if (i == len(adj_tokenized) - 1):
					token_sentence = token_sentence + " ."
				if (i != 0):
					token_probs = bert_completions(sentence, model, tokenizer)
				else:
					token_probs = all_probs
				adj_prob = adj_prob * bert_score(token_probs, adj_tokenized[i])
				idx = token_sentence.index('[MASK]')
				token_sentence = token_sentence[:idx] + adj_tokenized[i] + " " + token_sentence[idx:]
			row = {'adj': adj_candidate, 'prob': adj_prob}
			all_adj_probs = all_adj_probs.append(row, ignore_index = True)
	sum = all_adj_probs['prob'].sum()
	this_prob = all_adj_probs[all_adj_probs['adj'] == adjective]['prob'].iloc[0]
	print(this_prob/sum)
	print('probs took '+str(time.time() - begin)+'s')
	return this_prob/sum

def get_adj_prob(row):
	begin = time.time()
	print(row['adjective'] + " " + row['noun'])
	sentence = '[MASK] ' + row['noun']
	all_probs = bert_completions(sentence, model, tokenizer)
	this_tokenized = tokenizer.tokenize(row['adjective'])
	if (len(this_tokenized) > 1):
		is_multi_token = True
		adj_prob = 1
		token_sentence = sentence
		for i in range(0, len(this_tokenized)):
			print(token_sentence)
			token_probs = bert_completions(sentence, model, tokenizer)
			adj_prob = adj_prob * bert_score(token_probs, this_tokenized[i])
			idx = token_sentence.index('[MASK]')
			token_sentence = token_sentence[:idx] + this_tokenized[i] + " " + "[MASK] " + row['noun']
	else:
		is_multi_token = False
		adj_prob = bert_score(all_probs, row['adjective'])
	print(adj_prob)
	print('probs took '+str(time.time() - begin)+'s')
	row['prob'] = adj_prob
	row['is_multi_token'] = is_multi_token
	return row

df = df.apply(get_adj_prob, axis = 1)
df.to_csv('../data/bert_judgments_ldp_cabnc.csv')
