import torch
import torch.nn.functional as F
import numpy as np
import sys
import pickle5 as pkl
import numpy as np
import pandas as pd
import time
import math
from scipy.stats import entropy
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
epsilon = 0.000000000001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

df = pd.read_csv("../data/final_pairs_ldp_cabnc.csv")
alt_adjs = pd.read_csv("../data/250_common_adjs.csv")
alternative_adjs = np.unique(alt_adjs.adjective).tolist()

def get_prob(lst, word):
	indices = [(i, sub.index(word)) for (i, sub) in enumerate(lst) if word in sub]
	if len(indices) == 0 or len(indices[0]) == 0:
		print(word + " not in gpt2 vocab")
		return epsilon
	else:
		return lst[indices[0][0]][1]

def next_gpt2_word(sentence):
	context_tokens = enc.encode(sentence)
	context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
	prev = context
	with torch.no_grad():
		logits, past = model(prev, past=None)
	logits=F.softmax(logits[:, -1, :], dim=-1)
	pw=zip([enc.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
	pw=sorted(pw, key = lambda x: -x[1])
	count=0
	pw=[i for i in pw if i[0][0] != "-"]
	for i in pw:
		count += i[1]
	pw=[[i[0], i[1]/count] for i in pw]
	return pw


def get_adj_prob(adjective, noun):
	begin = time.time()
	sentence = 'The ' + noun + ' is'
	all_probs = next_gpt2_word(sentence)
	all_adj_probs = pd.DataFrame(columns = ['adj','prob'])
	adjs = np.unique(np.append(alternative_adjs, adjective))
	for adj_candidate in adjs:
		adj_tokenized = enc.encode(" " + adj_candidate)
		if (len(adj_tokenized) == 1):
			adj_prob = get_prob(all_probs, " " + adj_candidate)
			row = {'adj': adj_candidate, 'prob': adj_prob}
			all_adj_probs = all_adj_probs.append(row, ignore_index = True)
		else:
			token_sentence = 'The ' + noun + ' is'
			adj_prob = 1
			for i in range(0, len(adj_tokenized)):
				if (i != 0):
					token_probs = next_gpt2_word(sentence)
					adj_prob = adj_prob * get_prob(token_probs, enc.decode([adj_tokenized[i]]))
					token_sentence = token_sentence + enc.decode([adj_tokenized[i]])
				else:
					token_probs = all_probs
					word = enc.decode([adj_tokenized[i]])
					adj_prob = adj_prob * get_prob(token_probs, word)
					token_sentence = token_sentence + word
			row = {'adj': adj_candidate, 'prob': adj_prob}
			all_adj_probs = all_adj_probs.append(row, ignore_index = True)
	sum = all_adj_probs['prob'].sum()
	this_prob = all_adj_probs[all_adj_probs['adj'] == adjective]['prob'].iloc[0]
	print(this_prob/sum)
	print('probs took '+str(time.time() - begin)+'s')
	return this_prob/sum

df['prob'] = df.apply(lambda x: get_adj_prob(x['adjective'], x['noun']), axis = 1)
df.to_csv('../data/gpt2_judgments_ldp_cabnc.csv')
