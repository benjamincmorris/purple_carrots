import torch
import torch.nn.functional as F
import numpy as np
import sys
import pickle5 as pkl
import numpy as np
import pandas as pd
import math
from scipy.stats import entropy
from pytorch_pretrained_bert import GPT2LMHeadModel, GPT2Tokenizer
epsilon = 0.000000000001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

enc = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.to(device)
model.eval()

df = pd.read_csv("../data/human_judgments_ldp_cabnc.csv")
alternative_adjs = np.unique(df.adjective).tolist()

def get_prob(lst, w):
	word = " " + w
	indices = [(i, sub.index(word)) for (i, sub) in enumerate(lst) if word in sub]
	if len(indices) == 0 or len(indices[0]) == 0:
		return epsilon
	else:
		return lst[indices[0][0]][1]

def next_gpt2_word(sentence, nmax=10000):
	context_tokens = enc.encode(sentence)
	context = torch.tensor(context_tokens, device=device, dtype=torch.long).unsqueeze(0)
	prev = context
	with torch.no_grad():
		logits, past = model(prev, past=None)
	logits=F.softmax(logits[:, -1, :], dim=-1)
	pw=zip([enc.decode([i]) for i in range(logits[0].size()[0])], logits[0].tolist())
	pw=sorted(pw, key = lambda x: -x[1])
	count=0
	pw=[i for i in pw[0:nmax] if i[0][0] != "-"]
	for i in pw:
		count += i[1]
	pw=[[i[0], i[1]/count] for i in pw]
	return pw[0:nmax]


def get_gpt2_probs_ents(context, sentence, log = True):
	probs = []
	ents = []
	for i in range(0, len(sentence.split())):
		fragment = context + ' '.join(sentence.split()[0:i])
		completions = next_gpt2_word(fragment)
		ent = entropy(list(zip(*completions))[1], base = 2)
		ents.append(ent)
		word = " " + sentence.split()[i]
		word_prob = epsilon
		for sublist in completions:
			if sublist[0] == word:
				word_prob = sublist[1]
		if log:
			word_prob = math.log2(word_prob)
		probs.append(word_prob)
	return(probs, ents)

def get_adj_prob(adjective, noun):
	sentence = 'The ' + noun + ' is'
	all_probs = next_gpt2_word(sentence)
	all_adj_probs = pd.DataFrame(columns = ['adj','prob'])
	adjs = alternative_adjs
	for adj_candidate in adjs:
		adj_prob = get_prob(all_probs, adj_candidate)
		row = {'adj': adj_candidate, 'prob': adj_prob}
		all_adj_probs = all_adj_probs.append(row, ignore_index = True)
	sum = all_adj_probs['prob'].sum()
	this_prob = get_prob(all_probs, adjective)
	return this_prob/sum

df['prob'] = df.apply(lambda x: get_adj_prob(x['adjective'], x['noun']), axis = 1)
df.to_csv('../data/gpt2_judgments_ldp_cabnc.csv')
