"""
Transformers on Typical Adjectives

Most of this script was originally written by Stephan Meylan with alterations by Claire Bergey.
"""
import numpy as np
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pytorch_pretrained_bert import BertTokenizer,BertForMaskedLM
import math
import time
import torch
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)



"""Human Data"""

df = pd.read_csv("../data/human_judgments_ldp_cabnc.csv")

"""higher turker judgment = more typical

# GPT2
"""

gp2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

def gpt2_score(text, model, tokenizer): 
  '''get sequence probability under gpt2'''
  input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
  tokenize_input = tokenizer.tokenize(text)
  #50256 is the token_id for <|endoftext|>
  tensor_input = torch.tensor([ [50256]  +  tokenizer.convert_tokens_to_ids(tokenize_input)])
  with torch.no_grad():
      outputs = model(tensor_input, labels=tensor_input)
      loss, logits = outputs[:2]  

  lp = 0.0
  for i in range(len(tokenize_input)):
      masked_index = i
      predicted_score = logits[0, masked_index]
      predicted_prob = softmax(np.array(predicted_score))
      lp += np.log(predicted_prob[tokenizer.convert_tokens_to_ids([tokenize_input[i]])[0]])      
  return(lp)

def gpt2_candidate_probs(context, target, model, tokenizer, candidates = None): 
  '''get probability under GPT2 of all alternatives to target in the context'''  
  begin = time.time()
  candidate_scores = [] 
  if candidates is None:
    candidates = tokenizer.get_vocab().keys()
  for candidate in candidates:
    text = context.replace(target, candidate)
    prob = gpt2_score(text, model, tokenizer)
    candidate_scores.append({'word':candidate, 'prob':prob})

  rdf = pd.DataFrame(candidate_scores)
  rdf = rdf.sort_values(by = 'prob', ascending=False)
  rdf['rank'] = range(rdf.shape[0])
  rdf['normalized_prob'] = np.exp(rdf.prob) / np.max(np.exp(rdf.prob))
  print('gpt2_candidate_probs took '+str(time.time() - begin)+'s')
  return(rdf)

def gpt2_normalized_score(text, target, model, tokenizer, candidates = None): 
  '''get the normalized probability of target under GPT2 normalized by highest prob in candidates'''
  begin = time.time()
  sequence_probs = gpt2_candidate_probs(text, target, gpt2_model, gp2_tokenizer, candidates)
  print(time.time() - begin)
  return(sequence_probs.loc[sequence_probs.word == target].iloc[0].normalized_prob)

gpt2_normalized_score("I saw the yellow banana", "yellow", gpt2_model, gp2_tokenizer, candidates =np.unique(df.adjective))

# Get the sequence probabilities from gpt2

df['gpt2_normalized_prob'] = [gpt2_normalized_score(
  'I saw the ' + x['adjective'] + ' ' + x['noun'], x['adjective'], gpt2_model, gp2_tokenizer, candidates = np.unique(df.adjective)) for x in df.to_dict('records')]

ax = sns.regplot(x="turker_judgment", y="ldp_similarity", data=df,
  line_kws = {'color':'red'}, scatter_kws = {'alpha': 0.1})

sns.regplot(x="turker_judgment", y="gpt2_normalized_prob", data=df,
  line_kws = {'color':'red'}, scatter_kws = {'alpha': 0.1})

df.gpt2_normalized_prob

"""# BERT"""

bertMaskedLM = BertForMaskedLM.from_pretrained('bert-base-uncased')
bertMaskedLM.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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


def bert_score(text, completion, model, tokenizer, normalize = False, return_type = 'prob'):     
  continuations = bert_completions(text, model, tokenizer)
  if not completion in set(continuations.word):
    return(None) # continuation is not in the BERT vocab    
  score = continuations.loc[continuations.word == completion].prob.values[0]
  if return_type == 'normalized_prob':              
      highest_score = continuations.iloc[0].prob
      return( score /  highest_score)
  elif return_type == 'prob':
      return(score)
  elif return_type == 'rank':
      return(np.where(continuations.word == completion)[0][0])
  else:
      raise ValueError('return_type should be "prob" or "rank"')

for rt in ["normalized_prob", "prob", "rank"]:
  print(bert_score("[MASK] banana", "yellow", bertMaskedLM, tokenizer, return_type = rt))

compare_completions("[MASK] banana", ['yellow','green','blue'], bertMaskedLM, tokenizer)

# takes ~5 minutes to run on all of the words
df['bert_p'] = [bert_score('I saw the [MASK] ' + x['noun'], x['adjective'], bertMaskedLM, tokenizer,
  return_type = 'normalized_prob' ) for x in df.to_dict('records')]

print(str(np.round((np.sum(np.isnan(df.bert_p)) / df.shape[0])  * 100., 2))+ '% of pairs missing')

sns.regplot(x="turker_judgment", y="bert_p", data=df,
  line_kws = {'color':'red'}, scatter_kws = {'alpha': 0.1})

# takes ~5 minutes to run on all of the words
df['bert_rank'] = [bert_score('I saw the [MASK] ' + x['noun'], x['adjective'], bertMaskedLM, tokenizer,
  return_type = 'rank' ) for x in df.to_dict('records')]

g = sns.regplot(x="turker_judgment", y="bert_rank", data=df,
  line_kws = {'color':'red'}, scatter_kws = {'alpha': 0.1})
g.set(ylim=(0, 1000))

df.to_csv('../data/all_model_judgments_ldp_cabnc.csv', index = False)

"""# Todo: 
[ ] Pull out the matched set of high and low atypicality pairs like in Figure 3  
[ ] Try different frames (right now "I saw the <adj> <noun>", may not be appropriate for all bigrams)  
[ ] Look at coverage in the vocabulary    
[ ] Word2Vec: what happens when one normalizes by highest similarity in the lexicon?   
[ ] Evaluate GPT on a set of continuations for each word  

"""