import os
import pandas as pd
import openai

openai.api_key = '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'

df = pd.read_csv("../data/final_pairs_ldp_cabnc.csv")
prepend_prompt = "You are doing a task in which you rate how common it is for certain things to have certain features. You respond out of the following options: Never, Rarely, Sometimes, About half the time, Often, Almost always, or Always. "

def get_rating(adjective, noun, adj_article, article):
	if article == "NA" or not article or type(article) != str:
		question = "How common is it for " + noun + " to be " + adjective + " " + noun + "?"
	else:
		question = "How common is it for " + article + " " + noun + " to be " + adj_article + " " + adjective + " " + noun + "?"
	print(question)
	response = openai.Completion.create(
		model="text-davinci-003",
		prompt=prepend_prompt + " " + question,
		temperature=0.6,
		)
	return response.choices[0].text

#df['gpt3_judgment'] = df.apply(lambda x: get_rating(x['adjective'], x['noun'], x['adj_article'], x['article']).strip(), axis = 1)
#df.to_csv('../data/gpt3_judgments_ldp_cabnc_3.csv', index = None)

print(get_rating("wooden", "horse", "a", "a"))
print(get_rating("swimming", "suit", "a", "a"))