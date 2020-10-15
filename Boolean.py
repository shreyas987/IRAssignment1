import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from operator import itemgetter, attrgetter
import numpy as np
#nltk.download('punkt')
#nltk.download('wordnet')

class invindex:
	def __init__(self):
		self.word=""
		self.files = []

stopwords = set(stopwords.words("english"))
objlist = []
path = '/home/shreyas/Desktop/Sem3/IR/Assignment1/corpus'

print("Constructing inverted index : ")
for filename in os.listdir(path):
	#print(filename)
	f=open(path+"/"+filename,"r")


		#f=open("file.txt","r")
	strinp = f.read()
		##Lower case
	strinp = strinp.lower()
	##remove numbers
	strinp1 = re.sub(r'\d+', '', strinp)
	##HTML tags
	strinp2 = re.sub(r'<.*?>','',strinp1)
	#special characters
	strinp3 = re.sub(r'[^\w\s]', '', strinp2) 
	##newline
	strinp4 = re.sub(r'[\n]','',strinp3)
	##extra spaces
	strinp4 = strinp4.strip()
	##tokenize
	tokenize_words = word_tokenize(strinp4)
	##stopwords

	filtered_sample_text = [w for w in tokenize_words if not w in stopwords]
	ps = PorterStemmer()
	lemmatizer = WordNetLemmatizer()
	stemmed_sample_text = []
	for token in filtered_sample_text:
	    stemmed_sample_text.append(ps.stem(token))
	lemma_sample_text = []
	for token in filtered_sample_text:
	    lemma_sample_text.append(lemmatizer.lemmatize(token))

	##remove repeated words
	lemma_sample_text = list(dict.fromkeys(lemma_sample_text))
	for w in lemma_sample_text:
		fl=0
		for obj in objlist:
			if obj.word == w:
				obj.files.append(filename)
				fl=1
				break
			else:
				continue
		if fl==0:
			obj = invindex()
			obj.word = w
			obj.files.append(filename)
			objlist.append(obj)

sorted(objlist,key=attrgetter('word'))	

print("Completed...\nThe inverted index constructed : ")
for o in objlist:
	print(o.word)
	for f in o.files:
		print(f)
	print("\n")
	
	
while(True):

	s = input("Enter your search or enter ~ to exit : ")
	if s=='~':
		break

	##Lower case
	s = s.lower()
	##remove numbers
	strinp1 = re.sub(r'\d+', '', s)
	##HTML tags
	strinp2 = re.sub(r'<.*?>','',strinp1)
	#special characters
	strinp3 = re.sub(r'[^\w\s]', '', strinp2) 
	##newline
	strinp4 = re.sub(r'[\n]','',strinp3)
	##extra spaces
	strinp4 = strinp4.strip()
	##tokenize
	tokenize_words = word_tokenize(strinp4)
	##stopwords
	filtered_input_text = [w for w in tokenize_words if not w in stopwords]
	lemma_input_text = []
	for token in filtered_input_text:
	    lemma_input_text.append(lemmatizer.lemmatize(token))
	##remove repeated words
	lemma_input_text = list(dict.fromkeys(lemma_input_text))

	ans = []
	for word in lemma_input_text:
		for obj in objlist:
			if obj.word == word:
				ans.extend(obj.files)
				break;
	#ans = list(dict.fromkeys(ans))
	x=np.array(ans)
	print("Search words are found in following files in the corpus : ")
	print(np.unique(x))
		
		
