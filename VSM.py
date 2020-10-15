import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import os
from operator import itemgetter, attrgetter
import numpy as np
import math
from numpy import dot
from numpy.linalg import norm
import operator
#nltk.download('punkt')
#nltk.download('wordnet')

#cosine similarity
def cosine_sim(vec1,vec2):
	if norm(vec1)*norm(vec2) == 0:
		return 0
	result = np.dot(vec1, vec2)/(norm(vec1)*norm(vec2))
	return result
#########

class invindex:
	def __init__(self):
		self.word=""
		self.files = []
###########


stopwords = set(stopwords.words("english"))
objlist = []
path = '/home/shreyas/Desktop/Sem3/IR/Assignment1/corpus'

print("Calculating tf-idf for the docs...")

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

##Have a list of objects each object has a word and a list of documents in which it occurs
N = 100

vect = {filename : [0]*len(objlist) for filename in os.listdir(path)}

idfDict = {}
##Calculate df of each of these words
for ob in objlist:
	wrd = ob.word
	df = len(ob.files)
	idf = math.log((N/df),10)
	idfDict[wrd] = idf
	

tfdict = {}

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
	tempdict={}
	for token in filtered_sample_text:
	    
	    if token in tempdict:
	    	tempdict[token] = tempdict[token]+1
	    else:
	    	tempdict[token] = 1
	    	
	
	N = len(stemmed_sample_text)
	for key in tempdict:
		tempdict[key]=1+math.log(tempdict[key],10)
	
	tfdict[filename]=tempdict
	
###########################

while(True):

	s = input("Enter your search or enter ~ to exit ")
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

	#print(lemma_input_text)
	#query vector calculate tf
	Qdict = {}
	for word in lemma_input_text:
		if word in Qdict:
			Qdict[word] = Qdict[word]+1
		else:
			Qdict[word] = 1
	for key in Qdict:
		Qdict[key] = 1+math.log(Qdict[key],10)
	
	##Query vector
	Qvec =[]
	for word in Qdict:
		tf = Qdict[word]
		idf = 0
		if word in idfDict:
			idf = idfDict[word]
		else:
			idf=0
		prod = tf*idf
		Qvec.append(prod)
	#print(Qvec)
	#docs vectors
	res = {}
	for filename in os.listdir(path):
		dvec = []
		for word in Qdict:
			tf = 0
			if word in tfdict[filename]:
				tf = tfdict[filename][word]
				#print("TEST")
				#print(filename)
			else:
				tf=0
			idf=0
			if word in idfDict:
				idf = idfDict[word]
			else:
				idf = 0
			prod = tf*idf
			dvec.append(prod)
		#print(dvec)
		val = cosine_sim(Qvec,dvec)
		res[filename]=val
	res = sorted(res.items(), key=operator.itemgetter(1),reverse=True)
	res = dict(res)
	##top 10 only
	i = 1
	for k in res:
		if i<=10:
			if math.isnan(res[k])==False:
				print(k)
				i = i + 1
		else:
			break



