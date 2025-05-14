import json
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import string

#Download tokenizer if not already present
nltk.download('punkt')

stemmer = PorterStemmer()

#1. Tokenize a sentence into words
def tokenize(sentence):
    return nltk.word_tokenize(sentence.lower())

#2. Stem a word
def stem(word):
    return stemmer.stem(word)

#3. Create a bag of words
#   - 0 if the word does not exist in the sentence
#   - 1 if the word exists in the sentence
#   - 2 if the word exists in the sentence more than once
def bag_of_words(tokenized_sentence, vocab):
    sentence_words = [stem(word) for word in  tokenized_sentence]
    bag = np.zeros(len(vocab), dtype=np.float32)
    for idx, word in enumerate(vocab):
        if word in sentence_words:
            bag[idx] = 1.0
    return bag

#4. Load and preprocess data from intents.json
def preprocess_intents(json_data):
    all_words = []
    tags = []
    xy = []
    
    for intent in json_data['intents']:
        tag = intent["tag"]
        tags.append(tag)
        for pattern in intent["patterns"]:
            # Tokenize each word in the sentence
            word_list = tokenize(pattern)
            all_words.extend(word_list)
            # Add the sentence and its corresponding tag to xy
            xy.append((word_list, tag))
            
    #Remove punctuation and stem the words
    ignore_chars = set(string.punctuation)
    all_words = [stem(word) for word in all_words if word not in ignore_chars]
    all_words = sorted(set(all_words))
    tags = sorted(set(tags))
    
    #Create training data
    X_train = []
    y_train = []
    
    for (tokenized_sentence, tag) in xy:
        bag = bag_of_words(tokenized_sentence, all_words)
        X_train.append(bag)
        y_train.append(tags.index(tag))
     
    return np.array(X_train), np.array(y_train), all_words, tags   
    

