# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author: notyouravgjen
"""

import os
import sys
import nltk
import re
import random
from nltk.lm import MLE
from nltk.lm import Laplace
from nltk.lm import Vocabulary
from nltk.lm.preprocessing import padded_everygrams
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.tokenize.treebank import TreebankWordDetokenizer

# split strings into lowercase tokens
def tokenize_lower(string_list):    
    wordlist = []
    tokenlist = []
    for each_str in string_list:
        tokens = nltk.word_tokenize(each_str)

        for token in tokens:
            # removes empty strings
            if token:
                # Build a list of all tokens
                tokenlist.append(token.lower())
                # Build a list of alphanumeric-only tokens
                if re.search(r"[a-z0-9]", token.lower()):
                    wordlist.append(token.lower())
    return tokenlist
    
# use MLE language model to generate text
def mle_generate(tokenized_text, word_count):
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_sents)
    
    seed = random.randint(0,10)    
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for token in model.generate(word_count, random_seed=seed):
        if token == '<s>':
            continue
        if token == '</s>':
            break
        content.append(token)
    
    text = detokenize(content)
    for text_char in text:
        if (text_char == '.'):
            print (text_char)
        else:
            print (text_char, end = '')
            
def create_vocab_and_training_data(ngram_order, words, tokens2d):
    return (
        Vocabulary(words, unk_cutoff=1),
        [
            padded_everygrams(ngram_order, sent)
            for sent in (tokens2d)
        ],
    )
            
def la_place_generate(tokenized_text, tokens2d, word_count):
    vocab, training_text = create_vocab_and_training_data(2, tokenized_text, tokens2d)
    model = Laplace(2, vocabulary=vocab)
    model.fit(training_text)
    
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for token in model.generate(word_count):
        if token == '<UNK>':
            continue
        content.append(token)
    
    text = detokenize(content)
    for text_char in text:
        if (text_char == '.'):
            print (text_char)
        else:
            print (text_char, end = '')

# Program begins here
def main(argv):    
    brown = nltk.corpus.brown
    b_tokens = brown.words()
    b_train = b_tokens[:1000]
    
    reuters = nltk.corpus.reuters
    r_tokens = reuters.words()
    r_train = r_tokens[:1000]
    
    #Open training data
    files = os.listdir('data')
    
    # 1d list of all tokens
    tokens_1d = []
    
    # 2d list of all tokens
    tokens_2d = []
    
    total_word_count = 0
    
    for file_name in files:
        file = open('data/'+file_name, 'r')
        text = file.read()
        tokens = tokenize_lower([text])
        tokens_1d.extend(tokens)
        tokens_2d.append(tokens)
        total_word_count += len(tokens)
 
    avg_word_count = int(total_word_count / len(files))
 
    # Use MLE
    #mle_generate(tokens_2d, avg_word_count)
    
    # Use LaPlace
    tokens.extend(b_train)
    tokens.extend(r_train)
    la_place_generate(tokens_1d, tokens_2d, avg_word_count)
    
if __name__ == "__main__":
    main(sys.argv[1:])

    