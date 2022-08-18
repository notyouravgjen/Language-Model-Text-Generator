# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
@author: notyouravgjen
"""

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
    
# use a language model to generate text
def generate_text(tokenized_text):
    n = 3
    train_data, padded_sents = padded_everygram_pipeline(n, tokenized_text)
    model = MLE(n)
    model.fit(train_data, padded_sents)
    
    seed = random.randint(0,10)    
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for token in model.generate(3000, random_seed=seed):
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
            
def la_place_generate(tokenized_text, tokens2d):
    vocab, training_text = create_vocab_and_training_data(2, tokenized_text, tokens2d)
    model = Laplace(2, vocabulary=vocab)
    model.fit(training_text)
    
    detokenize = TreebankWordDetokenizer().detokenize
    content = []
    for token in model.generate(3000):
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
    f1 = open('data/ep1.trn', 'r')
    f2 = open('data/ep2.trn', 'r')
    f3 = open('data/ep3.trn', 'r')
    f4 = open('data/ep4.trn', 'r')
    f5 = open('data/ep5.trn', 'r')
    f6 = open('data/ep6.trn', 'r')
    
    text1 = f1.read()
    text2 = f2.read()
    text3 = f3.read()
    text4 = f4.read()
    text5 = f5.read()
    text6 = f6.read()
    
    tokens1 = tokenize_lower([text1])
    tokens2 = tokenize_lower([text2])
    tokens3 = tokenize_lower([text3])
    tokens4 = tokenize_lower([text4])
    tokens5 = tokenize_lower([text5])
    tokens6 = tokenize_lower([text6])
    
    # 2d list of all tokens
    tokens_2d = [tokens1, tokens2, tokens3, tokens4, tokens5, tokens6]
    
    # 1d list of all tokens
    tokens = []
    tokens.extend(tokens1)
    tokens.extend(tokens2)
    tokens.extend(tokens3)
    tokens.extend(tokens4)
    tokens.extend(tokens5)
    tokens.extend(tokens6)
    
    # Use MLE
    #generate_text(tokens_2d)
    
    # Use LaPlace
    tokens.extend(b_train)
    tokens.extend(r_train)
    la_place_generate(tokens, tokens_2d)
    
if __name__ == "__main__":
    main(sys.argv[1:])

    