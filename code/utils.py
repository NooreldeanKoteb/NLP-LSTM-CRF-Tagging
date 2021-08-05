from collections import Counter

import json
import fasttext.util

def read_data(f, sub=0, train_words=None):
    with open(f, 'r', encoding='utf-8') as inp:
        lines = inp.readlines()
    if sub == 0:
        words = []
        if train_words!= None:
            for key in train_words.keys():
                words.extend(train_words[key])
        
        data = []
        for line in lines:
            line = line.strip().split()
            sentence = []
            for token in line:
                token = token.split('|')
                
                if train_words == None:
                    word = token[0]
                else:
                    if token[0] not in words:
                        word = 'UNK'
                    else:
                        word = token[0]
                        
                tag = token[1]
                sentence.append((word,tag))
            data.append(sentence)
    else:
        words = []
        for line in lines:
            line = line.strip().split()

            for token in line:
                words.append(token.split('|')[0])

        count = dict(Counter(words))
        
        words = {}
        for key, value in count.items():
            if value in words:
                words[value].append(key)
            else:
                words[value] = [key]
        
        data = substitute_with_UNK((words, lines), train_words=train_words, n=sub)

    return data

def convert_data_for_training(data):
    #for d in data:
    #    tokens = [t[0] for t in d]
    #    tags = [t[1] for t in d]
    return [([t[0] for t in d],[t[1] for t in d]) for d in data]

def substitute_with_UNK(data, train_words=None, n=1):
    words = []
    if train_words == None:
        f = open('../models/words.json','w')
        json.dump(data[0], f)
        f.close()
        
        for i in range(1, n+1):
            try:
                words.extend(data[0][i])
            except:
                continue
    else:
        for key in train_words.keys():
            if int(key) > n:
                words.extend(train_words[key])

    results = []
    for line in data[1]:
        line = line.strip().split()
        sentence = []
        for token in line:
            token = token.split('|')
            
            if train_words == None:
                if token[0] in words:
                    word = 'UNK'
                else:
                    word = token[0]
                    
            else:
                if token[0] not in words:
                    word = 'UNK'
                else:
                    word = token[0]
            
            tag = token[1]
            sentence.append((word,tag))
        results.append(sentence)
        

    return results


def grab_word_vecs(lang, files):
    vecs = []
    if lang == 'irish':
        ft = fasttext.load_model('cc.ga.300.bin')
    elif lang == 'welsh':
        ft = fasttext.load_model('cc.cy.300.bin')
    else:
        return print('This language is not supported')
    
    words = []
    for file in files:
        with open(file, 'r', encoding='utf-8') as inp:
            lines = inp.readlines()
                    
        for line in lines:
            line = line.strip().split()
            for token in line:
                token = token.split('|')
                words.append(token[0])

    for word in words:
        vecs.append(ft.get_word_vector(word))

    return vecs


def con_files(name, files):
    data = ''
    for file in files:
        with open(file, 'r', encoding='utf-8') as inp:
            data += inp.read()
        data += '\n'
        
    with open(f'..\data\{name}.train', 'w', encoding="utf-8") as fp:
        fp.write(data)
        
    fp.close()
