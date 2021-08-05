from collections import Counter

from itertools import chain
import nltk
import sklearn
import scipy.stats

import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics
from sklearn.metrics import make_scorer

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import utils
import pickle

def word2features(sent, i, next_w=False, Q2=False, Q3=False):
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.isdigit()': word.isdigit(),
    }
    
    if Q2 == True:
        features.update({
            'last': word[-1],
            'last_two': word[-2:],
            'last_three': word[-3:],
            })
    if Q3 == True:
        features.update({
            'first': word[0],
            'first_two': word[:2],
            'first_three': word[:3],
            })
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True

    if next_w == True:
        if i < len(sent)-1:
            word2 = sent[i+1]
            features.update({
                '+1:word.lower()': word2.lower(),
                '+1:word.isupper()': word2.isupper(),
                })
        else:
            features['EOS'] = True
    

    return features


def sent2features(sent, Q=1):
    if Q == 2:
        return [word2features(sent, i, next_w=True, Q2=True) for i in range(len(sent))]
    elif Q == 3:
        return [word2features(sent, i, next_w=True, Q2=True, Q3=True) for i in range(len(sent))]

    return [word2features(sent, i) for i in range(len(sent))]

def print_transitions(trans_features):
    for (label_from, label_to), weight in trans_features:
        print("\t%-6s -> %-7s %0.6f" % (label_from, label_to, weight))

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("\t%0.6f %-8s %s" % (weight, label, attr))

def start(train_data=None, test_data=None, save=None, load=None, Q=1):
    try:
        if train_data != None:
            TRAINING_FILE = train_data
            training_data = utils.convert_data_for_training(utils.read_data(TRAINING_FILE))
            # The input will be the features for each token in each sentence
            X_train = [sent2features(s[0], Q) for s in training_data]
            # The outputs will be the tags for each token in each sentence
            y_train = [s[1] for s in training_data]
    
        if test_data != None:
            #Similarly, read the test data
            TEST_FILE = test_data
            test_data = utils.convert_data_for_training(utils.read_data(TEST_FILE))
            X_test = [sent2features(s[0], Q) for s in test_data]
            y_test = [s[1] for s in test_data]
    except:
        print('Failed to load data!')
        return

    print("Finished reading the files")

    if train_data != None:
        # Training
        crf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        print("Starting training...")
        crf.fit(X_train, y_train)
        print("Finished training...")

        if save != None:
            # save
            with open('../models/'+save+'.pkl','wb') as f:
                pickle.dump(crf,f)


    if test_data != None:

        if load != None:
            # load
            try:
                with open(load+'.pkl', 'rb') as f:
                    crf = pickle.load(f)
            except:
                print('Failed to load model!')
                return

        # Evaluation
        print("Starting Predictions on test...")
        labels = list(crf.classes_)
        y_pred = crf.predict(X_test)
        #y_pred[:2]

        print(f"Evaluation... Accuracy: {metrics.flat_accuracy_score(y_test, y_pred)}")
        
        # print("Let's check what the the classifier learned!")

        # print("Top likely transitions:")
        # print_transitions(Counter(crf.transition_features_).most_common(20))
        
        # print("\nTop unlikely transitions:")
        # print_transitions(Counter(crf.transition_features_).most_common()[-20:])
        
        
        # print("Top positive features:")
        # print_state_features(Counter(crf.state_features_).most_common(50))
        
        # print("\nTop negative features:")
        # print_state_features(Counter(crf.state_features_).most_common()[-30:])



##########
# Part 3 #
##########
# #Q1
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='crf1', load=None)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/crf1')

# #Q2
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='crf2', load=None, Q=2)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/crf2', Q=2)

# #Q3 - Bonus
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='crf3', load=None, Q=3)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/crf3', Q=3)

##################
# Part 4 - Bonus #
##################
# #Q1
# #Creating new concatenated data file #Run once
# # files = ['..\data\irish.train', '..\data\welsh.train']
# # utils.con_files('irish-welsh', files)

# #Run Training
# start(train_data="../data/irish-welsh.train", test_data=None, save='crf4', load=None, Q=3)
# #Run Prediction
# print('Irish Test Set:')
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/crf4', Q=3)
# print('Welsh Dev Set:')
# start(train_data=None, test_data="../data/welsh.dev", save=None, load='../models/crf4', Q=3)


# #Q2








