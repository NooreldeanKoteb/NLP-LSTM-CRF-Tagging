import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import utils
import compute_accuracy as score

torch.manual_seed(42)

import json
import datetime as dt
from torch import argmax
import matplotlib.pyplot as plt
from itertools import product
import fasttext.util

def use_gpu():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    print('Using device:', device)
    print()

    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
        print()
    
    return device, dtype

def prepare_sequence(seq, to_ix, device):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long, device=device)

class LSTMTagger(nn.Module):
    # Class that defines our model
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,  num_layers, 
                 bidirectional, lang=None, files=None):
        
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        if lang != None:
            print(f'loading {lang} pretrained embedings!')
            pretrain_vecs = torch.tensor(utils.grab_word_vecs(lang, files), device=device)
            self.word_embeddings = nn.Embedding.from_pretrained(pretrain_vecs)
        else:
            self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional)
        
        if bidirectional==True:
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim*2, tagset_size)
        else:
            # The linear layer that maps from hidden state space to tag space
            self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    # This is the forward computation, which constructs the computation graph
    def forward(self, sentence):
        # Get the embeddings
        embeds = self.word_embeddings(sentence)
        # put them through the LSTM and get its output
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # pass that output through the linnear layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # convert the logits to a log probability distribution
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

def start(train_data=None, test_data=None, dev_data=None, patience=None,  save=None, load=None, 
          sub=0, device=None, misc=None, epochs=20, bidirectional=False, EMBEDDING_DIM = 32,
          HIDDEN_DIM = 32, LAYERS = 1, optimize_fn='SGD', lang=None, files=None):
    try:
        # This is an example of the way to read the data for the assignment
        if train_data != None:
            training_data = utils.convert_data_for_training(utils.read_data(train_data, sub))
            if misc == 'Q2':
                training_data.extend(utils.convert_data_for_training(utils.read_data(train_data, sub=0)))
        # First, read in the test file
        if test_data != None:
            if sub > 0:
                with open('../models/words.json','r') as f:
                    data = json.load(f)
            else:
                data  = None
                
            test_data = utils.convert_data_for_training(utils.read_data(test_data, sub, train_words=data))
            
            if dev_data != None:
                dev = dev_data
                dev_data = utils.convert_data_for_training(utils.read_data(dev_data, sub, train_words=data))

    except:
        print('Failed to load data!')
        return


    if load == None:
        # Get token and tag vocabularies from the training set, and map them to integer IDs
        word_to_ix = {}
        ix_to_word = {}
        tag_to_ix = {}
        ix_to_tag = {}
        for sent, tags in training_data:
            for word in sent:
                if word not in word_to_ix:
                    word_to_ix[word] = len(word_to_ix)
                    ix_to_word[word_to_ix[str(word)]] = word
            for tag in tags:
                if tag not in tag_to_ix:
                    tag_to_ix[tag] = len(tag_to_ix)
                    ix_to_tag[tag_to_ix[str(tag)]] = tag


        # Hyperparameters
        EMBEDDING_DIM = EMBEDDING_DIM
        HIDDEN_DIM = HIDDEN_DIM
        LAYERS = LAYERS

        params = {'EMBEDDING_DIM': EMBEDDING_DIM,
                  'HIDDEN_DIM': HIDDEN_DIM,
                  'word_to_ix': word_to_ix,
                  'ix_to_word': ix_to_word,
                  'tag_to_ix': tag_to_ix,
                  'ix_to_tag': ix_to_tag,
                  'LAYERS': LAYERS,
                  'bidirectional': bidirectional,
                  'lang': lang,
                  'files': files,}

        # Initialize the model
        model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), LAYERS, bidirectional, lang, files)

        if device.type == 'cuda':
            model.cuda()
    else:
        try:
            # Initialize the model
            params = []
            with open(load+'.json','r') as f:
                data = json.load(f)
                EMBEDDING_DIM = data['EMBEDDING_DIM']
                HIDDEN_DIM = data['HIDDEN_DIM']
                word_to_ix = data['word_to_ix']
                ix_to_word = data['ix_to_word']
                tag_to_ix = data['tag_to_ix']
                ix_to_tag = data['ix_to_tag']
                LAYERS = data['LAYERS']
                bidirectional = data['bidirectional']
                lang = data['lang']
                files = data['files']

            model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), LAYERS, bidirectional, lang, files)
            model.load_state_dict(torch.load(load+'.pt'))
            model.eval()
            if device.type == 'cuda':
                model.cuda()
        except:
            print('Failed to load model!')
            return



#Training

    if train_data != None:
        # Loss function to use
        loss_function = nn.NLLLoss()
        if optimize_fn == 'Adam':
            # Optimizer to use during training
            optimizer = optim.Adam(model.parameters())
        else:
            # Optimizer to use during training
            optimizer = optim.SGD(model.parameters(), lr=0.1)
            
        # See what the scores are before training
        # Note that element i,j of the output is the score for tag j for word i.
        # Here we don't need to train, so the code is wrapped in torch.no_grad()
        with torch.no_grad():
            inputs = prepare_sequence(training_data[0][0], word_to_ix, device)
            tag_scores = model(inputs)
            print(tag_scores)
            for i,word in enumerate(training_data[0][0]):
                if device.type == 'cuda':
                    j = int(argmax(tag_scores[i]))
                else:
                    j = int(np.argmax(tag_scores[i]))

                print(f"\t{word}|{ix_to_tag[j]}")

        train_loss_track = []
        dev_loss_track = []
        dev_model = []
        # Training loop
        for epoch in range(epochs):  # normally you would NOT do 100 epochs, it is toy data
            if epoch != 0:
                print('\repoch: '+str(epoch)+' - 100.00%', end='', flush=True)
            
            if dev_data != None:
                if epoch%patience == 0:
                    dev_results = []
                    with torch.no_grad():
                        for instance in dev_data:
                            # Convert the dev sentence into a word ID tensor
                            dev_inputs = prepare_sequence(instance[0], word_to_ix, device)
                            
                            dev_targets = prepare_sequence(instance[1], tag_to_ix, device)
                            
                            # Forward pass
                            dev_tag_scores = model(dev_inputs)
                            
                            loss = loss_function(dev_tag_scores, dev_targets)
                            dev_loss_track.append((epoch, float(loss.data)))
                            
                            # Find the tag with the highest probability in each position
                            if device.type == 'cuda':
                                outputs = [int(argmax(ts)) for ts in dev_tag_scores]
                            else:
                                outputs = [int(np.argmax(ts)) for ts in dev_tag_scores]
                            

                            # Prepare the output to be written in the same format as the test file (word|tag)
                            formatted_output = ' '.join([f"{word}|{ix_to_tag[tag_id]}" for word,tag_id in zip(instance[0],outputs)])
                            # Save the output
                            dev_results.append(formatted_output)
                    
                    acc = score.start(output=None, dev=dev_results, gold=dev, verbose=False)
                    if epoch == 0:
                        print(f'\nDevelopment Set Accuracy before training: {acc}')
                    else:
                        print(f'\nDevelopment Set Accuracy after epoch {epoch}: {acc}')
                    
                    if dev_model != []: 
                        if acc < dev_model[-1][1]:
                            print('\nNo further Improvement. Taking last best model!')
                            model = dev_model[-1][0]
                            break
                        
                    dev_model.append((model, acc))
                    

                    
                                  
            
            print(f"\nStarting epoch {epoch+1}...")
            i = 0
            num_data = len(training_data)
            t = dt.datetime.now()
            for sentence, tags in training_data:
                i +=1
                if (dt.datetime.now() -t).seconds / 10 >= 1:
                    t = dt.datetime.now()
                    percent = i/num_data
                    print('\repoch: '+str(epoch+1)+' - {:.2%}'.format(percent), end='', flush=True)
                
                
                 # Step 1. Remember that Pytorch accumulates gradients.
                # We need to clear them out before each instance
                model.zero_grad()

                # Step 2. Get our inputs ready for the network, that is, turn them into
                # Tensors of word indices.
                # Eventually I suggest you use the DataLoader modules
                # The batching can take place here
                sentence_in = prepare_sequence(sentence, word_to_ix, device)
                targets = prepare_sequence(tags, tag_to_ix, device)
                
                # Step 3. Run our forward pass.
                tag_scores = model(sentence_in)

                # Step 4. Compute the loss, gradients, and update the parameters by
                #  calling optimizer.step()
                loss = loss_function(tag_scores, targets)
               
                train_loss_track.append((epoch, float(loss.data)))
                
                loss.backward()
                optimizer.step()

        print('\repoch: '+str(epochs)+' - 100.00%\n', end='', flush=True)
        

        # See what the scores are after training
        with torch.no_grad():
            inputs = prepare_sequence(training_data[0][0], word_to_ix, device)
            tag_scores = model(inputs)

            # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
            # for word i. The predicted tag is the maximum scoring tag.
            # Here, we can see the predicted sequence below is 0 1 2 0 1
            # since 0 is index of the maximum value of row 1,
            # 1 is the index of maximum value of row 2, etc.
            # Which is DET NOUN VERB DET NOUN, the correct sequence!
            print(tag_scores)
            # Print the actual words with their tags
            for i,word in enumerate(training_data[0][0]):
                if device.type == 'cuda':
                    j = int(argmax(tag_scores[i]))
                else:
                    j = int(np.argmax(tag_scores[i]))
                print(f"\t{word}|{ix_to_tag[j]}")


        if save != None:
            torch.save(model.state_dict(), '../models/'+save+'.pt')
            f = open('../models/'+save+'.json','w')
            json.dump(params, f)
            f.close()


#Testing

    if test_data != None and load!= None:
        # This snipet can be used to create outputs over the test set
        with torch.no_grad():
            # this will be the file to write the outputs
            with open('../results/'+str(load[10:])+"_output.txt", 'w') as op:
                for instance in test_data:
                    # Convert the test sentence into a word ID tensor
                    inputs = prepare_sequence(instance[0], word_to_ix, device)
                    # Forward pass
                    tag_scores = model(inputs)
                    # Find the tag with the highest probability in each position
                    if device.type == 'cuda':
                        outputs = [int(argmax(ts)) for ts in tag_scores]
                    else:
                        outputs = [int(np.argmax(ts)) for ts in tag_scores]

                    # Prepare the output to be written in the same format as the test file (word|tag)
                    formatted_output = ' '.join([f"{word}|{ix_to_tag[str(tag_id)]}" for word,tag_id in zip(instance[0],outputs)])
                    # Write the output
                    op.write(formatted_output + '\n')
        print('Done!')


#Will attempt to switch to gpu
device, dtype  = use_gpu()

##########
# Part 1 #
##########
# #Q1
# print('Part A')
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='lstm-cuda-unk', load=None, sub=1, device=device)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/lstm-cuda-unk', sub=1, device=device)
# #Accuracy
# score.start(output='../results/lstm-cuda-unk_output.txt', gold='../data/irish.test')

# #Q2
# print('Part B')
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='lstm-cuda-unk-Q2', load=None, sub=1, device=device, misc='Q2')
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/lstm-cuda-unk-Q2', sub=1, device=device)
# #Accuracy
# score.start(output='../results/lstm-cuda-unk-Q2_output.txt', gold='../data/irish.test')

# #Q3
# print('Part C')
# #Run Training
# start(train_data="../data/irish.train", test_data="../data/irish.test", 
#       dev_data="../data/irish.dev", patience=6, save='lstm-cuda-unk-early', 
#       load=None, sub=1, device=device, epochs=100)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, 
#       load='../models/lstm-cuda-unk-early', sub=1, device=device)
# #Accuracy
# score.start(output='../results/lstm-cuda-unk-early_output.txt', gold='../data/irish.test')

#======================================================================================================#

##########
# Part 2 #
##########
# #Q1
# print('Part A')
# #Run Training
# start(train_data="../data/irish.train", test_data=None, save='lstm-cuda-unk-bi', load=None, sub=1, device=device, bidirectional=True)
# #Run Prediction
# start(train_data=None, test_data="../data/irish.dev", save=None, load='../models/lstm-cuda-unk-bi', sub=1, device=device, bidirectional=True)
# #Accuracy
# score.start(output='../results/lstm-cuda-unk-bi_output.txt', gold='../data/irish.dev')

# #Q2
# #Hyper Parameter Testing
# params = {
#     'EMBEDDING_DIM': [512, 128, 64, 32], 
#     'HIDDEN_DIM': [512, 128, 64, 32], 
#     'LAYERS': [1, 2, 4, 8, 16], 
#     'optimize_fn': ['SGD', 'Adam']
#     }

# keys, values = zip(*params.items())
# combinations = []
# for val in product(*values):
#     combinations.append(dict(zip(keys, val)))

# results = {}
# for param in combinations:
#     print(param)
#     # Run Training
#     start(train_data="../data/irish.train", test_data=None, save='crossval/lstm-cuda-unk-bi', load=None, sub=1, 
#           device=device, bidirectional=True, EMBEDDING_DIM = param['EMBEDDING_DIM'], HIDDEN_DIM = param['HIDDEN_DIM'], 
#           LAYERS = param['LAYERS'], optimize_fn=param['optimize_fn'])
#     # Run Prediction
#     start(train_data=None, test_data="../data/irish.test", save=None, load='../models/crossval/lstm-cuda-unk-bi', sub=1, device=device, bidirectional=True)
#     # Accuracy
#     acc = score.start(output='../results/crossval/lstm-cuda-unk-bi_output.txt', gold='../data/irish.test')
    
#     if acc not in results.keys():
#         results[acc] = [param]
#     else:
#         results[acc].append(param)
        
# sorted_results = sorted(results.items(), reverse=True)
# f = open("sorted_results.txt","w")
# f.write(str(sorted_results))
# f.close()
################################
################################
# #Best Parameters from Hyperparameter testing
# #Run Training
# print('Part B')
# start(train_data="../data/irish.train", test_data="../data/irish.test", save='lstm-cuda-unk-bi-best', load=None, 
#       sub=1, device=device, bidirectional=True, EMBEDDING_DIM = 512, HIDDEN_DIM = 512, LAYERS = 1, 
#       optimize_fn='Adam', epochs=50, dev_data="../data/irish.dev", patience=6)
# # Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/lstm-cuda-unk-bi-best', sub=1, device=device, bidirectional=True)
# # Accuracy
# acc = score.start(output='../results/lstm-cuda-unk-bi-best_output.txt', gold='../data/irish.test')


# #Q3
# print('Part C')
# #Welsh
# fasttext.util.download_model('cy', if_exists='ignore')

# #Irish
# fasttext.util.download_model('ga', if_exists='ignore') 

# irish_files = ['../data/irish.train', '../data/irish.test', '../data/irish.dev']

# start(train_data="../data/irish.train", dev_data="../data/irish.dev", test_data="../data/irish.test", save='lstm-cuda-unk-bi-best-pretrain', load=None, sub=1, 
#         device=device, bidirectional=True, EMBEDDING_DIM = 300, HIDDEN_DIM = 300, LAYERS = 1, optimize_fn='Adam', epochs=50, patience=6,
#       lang='irish', files=irish_files)
# # Run Prediction
# start(train_data=None, test_data="../data/irish.test", save=None, load='../models/lstm-cuda-unk-bi-best-pretrain', 
#         sub=1, device=device, bidirectional=True, lang='irish', files=irish_files)
# # Accuracy
# acc = score.start(output='../results/lstm-cuda-unk-bi-best-pretrain_output.txt', gold='../data/irish.test')

#======================================================================================================#
