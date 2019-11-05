#!/usr/bin/python3

"""

Based on tutorial at https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#sequence-models-and-long-short-term-memory-networks

Owner : paul-tqh-nguyen

Created : 10/30/2019

File Name : pos_tagger.py

File Organization:
* Imports
* Main Runner

"""

###########
# Imports #
###########

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math

###############
# Main Runner #
###############

class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        print("within forward")
        print("sentence: {sentence}".format(sentence=sentence))
        embeds = self.word_embeddings(sentence)
        print("embeds.shape: {x}".format(x=embeds.shape))
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        print("lstm_out.shape: {x}".format(x=lstm_out.shape))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        print("forward done")
        return tag_scores

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

def tag_score_tensor_to_pos(tag_score_tensor, ix_to_tag):
    best_score_so_far = -math.inf
    best_tag_so_far = None
    for ix, tag_score in enumerate(tag_score_tensor):
        if tag_score>best_score_so_far:
            best_score_so_far = tag_score
            best_tag_so_far = ix_to_tag[ix]
    return best_tag_so_far

def main():
    training_data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_ix = {}
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    
    tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
    ix_to_tag = {v:k for k,v in tag_to_ix.items()}
    
    EMBEDDING_DIM = 12
    HIDDEN_DIM = 12
    
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(300):
        for sentence, tags in training_data:
            print("\n\n\n")

            model.zero_grad()

            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
        
            tag_scores = model(sentence_in)

            print("sentence: {sentence}".format(sentence=sentence))
            print("sentence_in: {sentence_in}".format(sentence_in=sentence_in))
            print("targets: {targets}".format(targets=targets))
            print("tag_scores: {tag_scores}".format(tag_scores=tag_scores))
            
            
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
    
    with torch.no_grad():
        
        input_example = training_data[0][0]
        
        print("Input sequence: {input}".format(input=input_example))
        
        inputs = prepare_sequence(input_example, word_to_ix)
        tag_score_tensors = model(inputs)
        
        tag_score_tensor_to_pos_for_ix_to_tag = lambda tag_score_tensor: tag_score_tensor_to_pos(tag_score_tensor, ix_to_tag)
        pos_list = map(tag_score_tensor_to_pos_for_ix_to_tag, tag_score_tensors)
        word_pos_doubletons = zip(input_example, pos_list)
        for word, pos in word_pos_doubletons:
            print("{word} has POS {pos}".format(word=word, pos=pos))
    
    return None

if __name__ == '__main__':
    main()
