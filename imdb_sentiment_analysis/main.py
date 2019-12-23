#!/usr/bin/python3 -O

"""
https://medium.com/@bhadreshpsavani/tutorial-on-sentimental-analysis-using-pytorch-b1431306a2d7
"""

import numpy as np
from string import punctuation
from collections import Counter 

with open('data/reviews.txt', 'r') as f:
    reviews = f.readlines()
with open('data/labels.txt', 'r') as f:
    labels = f.readlines()

def main():
    for review, label in list(zip(reviews, labels))[:5]:
        print("review {}".format(review))
        print("label {}".format(label))
        print()
    print("punctuation {}".format(punctuation))
    all_reviews = []
    for review in reviews:
        review = review.lower()
        review = "".join([character for character in review if character not in punctuation])
        all_reviews.append(review)
    all_text = " ".join(all_reviews)
    all_words = all_text.split()
    count_words = Counter(all_words)
    total_words = len(all_words)
    sorted_words = count_words.most_common(total_words)
    print("total_words {}".format(total_words))
    print("list(sorted_words)[:5] {}".format(list(sorted_words)[:5]))
    vocab_to_int = {w:i+1 for i,(w,c) in enumerate(sorted_words)}
    encoded_reviews = list()
    for review in all_reviews:
        encoded_review = list()
        for word in review.split():
            if word not in vocab_to_int.keys():
                encoded_review.append(0)
            else:
                encoded_review.append(vocab_to_int[word])
        encoded_reviews.append(encoded_review)
    print("list(encoded_reviews)[:5] {}".format(list(encoded_reviews)[:5]))
    sequence_length=250
    features=np.zeros((len(encoded_reviews), sequence_length), dtype=int)
    for i, review in enumerate(encoded_reviews):
        review_len=len(review)
        if (review_len<=sequence_length):
            zeros=list(np.zeros(sequence_length-review_len))
            new=zeros+review
        else:
            new=review[:sequence_length]
        features[i,:]=np.array(new)
    print("features[:3,:] {}".format(features[:3,:]))
    labels = [1 if label.strip()=='positive' else 0 for label in labels]
    #split_dataset into 80% training , 10% test and 10% Validation Dataset
    train_x=features[:int(0.8*len(features))]
    train_y=labels[:int(0.8*len(features))]
    valid_x=features[int(0.8*len(features)):int(0.9*len(features))]
    valid_y=labels[int(0.8*len(features)):int(0.9*len(features))]
    test_x=features[int(0.9*len(features)):]
    test_y=labels[int(0.9*len(features)):]
    print(len(train_y), len(valid_y), len(test_y))


if __name__ == '__main__':
    main()
