#!/usr/bin/python

# Simple program for quizzing GRE vocab

import random
import os

if __name__ == "__main__":
	os.system("clear")
	
	definitions = open('./definitions1.csv', 'r') # load GRE words from .csv file
	
	# We want to split the words by part of speech so that when quizzing, the quizzee cannot guess the answer based on the part of speech of the word in question
	nouns = dict() 
	verbs = dict()
	adjs = dict()
	
	for defn in definitions: # Separate words by part of speech
		x=defn.replace('\x00', '').split("\t")
		if len(x)==3:
			if ( x[1] == "n"):
				nouns[x[0]] = x[2][:-1]
			elif ( x[1] == "v"):
				verbs[x[0]] = x[2][:-1]
			elif ( x[1] == "adj"):
				adjs[x[0]] = x[2][:-1]
			else:
				print "Failed to load entry: "+str(x)
	
	for i in xrange(50): # Ask questions
		# pick which part of speech of the word we will quiz on 
		# Side effect is that we will ask approximately teh same amount of questions on each part of speech
		# This differs from just picking words at random since the number of words for each part of speech may nto be equal
		partOfSpeech = [nouns, verbs, adjs][random.randint(0,1)] 
		word = random.choice(partOfSpeech.keys()) # pick word
		answer = random.randint(1,4) # Which choice will be the right answer
		
		print word+":"
		for j in xrange(1,5):
			print j,": ",
			if (j != answer):
				print random.choice(partOfSpeech.values())
			else: 
				print partOfSpeech[word]
		print ""
		while ( raw_input('Guess: ')[0] != str(answer) ):
			print "Try Again..."
		print "\nCorrect!\n"+word+": "+partOfSpeech[word]+"\n\n\n"


