#import tensorflow as tf
import numpy as np
#load files
#working directory needs to be the same as the txts

with open("pl.txt") as io:
	polish_words=io.read().split("\n")
with open("en.txt") as io:
	english_words=io.read().split("\n")


def word_to_vec(word):
	#gets word and spits out a 15 by 26 tf vector
	vec_np=np.zeros((15,26))
	alphabet="abcdefghijklmnopqrstuvwxyz"
	#fuck itertools lol
	i=0
	for letter in word:
		vec_np[i][alphabet.find(letter)]=1
		i+=1
	vec_tf=tf.convert_to_tensor(vec_np, np.float32)
	return vec_tf
#give tf alternating and random words from the dictionaries
#pooling to check for patterns

