#import tensorflow as tf

#load files
#working directory needs to be the same as the txts

with open("pl.txt") as io:
	polish_words=io.read().split("\n")
with open("en.txt") as io:
	english_words=io.read().split("\n")


#give tf alternating and random words from the dictionaries
#pooling to check for patterns

