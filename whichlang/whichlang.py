#import tensorflow as tf

#load files
with open("en.txt") as o:
	english_words=o.read().split("\n")
with open("pl.txt") as o:
	polish_words=o.read().split("\n")
#remove words which are too long from the english dic
#max length is 15
i=0
i_list=[]
for word in english_words:
	if len(word)>15: 
		i_list.append(i)
	i+=1
i=0
for num in i_list:
	del english_words[num-i]
	i+=1

#give tf alternating and random words from the dictionaries
#pooling to check for patterns

