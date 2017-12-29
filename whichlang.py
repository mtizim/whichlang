#mtizim 2017-12-29
#Guesses if a word is from a language
#Inputs-training for training the model and word which provides a word to make the guess in
#This should work for any two languages,#provided both files only contain words
# of max length 15 (shortEN.py)and only contain letters from the english alphabet
#obviously needs source ~/tensorflow/bin/activate
import tensorflow as tf
import numpy as np
import random as rd
rd.seed()
#constants (kinda shitty lel)
ABC_LEN=15
MAX_WORD_LEN=26
RANGE=1000

training=bool(int(input("Training?(1/0)")))
#load files
#working directory needs to be the same as the txts
with open("pl.txt") as io:
	polish_words=io.read().split("\n")
with open("en.txt") as io:
	english_words=io.read().split("\n")

#weight initialization for ReLU
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def word_to_vec(word):
	#gets word and spits out a 15 by 26 np vector
	vec_np=np.zeros((ABC_LEN,MAX_WORD_LEN))
	alphabet="abcdefghijklmnopqrstuvwxyz"
	#fuck itertools lol
	i=0
	for letter in word:
		vec_np[i][alphabet.find(letter)]=1
		i+=1
	vec_np=vec_np.reshape(ABC_LEN*MAX_WORD_LEN)
	return vec_np

def get_data(datapoints):
	#returns a random batch of 100 categorized words
	data=[]
	labels=[]
	for i in range(datapoints):
		which=rd.randint(0,1) 
		#0 means en
		#1 means pl
		if which:
			data.append(word_to_vec( polish_words[rd.randint( 0 , len(polish_words)-1 )]) )
			labels.append([0,1]) #01=pl
		else:
			data.append(word_to_vec( english_words[rd.randint( 0 , len(english_words)-1 )]) )
			labels.append([1,0]) #10=en
	return data,labels



#give tf alternating and random words from the dictionaries
#pooling to check for patterns

#making tf's graph
	#1st layer input
x=tf.placeholder(tf.float32,[None,ABC_LEN*MAX_WORD_LEN])
w1=weight_variable([390,40])
b1=bias_variable([40])
	#2nd layer 1st hidden
x1=tf.nn.relu(tf.matmul(x,w1) + b1)
w2=weight_variable([40,5])
b2=bias_variable([5])
	#3rd layer 2nd hidden
x2=tf.nn.relu(tf.matmul(x1,w2) + b2)
w3=weight_variable([5,2])
b3=bias_variable([2])
	#4th layer output
y=tf.nn.softmax(tf.matmul(x2,w3) + b3)
	#answers and training
y_correct=tf.placeholder(tf.float32,[None,2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_correct * tf.log(y), reduction_indices=[1]))
saver = tf.train.Saver()
#training 
if training:
	with tf.Session() as sess:	
		train_step = tf.train.MomentumOptimizer(0.03,0.85).minimize(cross_entropy)
		tf.global_variables_initializer().run()
		for i in range(RANGE):
			data,labels=get_data(10)

			sess.run(train_step, feed_dict={x: data, y_correct: labels})
			if i%(RANGE/10)==0:
				print(int(round(i/RANGE*100)),"%")
		save_path = saver.save(sess, "model.ckpt")
		print("Model saved: %s" % save_path)


#testing
if not training:
	with tf.Session() as sess:	
		saver.restore(sess, "model.ckpt")
		print("Restored")
		ioin=str(input())
		while ioin!="end":
			word=word_to_vec(ioin)
			fake_label=np.array([0,0])
			word.shape=(1,390)
			fake_label.shape=(1,2)
			print(sess.run(y, feed_dict={x: word,y_correct: fake_label}))
			ioin=str(input())