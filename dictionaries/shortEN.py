with open("en.txt") as io:
	english_words=io.read().split("\n")
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
with open("en.txt",'w') as io:
	for word in english_words:
		io.write(word+"\n")

	