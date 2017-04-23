f = open("vocab_embedding_5.txt",'r')
f1 = open("new_vocab_embedding_5.txt",'w')

for line in f:
	if line!="\n":
		f1.write(line)

f.close()
f1.close()