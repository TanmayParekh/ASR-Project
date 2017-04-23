import string

# Parse the embedding file. Just extract all the words
def get_entire_wordlist():
	
	in_file = open("SBW-vectors-300-min5.txt",'r')
	wordList = []

	for line in in_file:
		word = line.split()[0]
		wordList.append(word)

	in_file.close()
	out_file = open("wordlist.txt",'w')

	for word in wordList:
		out_file.write(word + "\n")

	out_file.close()

# Extract the embedding wordlist
def extract_wordlist():

	file = open("wordlist.txt",'r')
	wordlist = []

	for line in file:
		word = line.split()[0]
		wordlist.append(word)

	file.close()
	return wordlist

################################################################

wordList = extract_wordlist()

# Remove the various punctuations from the text
def remove_punc(text):
    return text.translate(None, string.punctuation)

# Search if word present in embedding wordlist
def search(new_word):

	if new_word in wordList:
		return 1
	else:
		return 0

# Check if the words in the document are present in the embedding word list
def check_vocab():

	vocab = open("vocab.csv",'r')
	vocab_list = []
	vocab_present_list = []

	for line in vocab:
		vocab_list.append(line.split(',')[0])

	vocab.close()

	for word in vocab_list:
		is_present = search(remove_punc(word))
		vocab_present_list.append(is_present)

	out_file = open("isPresent.txt",'w')

	for i in range(len(vocab_list)):
		out_file.write(vocab_list[i] + "," + str(vocab_present_list[i]) + "\n")

	out_file.close()

###############################################################

# Searches and returns the feature vector corresponding to the word
def extract_feature(word):

	feature = ""

	word = remove_punc(word)
	if word in wordList:
		in_file = open("SBW-vectors-300-min5.txt",'r')

		for line in in_file:
			w1 = line.split()[0]
			if word == w1:
				feature = line

		in_file.close()

	return feature

def make_feature():

	vocab = open("extracted_vocab_5.txt",'r')
	vocab_list = []

	for line in vocab:
		if line != "\n":
			vocab_list.append(line.split()[0])

	vocab.close()

	feature_file = open("vocab_embedding_5.txt",'w')

	for word in vocab_list:
		feature = extract_feature(word)
		if feature != "":
			feature_file.write(feature + "\n")
		else:
			feature_file.write(word + "\n")

	feature_file.close()

make_feature()