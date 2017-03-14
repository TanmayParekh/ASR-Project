import sys
import os
import operator

global_vocab = {}					# GLOBAL VOCABULARY LIST

# Find all csv files from the path
def find_csv_files(dir_path, extension=".csv"):
	filenames = os.listdir(dir_path)
	return [filename for filename in filenames if filename.endswith(extension)]

# Remove symbols like ".", ",", etc. from words
def clean_word(word):

	word = word.replace(",","")
	word = word.replace(".","")
	word = word.replace("\"","")
	word = word.replace(";","")

	return word

# Split the transcription into words and add to vocab
# Also keep a count of the word
def add_to_vocab(transcription):

	split_trans = transcription.split()

	for word in split_trans:

		# Clean the word first
		word = clean_word(word)

		if (word in global_vocab.keys()):
			global_vocab[word] += 1
		else:
			global_vocab[word] = 1

# Print the sorted global vocab as csv into a file
def print_vocab(file):

	f = open(file,'w')

	for word in sorted(global_vocab, key=global_vocab.get, reverse=True):
		f.write(word + "," + str(global_vocab[word]) + "\n")

	f.close()

# Filter words with condition that word_count >= 10
def print_extracted_vocab(file):

	f = open(file,'w')

	for word in sorted(global_vocab, key=global_vocab.get, reverse=True):

		if (global_vocab[word] >= 10):
			f.write(word + "\n")
		else:
			break

	f.close()

#####################################
############ MAIN CODE ##############

dir = "transcriptions"
files = find_csv_files(dir)

for file in files:

	f = open(dir + '/' + file,'r')
	lineno = 0

	for line in f:

		# If lineno == 0 -> Do Nothing.
		# First line is header
		if (lineno > 0):
			
			# Extract the third element (transcription) and add to the global vocabulary
			split = line.split(";")
			for i in range(len(split)-3):
				trans = split[i+2]
				add_to_vocab(trans)

		lineno += 1

	f.close()

# Print the vocabulary
print_vocab("vocab.csv")
print_extracted_vocab("extracted_vocab.txt")