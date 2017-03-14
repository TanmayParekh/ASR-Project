import sys 
import os

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

# Write header (feature names) into file
def write_header(file, vocab):

	f = open(file,'w')
	header = "file_name,start_time,end_time"

	for word in vocab:
		header += "," + word

	header += ",sentiment\n"
	f.write(header)

	f.close()

# Add a feature corresponding to each line in a file
def build_feature(line, dataset_file, feature_file, vocab_list):

	f = open(feature_file,'a')

	# Extract features from line
	split_line = line.split(';')
	start_time = split_line[0]
	end_time = split_line[1]

	# Extract the transcription
	trans = ""
	for i in range(len(split_line)-3):
		trans = trans + split_line[i+2]

	sentiment = split_line[-1]

	# Extract words in trans and clean them
	trans_split = trans.split()
	for word in trans_split:
		word = clean_word(word)

	# Build feature count for the words extracted above
	feature_count = []
	for feature in vocab_list:
		feature_count.append(trans_split.count(feature))

	# Create final feature vector
	feature_line = dataset_file + "," + start_time + "," + end_time
	for count in feature_count:
		feature_line += "," + str(count)
	feature_line += "," + (sentiment[:-1]).replace("\"", "") + "\n"

	# Write to file
	f.write(feature_line)

	f.close()


############################
######## MAIN CODE #########

final_file = "final_feature.csv"

# Make a list of extracted words
vocab_file = open("extracted_vocab.txt")
extr_vocab = []

for line in vocab_file:
	extr_vocab.append(line[:-1])

# Write header
write_header(final_file, extr_vocab)

dir = "transcriptions"
files = find_csv_files(dir)

for file in files:

	f = open(dir + '/' + file,'r')
	lineno = 0

	for line in f:

		# If lineno == 0 -> Do Nothing.
		# First line is header
		if (lineno > 0):
			
			build_feature(line,file,final_file,extr_vocab)

		lineno += 1

	f.close()
