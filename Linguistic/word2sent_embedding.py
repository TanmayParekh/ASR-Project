import sys 
import os

embeddings = {}

# Extract word embeddings in form of a map
def make_embed_map(file):

	f = open(file,'r')
	for line in f:

		line_split = line.split()
		word = line_split[0]
		embed = line_split[1:]
		embed = [float(i) for i in embed]

		embeddings[word] = embed

# Combine word embeddings of words in a sentence to make a sentence embedding
def make_sent_feature(word_list):

	# METHOD 1 - Take sum of all word embeddings
	num = 0
	sum_embed = [0.0] * 300
	for word in word_list:

		if word in embeddings.keys():
			sum_embed = [x + y for x, y in zip(embeddings[word],sum_embed)]
			num += 1

	# return sum_embed

	# METHOD 2 - Take average of all word embeddings

	# If no word in embedding, then set to 0
	if num != 0:
		avg_embed = [float(val/num) for val in sum_embed]
	else:
		avg_embed = sum_embed
	# return avg_embed

	# METHOD 3 - Take max and min of all word embeddings
	min_embed = [2.0] * 300
	max_embed = [-2.0] * 300
	for word in word_list:

		if word in embeddings.keys():
			min_embed = [min(x,y) for x,y in zip(embeddings[word],sum_embed)]
			max_embed = [max(x,y) for x,y in zip(embeddings[word],sum_embed)]

	return min_embed + max_embed

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
def write_header(file):

	f = open(file,'w')
	header = "file_name,start_time,end_time"

	for i in range(300):
		name_feature = "feature" + str(i)
		header += "," + name_feature

	header += ",sentiment\n"
	f.write(header)

	f.close()

# Add a feature corresponding to each line in a file
def build_feature(line, dataset_file, feature_file):

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

	# Build feature vector for sentence
	feature = make_sent_feature(trans_split)

	# Create final feature vector
	feature_line = dataset_file + "," + start_time + "," + end_time
	for val in feature:
		feature_line += "," + str(val)
	feature_line += "," + (sentiment[:-1]).replace("\"", "") + "\n"

	# Write to file
	f.write(feature_line)

	f.close()


############################
######## MAIN CODE #########

final_file = "sent_embed_minmax.csv"
# final_file = "sent_embed_5_sum.csv"

# Get the embeddings from file
make_embed_map("vocab_embeddings.txt")
# make_embed_map("combined_vocab_embedding_5.txt")

# Write header
write_header(final_file)

dir = "transcriptions"
files = find_csv_files(dir)

for file in files:

	f = open(dir + '/' + file,'r')
	lineno = 0

	for line in f:

		# If lineno == 0 -> Do Nothing.
		# First line is header
		if (lineno > 0):
			
			build_feature(line,file,final_file)

		lineno += 1

	f.close()
