# modules for data cleansing and preparation
from gensim.summarization import summarize
from .utility import *

# Language index creation
class Lang:
	'''
	class to create mapping and reverse mappings of words and indexes
	'''
	def __init__(self, name):
		self.name = name
		self.word2index = {}
		self.word2count = {}
		self.index2word = {0: "SOS", 1: "EOS"}
		self.n_words = 2  # Count SOS and EOS
		self.max_length = -1

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

		if len(sentence.split(' ')) > self.max_length:
			self.max_length = len(sentence.split(' '))

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.n_words
			self.word2count[word] = 1
			self.index2word[self.n_words] = word
			self.n_words += 1
		else:
			self.word2count[word] += 1

# Data preparation
def indexesFromSentence(lang, sentence):
	'''
	takes in language index and sentence, to return list of vocab index mappings
	'''
	return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
	'''
	takes in language index and sentence, creates a 1D tensor of mappings
	'''
	indexes = indexesFromSentence(lang, sentence)
	indexes.append(EOS_token)
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(input_lang, output_lang, pair):
	'''
	takes in abstract,title language indexes and single pair of abstract & title
	returns two 1D corresponding tensors of mappings for both
	'''
	input_tensor = tensorFromSentence(input_lang, pair[0])
	target_tensor = tensorFromSentence(output_lang, pair[1])
	return (input_tensor, target_tensor)

# def normalizeString(sent):
# 	'''
# 	Minimal natural language cleansing and preprocessing
# 	'''
# 	sent = sent.lower().strip()
# 	sent = re.sub(r"([.,?])", r" \1", sent)
# 	sent = re.sub(r"[^a-zA-Z.,?]+", r" ", sent)
# 	return sent

def normalizeString(sent):
	'''
	Minimal natural language cleansing and preprocessing
	'''
	sent = sent.strip()
	sent = re.sub(r"([.,?])", r" \1", sent)
	sent = re.sub(r"[^a-zA-Z.,?-]+", r" ", sent)

	# differentiate '- M' from '- m' with '$- M'
	sent = re.sub(r"(- [A-Z])", r"$\1", sent)
	# smooth '$- M' to ' M' eg Exponential- Multiplication
	sent = sent.replace("$-", "")
	# smooth '- m' to 'm' eg  multi- plicative
	sent = sent.replace("- ", "")
	# now leftover genuine cases are handled eg non-negative
	sent = sent.replace("-", " ")
	return sent.lower()

def shorten_abstract(abstract):
	'''
	performs extractive summarization on abstract for faster training,
	has hardcoded limits within this module
	'''
	if len(abstract.split()) > 300:
		abstract = summarize (abstract, word_count=100, split=False)

	return abstract
	