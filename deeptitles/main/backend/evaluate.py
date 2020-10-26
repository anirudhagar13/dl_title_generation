# module to load saved models and perform prediction for client

from .model import *
from .preprocess import *

def generate(abstract):
	'''
	takes in abstract from UI and returns generated title 
	'''
	try:
		# preprocessing
		abstract = normalizeString(abstract)
		
		# index loading
		input_lang = load_index(ABS_LANG_PATH)
		output_lang = load_index(TITLE_LANG_PATH)

		# need to remove words in abstract that are not in vocab
		abstract = ' '.join([x for x in abstract.split() if 
								x in input_lang.word2index])

		# model loading
		encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)
		attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, 
										max_length=input_lang.max_length+1, 
										dropout_p=DROPOUT).to(device)

		encoder = load_model(encoder, ENC_MODEL_PATH)
		attn_decoder = load_model(attn_decoder, DEC_MODEL_PATH)
	except Exception as e:
		raise Exception('Model unable to load, model possibly not trained.')

	# prediction
	output_words, attentions = evaluate(encoder, attn_decoder, abstract, 
										input_lang, output_lang, 
										max_length=input_lang.max_length+1)
	return ' '.join(output_words[:-1])

def evaluate(encoder, decoder, abstract, input_lang, output_lang, 
				max_length):
	'''
	loads model and generates title for abstract
	'''
	with torch.no_grad():
		input_tensor = tensorFromSentence(input_lang, abstract)
		input_length = input_tensor.size()[0]
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, 
										device=device)

		# getting hidden and outputs of trained encoder
		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei], 
														encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		# initializing decoder inputs
		decoder_hidden = encoder_hidden
		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
																decoder_input, 
																decoder_hidden, 
																encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<EOS>')
				break
			else:
				decoded_words.append(output_lang.index2word[topi.item()])

			decoder_input = topi.squeeze().detach()

		return decoded_words, decoder_attentions[:di + 1]

def evaluateRandomly(encoder, decoder, pairs, input_lang, output_lang, n=20):
	'''
	quick random evaluation of validation set during training
	'''
	for i in range(n):
		pair = random.choice(pairs)
		print('>', pair[0])
		print('=', pair[1])
		output_words, attentions = evaluate(encoder, decoder, pair[0], 
											input_lang, output_lang, 
											max_length=input_lang.max_length+1)
		output_sentence = ' '.join(output_words)
		print('<', output_sentence)
		print('')
