# modules to use models and train in batches

from .model import *
from .preprocess import *
from .evaluate import evaluateRandomly

# Actual tranining iterations
def trainIters(encoder, decoder, pairs, max_length, input_lang, output_lang, 
				n_iters, print_every=1000, plot_every=1000, learning_rate=0.01):
	'''
	encoder, decoder are model instances to be trained, 
	pairs are abstract and title pairs, 
	max_length is maximum worded abstract length to make appr tensors,
	input_lang is language index of abstracts, output_lang is for title
	'''
	plot_losses = []
	plot_loss_total = 0  # Reset every plot_every
	print_loss_total = 0  # Reset every print_every
	start = time.time()

	encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
	training_pairs = [tensorsFromPair(input_lang, output_lang, 
						random.choice(pairs)) for i in range(n_iters)]
	criterion = nn.NLLLoss()

	for iter in range(1, n_iters + 1):
		training_pair = training_pairs[iter - 1]
		input_tensor = training_pair[0]
		target_tensor = training_pair[1]

		loss = train(input_tensor, target_tensor, encoder,
			decoder, encoder_optimizer, decoder_optimizer, 
			criterion, max_length=max_length)
		plot_loss_total += loss
		print_loss_total += loss

		if iter % print_every == 0:
			print_loss_avg = print_loss_total / print_every
			print_loss_total = 0
			print ('#{0}({1}%) ; avg_loss={2} ; time={3}'.format(
				iter, round(iter / n_iters * 100, 4), round(print_loss_avg, 4), 
				datetime.timedelta(seconds=time.time() - start)))

		if iter % plot_every == 0:
			plot_loss_avg = plot_loss_total / plot_every
			plot_losses.append(plot_loss_avg)
			plot_loss_total = 0

	return plot_losses

# Training module
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, 
			decoder_optimizer, criterion, max_length):

	'''
	runs for each pair of title and abstract one by one,
	takes both input and output as tensors, with encoder and decoder instances,
	first trains encoder, initially uses last encoder hidden state for attention,
	then trains decoder and uses decoder last hidden as input for attention
	'''
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	# initializing encoder output and loss
	loss = 0
	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	# encoder training
	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(input_tensor[ei], 
													encoder_hidden)
		encoder_outputs[ei] = encoder_output[0, 0]

	# initializing decoder tensors
	decoder_hidden = encoder_hidden
	decoder_input = torch.tensor([[SOS_token]], device=device)

	# Teacher forcing: Feed the target as the next input
	for di in range(target_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, 
												decoder_hidden, encoder_outputs)
		loss += criterion(decoder_output, target_tensor[di])
		decoder_input = target_tensor[di]  # Teacher forcing

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

if __name__ == '__main__':
	# client code to run training

	# checking if no of iterations passed as sys argument, else taking default
	if len(sys.argv) >= 2:
		N_ITER = int(sys.argv[1])

	# logging model description for more transparent batch processing
	model_desc = ''
	if len(sys.argv) >= 3:
		model_desc = sys.argv[2]

	# read data
	data = load_data(DATA_PATH)

	# preprocess data
	data = data.applymap(lambda x: normalizeString(x))

	# summarizing abstract for faster training
	data['abstract'] = data['abstract'].apply(lambda x: shorten_abstract(x))

	# prepare language indexes
	data_pairs = list()
	input_lang = Lang('abstract')
	output_lang = Lang('title')

	for index, row in data.iterrows():
		input_lang.addSentence(row['abstract'])
		output_lang.addSentence(row['title'])
		data_pairs.append([row['abstract'],row['title']])

	# train data (instanstiate encoder and decoder networks)
	encoder = EncoderRNN(input_lang.n_words, HIDDEN_SIZE).to(device)

	attn_decoder = AttnDecoderRNN(HIDDEN_SIZE, output_lang.n_words, 
									max_length=input_lang.max_length+1, 
									dropout_p=DROPOUT).to(device)

	losses = trainIters(encoder, attn_decoder, data_pairs[:-validate_num], 
						input_lang.max_length+1, input_lang, output_lang, N_ITER, 
						learning_rate=LR)

	# plot results (maybe)
	print ('Losses Incurred: \n=> ', losses)

	# using current date time as suffix to train multiple models
	suffix = '_'+ datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

	# saving model
	save_index(input_lang, ABS_LANG_PATH + suffix)
	save_index(output_lang, TITLE_LANG_PATH + suffix)
	save_model(encoder, ENC_MODEL_PATH + suffix)
	save_model(attn_decoder, DEC_MODEL_PATH + suffix)
	print ('++> Model: {0} :: @ {1}: \n'.format(model_desc, suffix))

	# validation (just a small set of generation from validation set)
	print ('*** Validation Set ***')
	evaluateRandomly(encoder, attn_decoder, data_pairs[-validate_num:], 
						input_lang, output_lang)

