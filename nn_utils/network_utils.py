from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
import keras.layers.wrappers
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense

def create_lstm_network(num_frequency_dimensions, num_hidden_dimensions, sz, num_recurrent_units=1):
	model = Sequential()
	#This layer converts frequency space to hidden space
	model.add( TimeDistributed( Dense( num_hidden_dimensions ), input_shape=sz ) )

	for cur_unit in xrange(num_recurrent_units):
		model.add(LSTM(input_dim=num_hidden_dimensions, output_dim=num_hidden_dimensions, return_sequences=True))
	#This layer converts hidden space back to frequency space
	model.add(TimeDistributed(Dense(input_dim=num_hidden_dimensions, output_dim=num_frequency_dimensions)))
	model.compile(loss='mean_squared_error', optimizer='rmsprop')
	return model

