from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, LSTM, Convolution2D, Embedding
from keras.layers.merge import concatenate
from keras.utils import plot_model


window_size = 5
mesh_rows = 20
mesh_columns = 22
cnn_activation ="relu"
lstm_activation="relu"
model_activation="softmax"
pool_size = (1,1)
number_conv2D_filters = 10
kernel_shape = (7)
embedding_output_dim = 256
number_lstm_cells = 10
number_nodes_hidden = 10
num_channels = 128
inputs = []
convs = []

for i in range(window_size):  # Adding mesh inputs
    input = Input(shape=(mesh_rows, mesh_columns, 1), name = "input-mesh"+str(i+1))
    inputs.append(input)

input = Input(shape=(num_channels, window_size), name="input-list")
inputs.append(input)

for i in range(window_size): # Connecting mesh inputs to CNNs
    conv = Conv2D(number_conv2D_filters, kernel_shape, activation=cnn_activation, input_shape=(mesh_rows, mesh_columns, 1))(inputs[i])# modify shape and kernel
    pool = MaxPool2D(pool_size=pool_size)(conv) # modify pool size
    convs.append(Flatten()(pool))

embed = Embedding(input_dim=11200, output_dim=embedding_output_dim, input_length=11200)(inputs[len(inputs)-1]) # Connecting list input to RNN
lstm = LSTM(number_lstm_cells, return_sequences=False)(embed)
hidden1 = Dense(number_nodes_hidden, activation=lstm_activation)(lstm)

fusion = list(convs,hidden1)

fused = concatenate(fusion)



output = Dense(1, activation=model_activation)(fused)


#hard coded value of 11200 = 5 * 2240 but we can see later how to get it programatically


model = Model(inputs=inputs, outputs=output)

print(model.summary())

plot_model(model, show_shapes=True, dpi=50, to_file='Parallel_CNN_RNN.png')