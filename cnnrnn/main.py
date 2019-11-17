from keras.models import Model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, Input, LSTM, Convolution2D
from keras.layers.merge import concatenate
from keras.utils import plot_model

window_size = 5
mesh_xsize = 10
mesh_ysize = 10
cnn_activation ="relu"
lstm_activation="relu"
model_activation="softmax"
inputs = []
convs = []

for i in range(window_size):
    input = Input(shape=(mesh_xsize, mesh_ysize, 1))
    inputs.append(input)

for i in range(window_size):
    conv = Conv2D(10, (7), activation=cnn_activation, input_shape=(mesh_xsize, mesh_ysize, 1))(inputs[i])# modify shape and kernel
    pool = MaxPool2D(pool_size=(1,1))(conv) # modify pool size
    convs.append(Flatten()(pool))

merge = concatenate(convs)
lstm = LSTM(10, return_sequences=True)(merge)
hidden1 = Dense(10, activation=lstm_activation)(lstm)
output = Dense(1, activation=model_activation)(hidden1)

model = Model(inputs=inputs, outputs=output)

print(model.summary())

plot_model(model, to_file='my_net.png')
