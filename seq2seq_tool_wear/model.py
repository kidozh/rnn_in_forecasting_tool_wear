import seq2seq
from seq2seq.models import AttentionSeq2Seq
from keras.models import Sequential
from keras.layers import LSTM,CuDNNLSTM,Dense,RepeatVector




def build_model(input_dim, input_length, hidden_dim,output_length,out_dim,depth):
    model = AttentionSeq2Seq(input_dim=input_dim,
                             input_length=input_length,
                             hidden_dim=hidden_dim,
                             output_length=output_length,
                             output_dim=out_dim,
                             depth=depth)
    model.compile(loss='logcosh', optimizer='adam',metrics=["mae","mse"])
    return model

def build_simple_RNN(input_shape,output_length,output_dim):

    model = Sequential()
    model.add(CuDNNLSTM(64,input_shape=input_shape,return_sequences=True))
    model.add(CuDNNLSTM(32,return_sequences=False))
    model.add(Dense(32))
    model.add(RepeatVector(output_length))
    model.add(CuDNNLSTM(32,return_sequences=True))
    model.add(CuDNNLSTM(16,return_sequences=True))
    model.add(Dense(output_dim))

    model.compile(loss='logcosh', optimizer='adam', metrics=["mae", "mse"])
    return model