from seq2seq_tool_wear.model import *
from data import RNNSeriesDataSet
from keras.callbacks import TensorBoard,ModelCheckpoint
import os.path
from keras.models import load_model
import matplotlib.pyplot as plt

# ---- CONF ------
LOG_DIR = "MAX_KERAS_ROI_LOG/"
PREDICT = True

# ---- GEN Data ----
data = RNNSeriesDataSet(2,5)
x,y = data.get_rnn_data()

# ---- shuffle -----
import random
# set random seeds so that we can get the same random data!
SEED = 12347
random.seed(SEED)
index = [i for i in range(len(y))]
random.shuffle(index)
train_y = y[index]
train_x = x[index]

print("Size :",train_x.shape,train_y.shape)

for DEPTH in [5]:

    HIDDEN_DIM = 128
    TRAIN_NAME = "Simple_2_5_Separate_RNN_Depth_%s_hidden_dim_%s" % (DEPTH,HIDDEN_DIM)
    MODEL_NAME = "%s.kerasmodel" % (TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight" % (TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (TRAIN_NAME)

    # model = build_model(1, 2, HIDDEN_DIM, 3, 1, DEPTH)
    model = build_simple_RNN((2,1),5,1)
    print(model.summary())
    print("Model has been built.")

    if not PREDICT:
        print("In [TRAIN] mode")
        # CALLBACK
        tb_cb = TensorBoard(log_dir=LOG_DIR + TRAIN_NAME)
        ckp_cb = ModelCheckpoint(MODEL_CHECK_PT, monitor='val_loss', save_weights_only=True, verbose=1,
                                 save_best_only=True, period=5)

        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)
            print("load checkpoint successfully")
        else:
            print("No checkpoints found !")
        print("Start to train the model")

        model.fit(train_x,train_y,batch_size=16,epochs=5000,validation_split=0.2,callbacks=[tb_cb,ckp_cb])

        model.model.save(MODEL_NAME)
        model.save_weights(MODEL_WEIGHT_NAME)

    else:
        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)
            print("load checkpoint successfully")
        else:
            print("No checkpoints found! try to load model directly")
            model = load_model(MODEL_NAME)

        tool_wear_data = data.max_tool_wear_data

        for knife_number in range(3):
            knife_data = tool_wear_data[knife_number*315:(knife_number+1)*315]
            print("Current Knife shape:",knife_data.shape)
            first_list = [None,None,]
            second_list = [None,None,None]
            third_list = [None,None,None,None]
            fig = plt.figure()
            for every_start in range(knife_data.shape[0]-5):
                predicted = model.predict(knife_data[every_start:every_start+2].reshape(1,2,1)).reshape(3)

                first_list.append(predicted[0])
                second_list.append(predicted[1])
                third_list.append(predicted[2])


            plt.plot(knife_data,label="REAL")
            plt.plot(first_list,label="1st")
            plt.plot(second_list, label="2nd")
            plt.plot(third_list, label="3rd")
            plt.legend()
            plt.savefig("../res/c%s.svg"%(knife_number+1))
            plt.show()






