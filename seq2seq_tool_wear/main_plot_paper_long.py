from seq2seq_tool_wear.model import *
from data import RNNSeriesDataSet
from keras.callbacks import TensorBoard,ModelCheckpoint
import os.path
from keras.models import load_model
import matplotlib.pyplot as plt

# ---- need func --
def get_loss_value(real,pred):
    from sklearn.metrics import mean_squared_error,mean_absolute_error
    cnt = 0
    for i in pred:
        if i == None:
            cnt += 1
        else:
            break
    length = min(len(pred),len(real))
    return mean_squared_error(real[cnt:length],pred[cnt:length]),mean_absolute_error(real[cnt:length],pred[cnt:length])

# ---- CONF ------
LOG_DIR = "MAX_KERAS_ROI_LOG/"
PREDICT = True

# ---- GEN Data ----
data = RNNSeriesDataSet(10,20)
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
    TRAIN_NAME = "Simple_10_20_Separate_RNN_Depth_%s_hidden_dim_%s" % (DEPTH,HIDDEN_DIM)
    MODEL_NAME = "%s.kerasmodel" % (TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight" % (TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (TRAIN_NAME)

    # model = build_model(1, 2, HIDDEN_DIM, 3, 1, DEPTH)
    model = build_simple_RNN((10,1),20,1)
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
            if knife_number < 2 :
                continue
            knife_data = tool_wear_data[knife_number*315:(knife_number+1)*315]
            print("Current Knife shape:",knife_data.shape)
            import matplotlib.pyplot as plt

            # plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
            # plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号
            fig = plt.figure(0)

            START_NUMBER = 10




            every_start = START_NUMBER
            predicted = model.predict(knife_data[every_start:every_start+10].reshape(1,10,1)).reshape(20)


            plt.plot(knife_data,label="Tool wear degradation",c="#7f8c8d",linestyle="-.")
            plt.plot([None]*(every_start+10)+predicted.tolist(),label="Forecast value",c="#e74c3c")
            plt.plot([None]*every_start+knife_data[every_start:every_start+10].tolist(),label="Historical input",c="#1abc9c",linestyle="--")
            # print(len([None]*(every_start+2)+predicted.tolist()))
            plt.scatter([i for i in range(315)], knife_data, s=50, marker=".", c="#7f8c8d")
            plt.scatter([i for i in range(every_start,every_start+10)],knife_data[every_start:every_start+10],s=80,marker=".",c="#1abc9c")
            plt.scatter([i for i in range(every_start+10,every_start+30)],predicted.tolist(),s=80,marker=".",c="#e74c3c")

            # annotate
            plt.annotate("Forecast next tool wear after this point",
                         xy=(every_start+9,knife_data[every_start+9]),
                         xycoords='data', xytext=(+0, -30),
                        textcoords='offset points',
                         arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))



            plt.xlabel("Run")
            plt.ylabel("Tool wear ($\mu m$)")

            plt.xlim(every_start-1,every_start+30+1)
            plt.ylim(knife_data[every_start:every_start+30].min()-1,knife_data[every_start:every_start+30].max()+2)

            plt.legend()
            plt.savefig("../res/paper_long_c%s_en.pdf"%(knife_number+1))
            plt.show()
            plt.close(0)






