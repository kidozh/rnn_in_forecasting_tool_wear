import os.path

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from data import RNNSeriesDataSet
from seq2seq_tool_wear.model import *
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
# plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

def get_loss_value(real, pred):

    return mean_squared_error(real,pred),mean_absolute_error(real,pred)
    cnt = 0
    for i in pred:
        if i == None:
            cnt += 1
        else:
            break
    post_prex = 0
    for i in range(len(pred)):
        if pred[-i] == None:
            post_prex += 1
        else:
            break
    length = min(len(pred) - post_prex, len(real))
    # print(cnt, length)
    return mean_squared_error(real[cnt:length], pred[cnt:length]), mean_absolute_error(real[cnt:length],
                                                                                       pred[cnt:length])


model = None

KNIFE_NUMBER = 1

INPUT_NUMBER = 10
OUTPUT_NUMBER = 20

# ---- PREPARATION ----
def plot_forecast_data(start_point, model, direct_knife=KNIFE_NUMBER, input_length=INPUT_NUMBER, output_length=OUTPUT_NUMBER):
    # LOOP TO PREDICT THE WHOLE CURVE
    print("Draw for", start_point)

    for knife_number in range(direct_knife - 1, direct_knife):
        knife_data = tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
        # print("Current Knife shape:", knife_data.shape)

        START_POINT = start_point
        next = model.predict(knife_data[START_POINT:START_POINT + input_length].reshape(1, input_length, 1)).reshape(
            output_length)
        # life book
        life_total_data = [None for index in range(800)]
        cnt_total_data = [0 for index in range(800)]

        life_total_data[START_POINT:START_POINT + input_length] = knife_data[START_POINT:START_POINT + input_length]
        for _ in range(input_length):
            cnt_total_data[START_POINT + _] += 1
        previous = next

        for every_start in range(knife_data.shape[0]-START_POINT):
            # predicted = model.predict(knife_data[every_start:every_start+2].reshape(1,2,1)).reshape(5)
            previous = np.array(life_total_data[every_start + START_POINT:every_start + START_POINT + input_length])
            # print(previous)
            next = model.predict(previous.reshape(1, input_length, 1)).reshape(output_length)
            for next_cur in range(output_length):
                if life_total_data[every_start + START_POINT + input_length + next_cur] == None:

                    life_total_data[every_start + START_POINT + next_cur + input_length] = next[next_cur]
                else:

                    life_total_data[every_start + START_POINT + next_cur + input_length] = \
                        (next[next_cur] + life_total_data[every_start + START_POINT + next_cur + input_length] *
                         cnt_total_data[
                             every_start + START_POINT + input_length + next_cur]) \
                        / (cnt_total_data[every_start + START_POINT + input_length + next_cur] + 1)

                cnt_total_data[every_start + START_POINT + input_length + next_cur] += 1

        life_total_data[START_POINT:START_POINT + input_length] = [None]*input_length
        fig, ax = plt.subplots()

        line, = ax.plot(tool_wear_data[315 * (KNIFE_NUMBER - 1):315 * (KNIFE_NUMBER)], label="Tool wear degradation",
                        linestyle='-.', color="#95a5a6")
        predict_line_obj, = ax.plot(tool_wear_data[315 * (KNIFE_NUMBER - 1):315 * (KNIFE_NUMBER)],
                                    label="Forecast value", color="#e74c3c")
        real_line, = ax.plot([], [], label="Historical input", color="#1abc9c",linestyle = "--")

        # plot_curve(model,15)
        plt.legend()
        # plt.plot(knife_data, label="REAL Tool wear")
        predict_line_obj.set_data([index for index in range(800)], life_total_data)

        real_line.set_data([index for index in range(start_point,start_point+input_length)], knife_data[start_point:start_point+input_length])
        plt.annotate("Forecast next tool wear\nafter this point",
                     xy=(start_point+ input_length, knife_data[start_point+input_length]),
                     xycoords='data', xytext=(20, +80),
                     textcoords='offset points',
                     arrowprops=dict(arrowstyle='->', connectionstyle="arc3,rad=.2"))
        for i in range(12):
            x0 = start_point+(i)*output_length+input_length
            y0 = life_total_data[x0]
            if i != 0:
                if i == 11:
                    pass
                else:
                    plt.text(x0+5,85,'%s'%(i+1))
            else:
                plt.text(x0+1,85,"base")
            plt.plot([x0,x0],[0,y0],linestyle=":",color="#34495e",linewidth=1.5)
            plt.scatter([x0],[y0],marker="o",color="#e74c3c")

        plt.xlim(start_point-3,start_point+11*output_length+input_length+3)
        min = np.array(knife_data[start_point-1:start_point+input_length+1]).min()
        max = np.array(life_total_data[start_point+input_length:start_point+11*output_length+input_length+1]).max()
        plt.ylim(min-3,max+3)


        plt.xlabel("Run")
        plt.ylabel("Tool wear ($\mu m$)")

        plt.savefig("../res/FORECAST_%s_%s_in_%s_of_c%s_annotate_en.pdf" % (INPUT_NUMBER,OUTPUT_NUMBER,START_POINT,knife_number + 1))
        plt.show()


# ---- CONF ------
LOG_DIR = "MAX_KERAS_ROI_LOG/"
PREDICT = True

# ---- GEN Data ----
data = RNNSeriesDataSet(10, 20)
# x,y = data.get_rnn_data()
x, y, test_x, test_y = data.get_separate_rnn_data()

# ---- shuffle -----
import random

# set random seeds so that we can get the same random data!
SEED = 12347
random.seed(SEED)
index = [i for i in range(len(y))]
random.shuffle(index)
train_y = y[index]
train_x = x[index]

print("Size :", train_x.shape, train_y.shape, test_x.shape, test_y.shape)

for DEPTH in [5]:

    HIDDEN_DIM = 128

    TRAIN_NAME = "Simple_%s_%s_Separate_RNN_Depth_%s_hidden_dim_%s" % (INPUT_NUMBER,OUTPUT_NUMBER,DEPTH, HIDDEN_DIM)
    MODEL_NAME = "%s.kerasmodel" % (TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight" % (TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (TRAIN_NAME)

    # model = build_model(1, 2, HIDDEN_DIM, 3, 1, DEPTH)
    model = build_simple_RNN((INPUT_NUMBER, 1), OUTPUT_NUMBER, 1)
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

        model.fit(train_x, train_y, batch_size=16, epochs=5000, callbacks=[tb_cb, ckp_cb],
                  validation_data=(test_x, test_y), shuffle=True)

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
        from matplotlib import pyplot as plt

        # plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
        # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        import numpy as np
        import matplotlib.animation as animation
        STARTPOINT = 20
        plot_forecast_data(STARTPOINT,model)




