import os.path

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from data import *
from seq2seq_tool_wear.model import *
import numpy as np
import matplotlib.pyplot as plt

model = None

INPUT_NUMBER = 2
OUTPUT_NUMBER = 5


# ---- PREPARATION ----
# ---- need func --
def get_loss_value(real,pred):
    from sklearn.metrics import mean_squared_error,mean_absolute_error
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
    length = min(len(pred)-post_prex,len(real))
    print(cnt,length)
    return mean_squared_error(real[cnt:length],pred[cnt:length]),mean_absolute_error(real[cnt:length],pred[cnt:length])

def plot_curve(start_point,model,predict_line_obj,direct_knife=2):
    # LOOP TO PREDICT THE WHOLE CURVE
    print("Draw for",start_point)
    for knife_number in range(direct_knife-1,direct_knife):
        knife_data = tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
        # print("Current Knife shape:", knife_data.shape)


        START_POINT = start_point
        next = model.predict(knife_data[START_POINT:START_POINT + INPUT_NUMBER].reshape(1, INPUT_NUMBER, 1)).reshape(OUTPUT_NUMBER)
        # life book
        life_total_data = [None for index in range(800)]
        cnt_total_data = [0 for index in range(800)]

        life_total_data[START_POINT:START_POINT + INPUT_NUMBER] = knife_data[START_POINT:START_POINT + INPUT_NUMBER]
        for _ in range(INPUT_NUMBER):
            cnt_total_data[START_POINT+_] += 1
        previous = next

        for every_start in range(knife_data.shape[0]):
            # predicted = model.predict(knife_data[every_start:every_start+2].reshape(1,2,1)).reshape(5)
            previous = np.array(life_total_data[every_start + START_POINT:every_start + START_POINT + INPUT_NUMBER])
            # print(previous)
            next = model.predict(previous.reshape(1, INPUT_NUMBER, 1)).reshape(OUTPUT_NUMBER)
            for next_cur in range(OUTPUT_NUMBER):
                if life_total_data[every_start + START_POINT + INPUT_NUMBER + next_cur] == None:
                    # print("DIRECTLY GIVEN")
                    life_total_data[every_start + START_POINT + next_cur + INPUT_NUMBER] = next[next_cur]
                else:
                    # print(life_total_data[every_start + START_POINT + next_cur + 2],
                    #       cnt_total_data[every_start + START_POINT + 2 + next_cur])
                    life_total_data[every_start + START_POINT + next_cur + INPUT_NUMBER] = \
                        (next[next_cur] + life_total_data[every_start + START_POINT + next_cur + INPUT_NUMBER] * cnt_total_data[
                            every_start + START_POINT + INPUT_NUMBER + next_cur]) \
                        / (cnt_total_data[every_start + START_POINT + INPUT_NUMBER + next_cur] + 1)

                cnt_total_data[every_start + START_POINT + INPUT_NUMBER + next_cur] += 1

        # plt.plot(knife_data, label="REAL")
        predict_line_obj.set_data([index for index in range(800)],life_total_data)
        plt.legend()
        # plt.savefig("../res/PURE_c%s.svg" % (knife_number + 1))
        # plt.show()


# ---- CONF ------
LOG_DIR = "MAX_KERAS_ROI_LOG/"
PREDICT = True

# ---- GEN Data ----
data = RNNSeriesDataSet(INPUT_NUMBER,OUTPUT_NUMBER)
# x,y = data.get_rnn_data()
x,y,test_x,test_y = data.get_separate_rnn_data()

# ---- shuffle -----
import random
# set random seeds so that we can get the same random data!
SEED = 12347
random.seed(SEED)
index = [i for i in range(len(y))]
random.shuffle(index)
train_y = y[index]
train_x = x[index]

print("Size :",train_x.shape,train_y.shape,test_x.shape,test_y.shape)

for DEPTH in [5]:

    HIDDEN_DIM = 128
    TRAIN_NAME = "Simple_%s_%s_Separate_RNN_Depth_%s_hidden_dim_%s" % (INPUT_NUMBER,OUTPUT_NUMBER,DEPTH,HIDDEN_DIM)
    MODEL_NAME = "%s.kerasmodel" % (TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight" % (TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (TRAIN_NAME)

    # model = build_model(1, 2, HIDDEN_DIM, 3, 1, DEPTH)
    model = build_simple_RNN((INPUT_NUMBER,1),OUTPUT_NUMBER,1)
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

        model.fit(train_x,train_y,batch_size=16,epochs=5000,callbacks=[tb_cb,ckp_cb],validation_data=(test_x,test_y),shuffle=True)

        model.model.save(MODEL_NAME)
        model.save_weights(MODEL_WEIGHT_NAME)

    else:
        if os.path.exists(MODEL_CHECK_PT):
            model.load_weights(MODEL_CHECK_PT)
            print("load checkpoint successfully")
        else:
            print("No checkpoints found! try to load model directly")
            model = load_model(MODEL_NAME)

        a = CNNMonitoredDataSet(INPUT_NUMBER,OUTPUT_NUMBER)

        tool_wear_data = a.cnn_max_predict_wear
        real_tool_wear_data = data.max_tool_wear_data

        for i in range(3):
            print("LOSS:",i,get_loss_value(real_tool_wear_data[i*315:(i+1)*315],tool_wear_data[i*315:(i+1)*315]))



        for knife_number in range(3):
            knife_data = tool_wear_data[knife_number*315:(knife_number+1)*315]
            real_knife_data = real_tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
            print("Current Knife shape:",knife_data.shape)
            first_list = [None,None,]
            second_list = [None,None,None]
            third_list = [None,None,None,None]
            fourth_list = [None for _ in range(5)]
            fifth_list = [None for _ in range(6)]
            fig = plt.figure()
            for every_start in range(knife_data.shape[0]-5):
                predicted = model.predict(knife_data[every_start:every_start+INPUT_NUMBER].reshape(1,INPUT_NUMBER,1)).reshape(OUTPUT_NUMBER)

                first_list.append(predicted[0])
                second_list.append(predicted[1])
                third_list.append(predicted[2])
                fourth_list.append(predicted[3])
                fifth_list.append(predicted[4])


            plt.plot(real_knife_data,label="REAL")
            plt.scatter([i for i in range(len(first_list))], first_list, label="1st forecast value", s=2, marker="x")
            plt.scatter([i for i in range(len(second_list))], second_list, label="2nd forecast value", s=2, marker="o")
            plt.scatter([i for i in range(len(third_list))], third_list, label="3rd forecast value", s=2, marker="v")
            plt.scatter([i for i in range(len(fourth_list))], fourth_list, label="4th forecast value", s=2, marker=",")
            plt.scatter([i for i in range(len(fifth_list))], fifth_list, label="5th forecast value", s=2, marker=".")
            plt.legend()

            # calculate MSE
            print(get_loss_value(knife_data, first_list))
            print(get_loss_value(knife_data, second_list))
            print(get_loss_value(knife_data, third_list))
            print(get_loss_value(knife_data, fourth_list))
            print(get_loss_value(knife_data, fifth_list))

            plt.xlabel("Run")
            plt.ylabel("Tool wear ($\mu m$)")
            plt.savefig("../res/CNN_short_invovled_c%s.pdf"%(knife_number+1))
            plt.show()

        # from matplotlib import pyplot as plt
        # import numpy as np
        # import matplotlib.animation as animation
        #
        # fig, ax = plt.subplots()
        # line, = ax.plot(tool_wear_data[315*1:315*2],label="real")
        # predict_line_obj, = ax.plot(tool_wear_data[315:315*2],label="RNN_curve")
        #
        # # plot_curve(model,15)
        #
        # ani = animation.FuncAnimation(fig, plot_curve, frames=300, fargs=(model, predict_line_obj))
        # plt.legend()
        # ani.save('LONG_TERM_cut_2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        # plt.show()

