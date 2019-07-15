import os.path

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from data import *
from seq2seq_tool_wear.model import *
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Hiragino Sans GB']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False # 用来正常显示负号

model = None

# --- MID NUMBER FUNC ---
def get_median(data):
    data.sort()
    half = len(data) // 2
    return (data[half] + data[~half]) / 2

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

# ---- PREPARATION ----
def plot_curve(start_point, model, predict_line_obj, direct_knife=2):
    # LOOP TO PREDICT THE WHOLE CURVE
    print("Draw for", start_point)
    for knife_number in range(direct_knife - 1, direct_knife):
        knife_data = tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
        # print("Current Knife shape:", knife_data.shape)

        START_POINT = start_point
        next = model.predict(knife_data[START_POINT:START_POINT + 10].reshape(1, 10, 1)).reshape(20)
        # life book
        life_total_data = [None for index in range(800)]
        cnt_total_data = [0 for index in range(800)]


        life_total_data[START_POINT:START_POINT + 10] = knife_data[START_POINT:START_POINT + 10]
        for _ in range(10):
            cnt_total_data[START_POINT + _] += 1
        previous = next

        for every_start in range(knife_data.shape[0]):
            # predicted = model.predict(knife_data[every_start:every_start+2].reshape(1,2,1)).reshape(5)
            previous = np.array(life_total_data[every_start + START_POINT:every_start + START_POINT + 10])
            # print(previous)
            next = model.predict(previous.reshape(1, 10, 1)).reshape(20)
            for next_cur in range(20):
                if life_total_data[every_start + START_POINT + 10 + next_cur] == None:
                    # print("DIRECTLY GIVEN")
                    life_total_data[every_start + START_POINT + next_cur + 10] = next[next_cur]
                else:
                    # print(life_total_data[every_start + START_POINT + next_cur + 2],
                    #       cnt_total_data[every_start + START_POINT + 2 + next_cur])
                    life_total_data[every_start + START_POINT + next_cur + 10] = \
                        (next[next_cur] + life_total_data[every_start + START_POINT + next_cur + 10] * cnt_total_data[
                            every_start + START_POINT + 10 + next_cur]) \
                        / (cnt_total_data[every_start + START_POINT + 10 + next_cur] + 1)

                cnt_total_data[every_start + START_POINT + 10 + next_cur] += 1

        # plt.plot(knife_data, label="REAL")
        #predict_line_obj.set_data([index for index in range(800)], life_total_data)
        #plt.legend()
        # plt.savefig("../res/PURE_c%s.svg" % (knife_number + 1))
        # plt.show()


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
    TRAIN_NAME = "Simple_2_5_Separate_RNN_Depth_%s_hidden_dim_%s" % (DEPTH, HIDDEN_DIM)
    MODEL_NAME = "%s.kerasmodel" % (TRAIN_NAME)
    MODEL_WEIGHT_NAME = "%s.kerasweight" % (TRAIN_NAME)
    MODEL_CHECK_PT = "%s.kerascheckpts" % (TRAIN_NAME)

    # model = build_model(1, 2, HIDDEN_DIM, 3, 1, DEPTH)
    model = build_simple_RNN((2, 1), 5, 1)
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

        a = CNNMonitoredDataSet(2, 5)

        tool_wear_data = a.cnn_max_predict_wear
        real_tool_wear_data = data.max_tool_wear_data

        for knife_number in range(3):
            knife_data = tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
            real_knife_data = real_tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
            print("Current Knife shape:", knife_data.shape)
            first_list = [None, None, ]
            second_list = [None, None, None]
            third_list = [None, None, None, None]
            fourth_list = [None for _ in range(5)]
            fifth_list = [None for _ in range(6)]
            life_total_data = [None for index in range(315)]
            cnt_total_data = [0 for index in range(315)]
            predict_wear_status = [[] for index in range(315)]


            for every_start in range(knife_data.shape[0] - 6):
                predicted = model.predict(knife_data[every_start:every_start + 2].reshape(1, 2, 1)).reshape(5)
                predict_cur_time_start_idx = every_start + 2
                for pred_time_index in range(5):
                    current_total_index = predict_cur_time_start_idx + pred_time_index
                    predict_wear_status[current_total_index].append(predicted[pred_time_index])
                    if life_total_data[current_total_index] == None:
                        life_total_data[current_total_index] = predicted[pred_time_index]
                    else:
                        print("CHANGED DATA :",life_total_data[current_total_index], predicted[pred_time_index])
                        life_total_data[current_total_index] = (life_total_data[current_total_index] * cnt_total_data[current_total_index] + predicted[pred_time_index]) / (
                                                                           cnt_total_data[current_total_index] + 1)
                        # life_total_data[current_total_index] = (life_total_data[current_total_index] + predicted[pred_time_index]) / 2
                    cnt_total_data[current_total_index] += 1

                    # print(current_total_index)
                print(cnt_total_data)
                first_list.append(predicted[0])
                second_list.append(predicted[1])
                third_list.append(predicted[2])
            res_mid_number_list = [get_median(predict_wear_status[i]) for i in range(315) if predict_wear_status[i]!= []]
            res_mid_number_list = [None] * 2 + res_mid_number_list

            fig = plt.figure()

            plt.plot(real_knife_data, label="真实磨损量")
            plt.plot(knife_data,"s--",label="CNN监测值")
            plt.plot(res_mid_number_list,"o-.", label="基于中位修正的磨损量预报")
            plt.plot(life_total_data,"*:",label="基于平均修正的磨损量预报")
            plt.legend()
            plt.xlim(80-1, 110+1)
            min = real_knife_data[80:110].min()
            max = real_knife_data[80:110].max()
            plt.ylim(min-1,max+3)
            plt.xlabel("行程")
            plt.ylabel("磨损量 ($\mu m$)")

            plt.savefig("../res/CNN_invovled_comprehensive_all_c%s_with_CNN_zh_new_type_limited.svg" % (knife_number + 1))
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
