import os.path

from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.models import load_model

from data import RNNSeriesDataSet
from seq2seq_tool_wear.model import *

model = None

# ---- PREPARATION ----
def plot_curve(start_point,model,predict_line_obj,direct_knife=3):
    # LOOP TO PREDICT THE WHOLE CURVE
    print("Draw for",start_point)
    for knife_number in range(direct_knife-1,direct_knife):
        knife_data = tool_wear_data[knife_number * 315:(knife_number + 1) * 315]
        # print("Current Knife shape:", knife_data.shape)


        START_POINT = start_point
        next = model.predict(knife_data[START_POINT:START_POINT + 2].reshape(1, 2, 1)).reshape(5)
        # life book
        life_total_data = [None for index in range(800)]
        cnt_total_data = [0 for index in range(800)]

        life_total_data[START_POINT:START_POINT + 2] = knife_data[START_POINT:START_POINT + 2]
        cnt_total_data[START_POINT] += 1
        cnt_total_data[START_POINT+1] += 1
        previous = next

        for every_start in range(knife_data.shape[0]):
            # predicted = model.predict(knife_data[every_start:every_start+2].reshape(1,2,1)).reshape(5)
            previous = np.array(life_total_data[every_start + START_POINT:every_start + START_POINT + 2])
            # print(previous)
            next = model.predict(previous.reshape(1, 2, 1)).reshape(5)
            for next_cur in range(5):
                if life_total_data[every_start + START_POINT + 2 + next_cur] == None:
                    # print("DIRECTLY GIVEN")
                    life_total_data[every_start + START_POINT + next_cur + 2] = next[next_cur]
                else:
                    # print(life_total_data[every_start + START_POINT + next_cur + 2],
                    #       cnt_total_data[every_start + START_POINT + 2 + next_cur])
                    life_total_data[every_start + START_POINT + next_cur + 2] = \
                        (next[next_cur] + life_total_data[every_start + START_POINT + next_cur + 2] * cnt_total_data[
                            every_start + START_POINT + 2 + next_cur]) \
                        / (cnt_total_data[every_start + START_POINT + 2 + next_cur] + 1)

                cnt_total_data[every_start + START_POINT + 2 + next_cur] += 1

        # plt.plot(knife_data, label="REAL")
        predict_line_obj.set_data([index for index in range(800)],life_total_data)
        plt.legend()
        # plt.savefig("../res/PURE_c%s.svg" % (knife_number + 1))
        # plt.show()


# ---- CONF ------
LOG_DIR = "MAX_KERAS_ROI_LOG/"
PREDICT = True

# ---- GEN Data ----
data = RNNSeriesDataSet(2,5)
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

        tool_wear_data = data.max_tool_wear_data
        from matplotlib import pyplot as plt
        import numpy as np
        import matplotlib.animation as animation

        fig, ax = plt.subplots()
        line, = ax.plot(tool_wear_data[315*2:315*3],label="real")
        predict_line_obj, = ax.plot(tool_wear_data[315:315*2],label="RNN_curve")

        # plot_curve(model,15)

        ani = animation.FuncAnimation(fig, plot_curve, frames=300, fargs=(model, predict_line_obj))
        plt.legend()
        ani.save('cut_3.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
        # plt.show()

