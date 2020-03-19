import numpy as np
from keras.layers import *
from keras.models import *
from sklearn.model_selection import RepeatedKFold

seq_length = 24  # 步长，每个摘要中最多句子数
embed_size = 200  # doc2vec使用的维度
text_size = 3000  # 句子总个数
ref_input_dim = 3  # 输入中的被引次数的个数
output_dim = 3  # 输出中的被引次数的个数
lstm_units = 128  # lstm的输出向量的维度

def resha(att):
    return K.reshape(x=att, shape=(-1,2*seq_length*lstm_units))

def concat(att1, att2):
    return K.concatenate(tensors=[att1, att2], axis=1)

if __name__ == "__main__":

    doc_inputs = np.load('doc_inputs.npy')
    labels = np.load('labels.npy')
    ref_inputs = np.load('ref_inputs.npy')

    inputs1 = Input(shape=(seq_length, embed_size))
    inputs2 = Input(shape=(3,))
    drop1 = Dropout(0.3)(inputs1)
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True), name='bilstm')(drop1)
    lstm_re = Lambda(resha)(lstm_out)
    con = Lambda(concat, arguments={'att2':inputs2})(lstm_re)
    output = Dense(3, activation='relu')(con)
    model = Model(inputs=[inputs1, inputs2], outputs=output)
    model.compile(optimizer='adam', loss='mse')

    print(model.summary())

    kf = RepeatedKFold(n_splits=10, n_repeats=1, random_state=0)
    i = 1
    for train, test in kf.split(doc_inputs):
        doc_inputs_train = doc_inputs[train]
        ref_inputs_train = ref_inputs[train]
        labels_train = labels[train]

        doc_inputs_test = doc_inputs[test]
        ref_inputs_test = ref_inputs[test]
        labels_test = labels[test]

        print('Training------------')
        print("第{:d}折的训练开始：".format(i))

        model.fit([doc_inputs_train, ref_inputs_train], labels_train, epochs=500, batch_size=16)

        print()
        print('Testing--------------')
        print("第{:d}折的测试开始：".format(i))

        loss = model.evaluate([doc_inputs_test, ref_inputs_test], labels_test)

        print('test loss:', loss)
        i += 1

    model.save_weights("bilstm.h5")
