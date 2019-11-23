#!/usr/bin/env python
# coding: utf-8



#读取训练数据
def load_train(path, J=4):
    uid, pos_item, neg_items = [], [], [[] for _ in range(J)]
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(",")
            uid.append(line[0])
            pos_item.append(line[1])
            for i in range(J):
                neg_items[i].append(line[i+2])
    return uid, pos_item, neg_items
uid, pos_item, neg_items = load_train("train.raw")




#读取预训练的embedding
import numpy as np
def load_emb(path):
    emb = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split(" ")
            if(len(line) < 5): continue
            emb.setdefault(line[0], np.array([float(v) for v in line[1:]]))
    return emb
emb = load_emb("embmodel.dat")

#生成dict:{itemid -> itemindex}
def get_item2index(items):
    res = {}
    for item in items:
        if item not in res:
            res[item] = len(res) + 1
    return res
dict_item2index = get_item2index(emb.keys())

#生成dict:{itemindex -> itemvec}
def get_index2vec(item2index, item2vec, emb_dim):
    mat = np.zeros((len(item2index)+1, emb_dim))
    for item, i in item2index.items():
        if i < len(item2index)+1:
            vec = item2vec.get(item)
            if vec is not None:
                mat[i] = vec
    return mat
dict_index2vec = get_index2vec(dict_item2index, emb, 32)




#读取用户review历史
def load_review(path):
    res = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split("\t")
            if(len(line) != 2): continue
            uid, reviews = line
            res.setdefault(line[0], reviews.split(";"))
    return res
review = load_review("review.raw")

#取user的最近N条review
N = 5
user_input = np.zeros((len(uid), N))
for i,user in enumerate(uid):
    if user in review:
        item_list = list(map(lambda x: dict_item2index[x] if x in dict_item2index else 0, review[user][-N:]))
        user_input[i][:len(item_list)] = item_list
user_input = user_input.astype(int)




#正/负例的input
def make_input(items, item2index):
    res = np.zeros((len(items)))
    for i in range(len(items)):
        if items[i] in item2index:
            res[i] = item2index[items[i]]
        else:
            res[i] = 0
    return res
pos_input = make_input(pos_item, dict_item2index)
neg_inputs = [make_input(neg_item, dict_item2index) for neg_item in neg_items]




#label
y = np.zeros((len(user_input), 1+4))
y[:,0] = 1



#模型设计
from tensorflow.keras.layers import Activation, Input, Embedding, Flatten, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, GlobalAveragePooling1D
from tensorflow.keras.layers import dot
from tensorflow.keras import backend as K
from tensorflow.keras.layers import concatenate

from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow import keras

def dssm(index2vec, max_reviews=5, dim=32, J=4):
    user_input = Input(shape=(max_reviews,), name='user_input')
    pos_input = Input(shape=(1,), name='pos_input')
    neg_inputs = [Input(shape=(1,)) for _ in range(J)]

    user_embedding = Embedding(len(index2vec), dim, weights=[index2vec], input_length=max_reviews, trainable=False)(user_input)
    user_average = GlobalAveragePooling1D()(user_embedding)
    user_fc = Dense(32, activation='relu', name='ufc')(user_average)

    pos_embedding = Embedding(len(index2vec), dim, weights=[index2vec], trainable=False)(pos_input)
    neg_embeddings = [Embedding(len(index2vec), dim, weights=[index2vec], trainable=False)(neg_input) for neg_input in neg_inputs]

    pos_flatten = Flatten()(pos_embedding)
    neg_flattens = [Flatten()(neg_embedding) for neg_embedding in neg_embeddings]

    item_fc = Dense(32, activation='relu', name='ifc')

    pos_fc = item_fc(pos_flatten)
    neg_fcs = [item_fc(neg_flatten) for neg_flatten in neg_flattens]

    user_product_pos = dot([user_fc, pos_fc], axes=1, normalize=True)
    user_product_negs = [dot([user_fc, neg_fc], axes=1, normalize=True) for neg_fc in neg_fcs]

    concat = concatenate([user_product_pos] + user_product_negs)

    ctr = Activation("softmax")(concat)

    model = Model(inputs=[user_input, pos_input] + neg_inputs, outputs=ctr)
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['acc'])
    return model


model = dssm(dict_index2vec, max_reviews=5)
print(model.summary())


#开始训练
history = model.fit([user_input, pos_input] + neg_inputs, y, batch_size=1024, epochs=10, verbose=2, validation_split=0.2, shuffle=True)



#可视化
import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()




#实时可视化
d = MyCallback(runtime_plot=True)
model.fit([user_input, pos_input] + neg_inputs, y, batch_size=1024, epochs=10, verbose=0, validation_split=0.2, shuffle=True, callbacks=[d])



#自定义回调函数
import pylab as pl
from IPython import display
from tensorflow.keras.callbacks import Callback

class MyCallback(Callback):

    #初始化参数和数据
    def __init__(self, runtime_plot=True):
        super().__init__()
        self.init_loss = None
        self.runtime_plot = runtime_plot
        
        self.xdata = []
        self.ydata = []
        
    #具体如何画图
    def _plot(self, epoch=None):
        epochs = self.params.get("epochs")
        pl.ylim(0, int(self.init_loss*2))
        pl.xlim(0, epochs)
    
        pl.plot(self.xdata, self.ydata)
        pl.xlabel('Epoch {}/{}'.format(epoch or epochs, epochs))
        pl.ylabel('Loss {:.4f}'.format(self.ydata[-1]))
        
    def _runtime_plot(self, epoch):
        self._plot(epoch)
        
        display.clear_output(wait=True)
        display.display(pl.gcf())
        pl.gcf().clear()

    def plot(self):
        self._plot()
        pl.show()

    #在每一个epoch结束被调用来画图
    def on_epoch_end(self, epoch, logs = None):
        logs = logs or {}
        loss = logs.get("loss")
        if self.init_loss is None:
            self.init_loss = loss
        self.xdata.append(epoch)
        self.ydata.append(loss)
        if self.runtime_plot:
            self._runtime_plot(epoch)

