from model import unet
from data import trainGenerator,testGenerator,saveResult
import os
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def show_train_history(train_history,train):
    plt.plot(train_history.history[train])
    plt.title('Train history')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train'],loc = 'upper left')
    plt.show()

data_gen_args = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')
myGene = trainGenerator(2,'data/membrane/train', 'image', 'label', data_gen_args, save_to_dir=None)

model = unet()
model_checkpoint = ModelCheckpoint('unet_membrane.hdf5', monitor='loss',verbose=1, save_best_only=True)
history = model.fit_generator(myGene, steps_per_epoch=30, epochs=50, callbacks=[model_checkpoint])
# 显示训练准确率
show_train_history(history,'acc')
# 显示误差率图
show_train_history(history,'loss')

testGene = testGenerator("data/membrane/test")
results = model.predict_generator(testGene, 30, verbose=1)
saveResult("data/membrane/test", results)
