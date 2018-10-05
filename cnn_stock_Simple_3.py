# Written by H.Sadeghian


from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import regularizers
from sklearn.model_selection import train_test_split
import os
from sklearn import preprocessing
import numpy as np
import imageio
import glob
import matplotlib.pyplot as plt
from PIL import Image
import math
import pathlib

# initializing
width = 1                                             # width of the figures in inch
height = 1                                            # height of the figures in inch
Alpha = 0.05                                        # the value for price difference range
                                                      #  (log (1-alpha)<log (p2/p1) < log (1+alpha))
batch_size = 32                                       # Batch size for train
num_classes = 3                                       # Number of classes
                                                      # -1:Price is going down,0:Price is sideways,1:Price is going Up
epochs = 50                                         # Number of Epochs for training
save_dir = os.path.join(os.getcwd(), 'saved_models')  # Directory to save the model
model_name = 'stock_simple_model.h5'                  # Name of the model to be saved
folder_fig_name = r'figures'                          # Directory to save the Figures
split_char = ","                                     # The character to split the reading data, "\t" for other files
column_for_high = 2                                   # The Column in file that represents the high value of stock
column_for_low = 3                                    # The Column in file that represents the low value of stock
column_for_volume = 5                                 # The Column in file that represents the volume value of stock
file_name_stock = r'data_google_5min.txt'            # The reading file of stock
number_of_data_to_be_used = 2000                      # Number of Sample in reading file of stock
file_type = 'dec'                                     # If file is descending or ascending
RUN_NAME = "Run with " + str(number_of_data_to_be_used) + " input"   # Log file name with different input
pathlib.Path(folder_fig_name).mkdir(parents=True, exist_ok=True)


def find_returns(data):
    returns = []
    for group in data:
        count = 30
        while count <= (len(group) - 5):
            current_data = group[count - 1]
            future_data = group[count + 4]
            p1 = np.mean(current_data)
            p2 = np.mean(future_data)
            returns.append(math.log(p2 / p1))
            count += 1
    return returns


def get_pixel_values():
    pixels = []
    for filename in glob.glob(folder_fig_name + '\*.png'):
        im = imageio.imread(filename)
        pixels.append(im)
    return pixels


def convert_image():
    for filename in glob.glob(folder_fig_name + '\*.png'):
        img = Image.open(filename)
        img = img.convert('RGB')
        img.save(filename)


def plot_data(data):
    t = np.arange(0, 29, 1)
    file_name_number = 0
    fig = plt.figure(frameon=False, figsize=(width, height))
    for group in data:
        count = 30
        while count <= (len(group) - 5):
            high = []
            low = []
            volume = []
            for item in group[count - 30:count]:
                high.append(item[0])
                low.append(item[1])
                volume.append(item[2])
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
            high = scaler.fit_transform(np.array(high).reshape(-1, 1))
            low = scaler.fit_transform(np.array(low).reshape(-1, 1))
            volume = scaler.fit_transform(np.array(volume).reshape(-1, 1))
            file_name = r'\fig_' + str(file_name_number)
            ax = plt.Axes(fig, [0., 0., 1., 1.])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.plot(t, high[0:-1], 'b', t, low[0:-1], 'g', t, volume[0:-1], 'r')
            fig.savefig(folder_fig_name + file_name, dpi=100)
            fig.clf()
            file_name_number += 1
            count += 1
    print('Created %d files!' % file_name_number)


def load_sample_data():
    original_data = extract_data()
    splitted_data = split_data(original_data)
    useful_data = extract_useful_data(splitted_data)
    return useful_data


def extract_useful_data(data):
    groups = []
    for group in data:
        temp_buffer = []
        for item in group:
            temp = [item[column_for_high], item[column_for_low], item[column_for_volume]]
            temp = [float(i) for i in temp]
            temp_buffer.append(temp)
        groups.append(temp_buffer)
    return groups


def split_data(data):
    groups = []
    for item in data:
        temp_buffer = []
        for string in item:
            number = string.split(split_char)
            temp_buffer.append(number)
        groups.append(temp_buffer)
    return groups


def extract_data():
    infile = open(file_name_stock, 'r')
    temp_buffer = []
    for line in infile:
        temp_buffer.append(line.strip('\n'))
    if file_type == 'acc':
        temp_buffer = temp_buffer[1:number_of_data_to_be_used + 2]
    else:
        temp_buffer = temp_buffer[:len(temp_buffer) - number_of_data_to_be_used - 1:-1]
    i = 0
    groups = []
    temp = []
    for item in temp_buffer:
        if i != number_of_data_to_be_used * 10:
            temp.append(item)
            i += 1
        else:
            groups.append(temp)
            temp = []
            i = 0
    groups.append(temp)
    infile.close()
    return groups


# The data, split between train and test sets:
print('Starting acquiring data from file:', file_name_stock)
data = load_sample_data()
print('Data completely acquired and packed')
print('Starting saving the image data to be inserted in CNN')
plot_data(data)
convert_image()
print('Images saved successfully')

# Add the x and y
x = np.asarray(get_pixel_values())
data_returned = find_returns(data)
y = np.asarray(data_returned)

y[np.where(y <= math.log(1-Alpha))] = -1                            # the price is going down
y[np.where((y > math.log(1-Alpha)) & (y < math.log(1+Alpha)))] = 0  # the price is sideways
y[np.where(y >= math.log(1+Alpha))] = 1                              # the price is going up
number_of_data_to_be_used = math.ceil(x.shape[0] * 0.95)

x_test = x[0:number_of_data_to_be_used]
y_test_raw = y[0:number_of_data_to_be_used]

# Convert class vectors to binary class matrices.
y_test = keras.utils.to_categorical(y_test_raw, num_classes)

x_train, x_valid, y_train, y_valid = train_test_split(x_test, y_test, test_size=0.2, random_state=13)

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_valid.shape[0], 'validation samples')

x_train = x_train.astype('float32')
x_valid = x_valid.astype('float32')
x_train /= 255
x_valid /= 255

# define the model
model = Sequential()

model.add(Conv2D(32, (3, 3),
                 padding='same',
                 activation='relu',
                 input_shape=x_train.shape[1:],
                 kernel_regularizer=regularizers.l2(0.01),  #overfiting mitigation
                 activity_regularizer=regularizers.l1(0.01))
)
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# initiate ADAM optimizer
opt = keras.optimizers.Adam(
    lr=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-8
)

# Let's compile and train the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy']
)

# Create a TensorBoard logger
logger = keras.callbacks.TensorBoard(
    log_dir='logs/{}'.format(RUN_NAME),
    histogram_freq=5,
    write_graph=True
)
keras.callbacks.EarlyStopping(
    monitor='loss',
    patience=0,
    verbose=0,
    mode='auto'
)
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_valid, y_valid),
    shuffle=True,
    callbacks=[logger]
)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_valid, y_valid, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

y_pred_raw = model.predict_classes(x_valid, verbose=1)
y_pred = keras.utils.to_categorical(y_pred_raw, num_classes)
# rr=np.multiply(y_pred,y_valid)
# ss=[sum(e) for e in rr]
# ss.count(1)/ss.__len__()


# Plotting results

print(history.history.keys())

plt.figure(1)

# summarize history for accuracy

plt.subplot(211)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
# plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

# summarize history for loss

plt.subplot(212)
plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
