import numpy
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
 
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

batch_size=32

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=(100, 30),
        batch_size=batch_size,
        class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=(100, 30),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

num_classes = len(train_generator.class_indices)

def larger_model():
	# create model
	model = Sequential()
	model.add(Convolution2D(30, (5, 5), padding='valid', input_shape=train_generator.image_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

# build the model
model = larger_model()

# Fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch=3869 // batch_size, 
        epochs=10,
        verbose=1)

# Final evaluation of the model
test_steps_per_epoch = numpy.math.ceil(float(test_generator.samples) / test_generator.batch_size)
raw_predictions = model.predict_generator(test_generator, steps=test_steps_per_epoch)
predictions = numpy.argmax(raw_predictions, axis=1)

print(numpy.bincount(test_generator.classes))
print(numpy.bincount(predictions))

from sklearn import metrics
class_labels = list(test_generator.class_indices.keys())
print(metrics.classification_report(test_generator.classes, predictions, target_names=class_labels))
