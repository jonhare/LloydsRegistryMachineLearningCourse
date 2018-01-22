# Learning to Deep Learn using Python, Keras, TensorFlow and a GPU

_[Jonathon Hare, 21st Jan 2018](https://github.com/jonhare/LloydsRegistryMachineLearningCourse)_

## Change History

- 20170121: Initial version

## Introduction

Now we've seen how we can use Keras to work towards the solution of a handwriting recognition problem, we'll turn our focus to data that is more realistic (and focussed on a maritime application), using deep-learning models that are much closer to state of the art. The problem with using better models is that we need massive amounts of labelled data to train these models from scratch, and also large amounts of time (typically days of training, even using multiple GPUs). Rather than training from scratch we'll explore using transfer learning and fine-tuning using pre-trained models. The pre-trained models that we'll play with were trained using the ImageNet dataset, which consists of about 1.3 million images in 1000 classes.

Through this part of the tutorial you'll learn how to:

* How to load image data from the file system
* How to develop and evaluate a simple CNN for classification.
* How to use custom callbacks to monitor training progress.
* How to load a pre-trained model and use it to make classifications.
* How to modify and fine-tune a pre-trained model to solve the a classification problem.
* How to extract _semantic_ features that can be used for transfer learning and finding similar features.

## Prerequisites
As with part 1 of the tutorial, you'll use Python 3 language the `keras`. We'll also again be using the `scikit-learn` and `numpy` packages.

You'll need access to a computer with the following installed:

- `Python` (> 3.6)
- `keras` (>= 2.0.0)
- `tensorflow` (>= 1.0.0)
- `NumPy` (>= 1.12.1)
- `SciPy` (>= 0.19.1)
- `scikit-learn` (>= 0.19.1)
- `opencv`
- `pillow` (>=4.0.0)

## Getting started 
Start by downloading and unzipping the data set:

```
wget
unzip data.zip
```

We'll start by exploring the data, and look at how we can get that data loaded into memory through python code. If you open the data directory you should see three folders:
	- The `train` folder contains the training data & is broken into subdirectories for each class. 
	- The `valid` folder contains the validation data & is broken into subdirectories for each class. 
	- The `test` folder contains the testing data & is broken into subdirectories for each class. 

The keras library has support for directly reading images from a directory structure like the one we have using the `ImageDataGenerator` class. In addition to loading the images directly, keras provides a mechanism to dynamically augment the data being read by applying random transformations (flipping, rotating, etc), as well as cropping and scaling the images. The following code will generate a visualisation of the first batch of images produced by the generator:

```python
# Plot ad hoc data instances
from keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy

datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
        'data/train',
        target_size=(240, 800),
        batch_size=4,
        class_mode='categorical')

# generate the first batch
(batch_images, batch_labels) = generator.next()

class_labels = [item[0] for item in sorted(generator.class_indices.items(), key=lambda x: x[1])] #get a list of classes
batch_labels = numpy.argmax(batch_labels, axis=1) #convert the one-hot labels to indices

# plot 4 images
plt.subplot(221).set_title(class_labels[batch_labels[0]])
plt.imshow(batch_images[0], aspect='equal')
plt.subplot(222).set_title(class_labels[batch_labels[1]])
plt.imshow(batch_images[1], aspect='equal')
plt.subplot(223).set_title(class_labels[batch_labels[2]])
plt.imshow(batch_images[2], aspect='equal')
plt.subplot(224).set_title(class_labels[batch_labels[3]])
plt.imshow(batch_images[3], aspect='equal')

# show the plot
plt.show()
plt.savefig("batch.png")
```

You can see that accessing the dataset is quite easy. The most important caveat of using the `ImageDataGenerator` comes when we are using it to load the test data - in such a case we need to ensure that no augmentation happens (other than the resizing of inputs through the `target_size` attribute of `flow_from_directory`), and that the `shuffle` attribute of `flow_from_directory` is `False`, to ensure that we can compare the true labels and target labels correctly.

![Examples from the dataset](https://raw.githubusercontent.com/jonhare/LloydsRegistryMachineLearningCourse/master/Thursday/practical-part2/batch.png "Examples from the dataset")

## A simple CNN for boat classification

Now let's try something a little more challenging and take our _larger_ convolutional network from the experiments with mnist and apply it to the problem of boat classification. Firstly we need to setup the data for training (this time using the generator so we don't have to worry about memory usage), and it would also be sensible to load the validation data to monitor performance during training, as well as load the test data:

```python
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

# the number of images that will be processed in a single step
batch_size=32
# the size of the images that we'll learn on - we'll shrink them from the original size for speed
image_size=(30, 100)

train_generator = train_datagen.flow_from_directory(
        'data/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
        'data/valid',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        'data/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

num_classes = len(train_generator.class_indices)
```

Note that for now we're not using any data augmentation from the training data, however we've structured the code so that we can easily add it by manipulating the `ImageDataGenerator` that creates the `train_datagen` object. 

Now we can add the network definition from part 1. We'll make a slight change to the previous `larger_model()` function so that it allows us to specify the input and output sizes, and we'll also pull out the compile statement as it would be the same for many model architectures:

```python
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D

def larger_model(input_shape, num_classes):
	# create model
	model = Sequential()
	model.add(Convolution2D(30, (5, 5), padding='valid', input_shape=input_shape, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Convolution2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	
	return model

# build the model
model = larger_model(train_generator.image_shape, num_classes)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```

Specifying the input shape using the shape given by the `train_generator.image_shape` allows us to avoid having to worry about how the backend is storing the images. We're now in a position to add the code to fit the model. Because this time we're loading the data using a generator rather than statically we use the `fit_generator()` method instead of `fit`:

```python
# Fit the model
# Fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch=3474 // batch_size, 
        validation_data=valid_generator,
        validation_steps=395 // batch_size,
        epochs=10,
        verbose=1)
```

We've specified `3474 // batch_size` `steps_per_epoch` to indicate that we want all images (there are 3474 training images) to be processed each epoch (each step within an epoch will process a single batch worth of images). The same applies for the `validation_steps`.

Finally, before we try running this model, lets make use of the test data and print a classification report using scikit-learn:

```python
# Final evaluation of the model
# Compute the number of epochs required so we see all data:
test_steps_per_epoch = numpy.math.ceil(float(test_generator.samples) / test_generator.batch_size)
# perform prediction:
raw_predictions = model.predict_generator(test_generator, steps=test_steps_per_epoch)
# convert predictions from one-hot to indices
predictions = numpy.argmax(raw_predictions, axis=1)

print("Prediction Distribution:  " + numpy.bincount(test_generator.classes))
print("Groundtruth Distribution: " + numpy.bincount(predictions))

from sklearn import metrics
#get a list of classes (this basically ensures that the list is in the correct order by index):
class_labels = [item[0] for item in sorted(generator.class_indices.items(), key=lambda x: x[1])] 
#print the report
print(metrics.classification_report(test_generator.classes, predictions, target_names=class_labels))
```

Running this should result in the following:

	Using TensorFlow backend.
	Epoch 1/10
	10016/10016 [==============================] - 25s - loss: 0.9179 - acc: 0.7318 - val_loss: 0.6572 - val_acc: 0.8310
	Epoch 2/10
	10016/10016 [==============================] - 7s - loss: 0.6426 - acc: 0.8190 - val_loss: 0.6077 - val_acc: 0.8490
	Epoch 3/10
	10016/10016 [==============================] - 7s - loss: 0.6037 - acc: 0.8259 - val_loss: 0.5387 - val_acc: 0.8500
	Epoch 4/10
	10016/10016 [==============================] - 7s - loss: 0.5786 - acc: 0.8242 - val_loss: 0.5549 - val_acc: 0.8550
	Epoch 5/10
	10016/10016 [==============================] - 7s - loss: 0.5646 - acc: 0.8359 - val_loss: 0.5586 - val_acc: 0.8520
	Epoch 6/10
	10016/10016 [==============================] - 7s - loss: 0.5355 - acc: 0.8404 - val_loss: 0.5150 - val_acc: 0.8560
	Epoch 7/10
	10016/10016 [==============================] - 7s - loss: 0.5466 - acc: 0.8334 - val_loss: 0.4932 - val_acc: 0.8550
	Epoch 8/10
	10016/10016 [==============================] - 7s - loss: 0.5469 - acc: 0.8352 - val_loss: 0.5890 - val_acc: 0.8580
	Epoch 9/10
	10016/10016 [==============================] - 7s - loss: 0.5180 - acc: 0.8427 - val_loss: 0.5248 - val_acc: 0.8610
	Epoch 10/10
	10016/10016 [==============================] - 7s - loss: 0.5025 - acc: 0.8481 - val_loss: 0.5215 - val_acc: 0.8550

with a set of sample predictions that looks something like this:

![Example classifications](https://github.com/jonhare/os-deep-learning-labs/raw/master/part2/images/cnn1-class.png "Example classifications")

In this particular case the overall accuracies are all quite high (in terms of both training and validation), which is pleasing. Be aware though that we're using a relatively small set of both training and validation data, and that there is a very high bias in the class distribution which inevitably could lead to higher accuracies because of common classes.

> __Exercise:__ Have a play with the above code and explore the effect of patch size and the amount of training and validation data.

## Mapping the classifications
Now we can make predictions for a tile it would be quite nice to make a modification so that we can try and reconstruct theme maps directly from the 3-band images. This is often called semantic segmentation. An easy approach that we can take is to take spatially sequential patches, and reconstruct an "image" based on the class assignments. This is effective, but highly computationally inefficient - better approaches are to convert the network to be fully convolutional (see e.g. https://devblogs.nvidia.com/parallelforall/image-segmentation-using-digits-5/#comment-1891), or to use a specialist type of network that actually outputs segmentation maps directly (usually these networks are based on layers of downsampling convolutions like we're using in our network, followed by layers of upsampling or deconvolution which aim to increase the spatial resolution back up to the size of the input). 

For now, let's implement the naive approach by building a theme map for some test data. We'll keep the same code as above, but modify the parts involving the test data as follows:

```python
import numpy as np

# load some test data; this time we specifically load patches from a 300x300 square of a tile in scan-order
test_data = load_labelled_patches(["SU4012"], patch_size, subcoords=((0,0), (300,300)))

# we can reshape the test data labels back into an image and save it
tmp = np.zeros(test_data[1].shape[0])
for x in xrange(0, test_data[1].shape[0]):
	tmp[x] = test_data[1][x].argmax()
tmp = tmp.reshape((300-patch_size, 300-patch_size))
plt.figure()
plt.imshow(tmp)
plt.savefig("test_gt.png")

# and we can do the same for the predictions
clzs = model.predict_classes(test_data[0])
clzs = clzs.reshape((300-patch_size, 300-patch_size))
plt.figure()
plt.imshow(clzs)
plt.savefig("test_pred.png")
```

If we now run this, we'll get a "ground-truth" image that looks like this:

![Test ground-truth theme map](https://github.com/jonhare/os-deep-learning-labs/raw/master/part2/images/test_gt.png "Test ground-truth theme map")

and a "predictions" image that looks something like this (results will vary depending on the sample of data used for training):

![Test predictions theme map](https://github.com/jonhare/os-deep-learning-labs/raw/master/part2/images/test_pred.png "Test predictions theme map")

In this case we can see that the result is not particularly good, although undoubtedly this is due to the tiny amount of training our network has had as well as the fact that our predictions are entirely based on looking at a 28x28 pixel window of the image (bearing in mind that this is only 7m x 7m on the ground, which is pretty tiny and obviously fails to capture any context from the surroundings). 

One additional change we might like to make to our code is to add a "callback" that produces and saves a theme-map for the validation data after each epoch; this will allow us to visually monitor how the network is learning. Firstly we need to modify the validation data to be loaded from a region of a tile in scan order rather than from random sampling:

```python
valid_data = load_labelled_patches(["SU4011"], patch_size, subcoords=((0,0), (300,300)))
```

Next we need to define the callback by extending the `keras.callbacks.Callback` class, and implement the `on_epoch_end` method to make predictions on the validation data, reformat them into an image and save the result:

```python
class DisplayMap(keras.callbacks.Callback):
	def on_epoch_end(self, epoch, logs={}):
		clzs = model.predict_classes(valid_data[0])
		clzs = clzs.reshape((300-patch_size, 300-patch_size))
		plt.figure()
		plt.imshow(clzs)
		plt.savefig("map_epoch%s.png" % epoch)
```

Finally, we need to modify the call to `fit_generator` to include the callback:

```python
model.fit_generator(train_data, steps_per_epoch=313, epochs=10, validation_data=valid_data, verbose=1, callbacks=[DisplayMap()])
```

If we run the code now after each epoch has passed an image will be saved. Here's the image from the first epoch:

![Epoch 1 theme map](https://github.com/jonhare/os-deep-learning-labs/raw/master/part2/images/map_epoch0.png "Epoch 1 theme map")

> __Exercise:__ It's a little difficult to interpret whether the above validation theme map is actually any good because we don't exactly know what it should look like (we can compare against the ground-truth in the data directory, but this is a different size). Add some additional code to save the validation data ground-truth theme map before training starts so we have something to compare against.

## Using a better network model - transferring and finetuning a pretrained ResNet
Training a network from scratch can be a lot of work. Is there some way we could take an existing network trained on some data with one set of labels, and adapt it to work on a different data set with different labels? Assuming that the inputs of the network are equivalent (for example, image with the same number of bands and size), then the answer is an emphatic yes! This process of "finetuning" a pre-trained network has become common-place as its much faster an easier than starting from scratch. 

Let's try this in practice - we'll start by loading a pre-trained network architecture called a Deep Residual Network (or ResNet for short) that has been trained on the 1000-class ImageNet dataset. The ResNet architecture is very deep - it has many (in our case 50) convolutional layers and is currently one of the best performing architectures on the ImageNet challenge. The tutorial git repo contains code that implements the resnet50 architecture, and automatically downloads the pre-trained model weights. We'll start by using this to load the model and test it by classifying an image:

```python
from resnet50 import ResNet50
from keras.preprocessing import image
from imagenet_utils import decode_predictions, preprocess_input
import numpy as np

model = ResNet50(include_top=True, weights='imagenet')

img_path = 'images/mf.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
print('Predicted:', decode_predictions(preds))
```

If we run this (it will take a little longer the first time as the model is downloaded), it should print the following:

	Using TensorFlow backend.
	K.image_dim_ordering: tf
	('Predicted:', [[(u'n02640242', u'sturgeon', 0.35990164), (u'n02641379', u'gar', 0.3399567), (u'n02514041', u'barracouta', 0.26639012), (u'n02536864', u'coho', 0.028537149), (u'n01484850', u'great_white_shark', 0.0025088955)]])

Indicating that our input image was likely to contain a fish!

> __Exercise:__ try the model with some of your own images

We're now in a position to start to hack the model structure. Fundamentally we need to first remove the classification layer at the end of the model and replace it with a new one (with a different number of classes):

```python
def hack_resnet(num_classes):
	model = ResNet50(include_top=True, weights='imagenet')

	# Get input
	new_input = model.input
	# Find the layer to connect
	hidden_layer = model.layers[-2].output
	# Connect a new layer on it
	new_output = Dense(num_classes) (hidden_layer)
	# Build a new model
	newmodel = Model(new_input, new_output)

	return newmodel
```

The actual process of finetuning involves us now training the model with our own data. This is as simple as compiling the model and running the `fit` or `fit_generator` methods as before. As the network is already largely trained, we'll likely want to use a small learning rate so not to make big changes in weights:

```python
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])
```

Often we'll first "freeze" the weights of the already trained layers whilst we learn initial weights for our new layer to avoid overfitting before training:

```python
# set weights in all but last layer
# to non-trainable (weights will not be updated)
for layer in model.layers[:len(model.layers)-2]:
    layer.trainable = False
```

If we have lots of training data we could then unlock these layers and perform end-to-end finetuning afterwards. The Standford CS231n course pages have lots of useful hints on fine-tuning: http://cs231n.github.io/transfer-learning/

> __Exercise:__ try finetuning the resnet50 with the theme data. You'll need a GPU to do this effectively as it's _very_ slow! 

## Extracting features from a model

Sometimes you want to do things that are not so easily accomplished with a deep network. You might want to build classifiers using very small amounts of data, or you might want a way of finding things in photographs that are in some way semantically similar, but don't have exactly the same classes. CNNs can actually help here using a technique known often called transfer learning (and related to the fine tuning that we just looked at). If we assume we have a trained network, then by extracting vectors from the layers before the final classifier we should have a means of achieving these tasks, as the vectors are likely to strongly encode semantic information about the content of the input image. If we wanted to quickly train new classifiers for new classes, we could for instance just use a relatively simple linear classifier trained on these vectors. If we wanted to find semantically similar images, we could just compare the Euclidean distance of these vectors.

Keras makes it pretty easy to get these vector representations. First we have to remove the end of out network to the point we with to extract the features - for example in the resnet we might want features from the final flatten layer before the final dense connections (this gives us a 2048 dimensional vector from every 224x224 dimensional inout image):

```python
model = ResNet50(include_top=True, weights='imagenet')

# Get input
new_input = model.input

# Find the layer to end on
new_output = model.layers[-2].output

# Build a new model
newmodel = Model(new_input, new_output)
```

With this model, we can use the `predict()` method to extract the features for some inputs. To demonstrate, we can put the whole thing together and generate some vectors from some samples of our 3-band images

```python
from resnet50 import ResNet50
from imagenet_utils import preprocess_input
from keras.models import Model
from utils import load_labelled_patches


model = ResNet50(include_top=True, weights='imagenet')

# Get input
new_input = model.input

# Find the layer to end on
new_output = model.layers[-2].output

# Build a new model
newmodel = Model(new_input, new_output)

(X, y_test_true) = load_labelled_patches(["SU4012"], 224, limit=4, shuffle=True)
X = preprocess_input(X)

features = newmodel.predict(X)

print features.shape
print features
```

(Obviously this will be more effective if the network has been trained or fine-tuned on the same kind of data that we're extracting features from.)

> __Exercise:__ try generating some features for different patches and calculating the Euclidean distances between these features. How do the Euclidean distances compare to your perception of similarity between the patches?
