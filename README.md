# Classifying MNIST digits using Convolutional Neural Networks

In this project, I show how TensorFlow can be used to implement convolutional neural networks.
Also, I briefly introduce basic concepts of convolutional neural networks.
The mathematical notations and its mapping TensowFlow graphs can be seen at below.
The accuracy of this model for Kaggle competition: **0.99**

## MNIST dataset
> The MNIST database (Mixed National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, NIST's complete dataset was too hard.[5] Furthermore, the black and white images from NIST were normalized to fit into a 20x20 pixel bounding box and anti-aliased, which introduced grayscale levels.

> The MNIST database contains 60,000 training images and 10,000 testing images. Half of the training set and half of the test set were taken from NIST's training dataset, while the other half of the training set and the other half of the test set were taken from NIST's testing dataset. There have been a number of scientific papers on attempts to achieve the lowest error rate; one paper, using a hierarchical system of convolutional neural networks, manages to get an error rate on the MNIST database of 0.23 percent. The original creators of the database keep a list of some of the methods tested on it. In their original paper, they use a support vector machine to get an error rate of 0.8 percent. -- "Wikipedia: MNIST database"


## Kaggle competition on digit recognizer
**Link: https://www.kaggle.com/c/digit-recognizer **

The goal in this competition is to take an image of a handwritten single digit, and determine what that digit is.  As the competition progresses, we will release tutorials which explain different machine learning algorithms and help you to get started.


The data for this competition were taken from the MNIST dataset. The MNIST ("Modified National Institute of Standards and Technology") dataset is a classic within the Machine Learning community that has been extensively studied.  More detail about the dataset, including Machine Learning algorithms that have been tried on it and their levels of success, can be found at http://yann.lecun.com/exdb/mnist/index.html.


### Competition dataset
The data files train.csv and test.csv contain gray-scale images of hand-drawn digits, from zero through nine.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like pixelx, where x is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed x as x = i * 28 + j, where i and j are integers between 0 and 27, inclusive. Then pixelx is located on row i and column j of a 28 x 28 matrix, (indexing by zero).

For example, pixel31 indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:
```
000 001 002 003 ... 026 027
028 029 030 031 ... 054 055
056 057 058 059 ... 082 083
 |   |   |   |  ...  |   |
728 729 730 731 ... 754 755
756 757 758 759 ... 782 783 
```
The test data set, (test.csv), is the same as the training set, except that it does not contain the "label" column.

Your submission file should be in the following format: For each of the 28000 images in the test set, output a single line with the digit you predict. For example, if you predict that the first image is of a 3, the second image is of a 7, and the third image is of a 8, then your submission file would look like:
```
3
7
8
(27997 more lines)
```

The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

### Libraries and settings
```python
import numpy as np
import pandas as pd

import tensorflow as tf

# Parameters
LEARNING_RATE = 0.001
TRAINING_EPOCHS = 3000
BATCH_SIZE = 100
DISPLAY_STEP = 10
DROPOUT_CONV = 0.8
DROPOUT_HIDDEN = 0.6
VALIDATION_SIZE = 2000      # Set to 0 to train on all available data
```

### Data preparation
To start, we read given train and test data from each csv file. At first we read train.csv file.
```python
# Read MNIST data set (Train data from CSV file)
data = pd.read_csv('./input/train.csv')
```

The data contains label and written images for number. `[label pixel_0, pixel_1, ... , pixel_784]`
So, we split data into label and image from each row.
```python
# Extracting images and labels from given data
# For images
images = data.iloc[:,1:].values
images = images.astype(np.float)

# For labels
labels_flat = data[[0]].values.ravel()
labels_count = np.unique(labels_flat).shape[0]
```

For easy implementation of output layer, we convert label with number into ont-hot-vector.
You can refer the idea of one-hot on this [link](https://en.wikipedia.org/wiki/One-hot).

For example, we convert the numbers as follow: 
`0:[1 0 0 0 0 0 0 0 0 0]`
`1:[0 1 0 0 0 0 0 0 0 0]`
...
`9:[0 0 0 0 0 0 0 0 0 1]`
```python
labels = dense_to_one_hot(labels_flat, labels_count)
labels = labels.astype(np.uint8)
```

```python
def dense_to_one_hot(labels_dense, num_classes):
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot
```

Then we normalize the intensity of each pixel from [0:255] into [0.0:1:0]

```python
# Normalize from [0:255] => [0.0:1.0]
images = np.multiply(images, 1.0 / 255.0)
image_size = images.shape[1]
image_width = image_height = np.ceil(np.sqrt(image_size)).astype(np.uint8)
```

Before applying our trained model to test data, we validate our trained model using validation dataset.
So, we split training data into [train, validation].
```python
# Split data into training & validation
validation_images = images[:VALIDATION_SIZE]
validation_labels = labels[:VALIDATION_SIZE]

train_images = images[VALIDATION_SIZE:]
train_labels = labels[VALIDATION_SIZE:]

```


### Create CNN model with TensorFlow graph

We start creating cnn model with definition of input and output.
This model handle each image and make decision for the image with digit classes [0-9].
```python
# Create Input and Output
X = tf.placeholder('float', shape=[None, image_size])       # mnist data image of shape 28*28=784
Y_gt = tf.placeholder('float', shape=[None, labels_count])    # 0-9 digits recognition => 10 classes
```

Using below functions, we can generate weight and bias easily.
Basically, the simple weight and bias are generated on normal distribution.
For better result, we implemented Xavier's initialization with input and output connections.
For the detail explanation, you can refer two blogs: [deepdish](http://deepdish.io/2015/02/24/network-initialization/) and [andyljones](http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization).
```python
# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# Weight initialization (Xavier's init)
def weight_xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)

# Bias initialization
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
```

Using above functions, we make two convolutional layers, and two fully connected layers.

```python
# Model Parameters
W1 = tf.get_variable("W1", shape=[5, 5, 1, 32], initializer=weight_xavier_init(5*5*1, 32))
W2 = tf.get_variable("W2", shape=[5, 5, 32, 64], initializer=weight_xavier_init(5*5*32, 64))
W3_FC1 = tf.get_variable("W3_FC1", shape=[64*7*7, 1024], initializer=weight_xavier_init(64*7*7, 1024))
W4_FC2 = tf.get_variable("W4_FC2", shape=[1024, labels_count], initializer=weight_xavier_init(1024, labels_count))

B1 = bias_variable([32])
B2 = bias_variable([64])
B3_FC1 = bias_variable([1024])
B4_FC2 = bias_variable([labels_count])

drop_conv = tf.placeholder('float')
drop_hidden = tf.placeholder('float')
```

### Model description
At first, we transform from 1D input vector into 2D image. 
For the convolutional layer, we apply three steps:
1. Convolution
2. Max-pooling
3. Dropout

For the fully connected layer, the process is same with basic neural network.

```python
# CNN model
X1 = tf.reshape(X, [-1,image_width , image_height,1])                   # shape=(?, 28, 28, 1)
    
# Layer 1
l1_conv = tf.nn.relu(conv2d(X1, W1) + B1)                               # shape=(?, 28, 28, 32)
l1_pool = max_pool_2x2(l1_conv)                                         # shape=(?, 14, 14, 32)
l1_drop = tf.nn.dropout(l1_pool, drop_conv)

# Layer 2
l2_conv = tf.nn.relu(conv2d(l1_drop, W2)+ B2)                           # shape=(?, 14, 14, 64)
l2_pool = max_pool_2x2(l2_conv)                                         # shape=(?, 7, 7, 64)
l2_drop = tf.nn.dropout(l2_pool, drop_conv) 

# Layer 3 - FC1
l3_flat = tf.reshape(l2_drop, [-1, W3_FC1.get_shape().as_list()[0]])    # shape=(?, 1024)
l3_feed = tf.nn.relu(tf.matmul(l3_flat, W3_FC1)+ B3_FC1) 
l3_drop = tf.nn.dropout(l3_feed, drop_hidden)


# Layer 4 - FC2
Y_pred = tf.nn.softmax(tf.matmul(l3_drop, W4_FC2)+ B4_FC2)              # shape=(?, 10)
```

### Cost function and Training
We define cross-entropy for the cost function.
And penalize using L2-regularization.
```python
# Cost function and training 
cost = -tf.reduce_sum(Y_gt*tf.log(Y_pred))
regularizer = (tf.nn.l2_loss(W3_FC1) + tf.nn.l2_loss(B3_FC1) + tf.nn.l2_loss(W4_FC2) + tf.nn.l2_loss(B4_FC2))
cost += 5e-4 * regularizer

#train_op = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)
train_op = tf.train.RMSPropOptimizer(LEARNING_RATE, 0.9).minimize(cost)
correct_predict = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_gt, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predict, 'float'))
predict = tf.argmax(Y_pred, 1)
```

### TensorFlow session
```python
epochs_completed = 0
index_in_epoch = 0
num_examples = train_images.shape[0]

# start TensorFlow session
init = tf.initialize_all_variables()
sess = tf.InteractiveSession()

sess.run(init)

# visualisation variables
train_accuracies = []
validation_accuracies = []

DISPLAY_STEP=1

for i in range(TRAINING_EPOCHS):

    #get new batch
    batch_xs, batch_ys = next_batch(BATCH_SIZE)        

    # check progress on every 1st,2nd,...,10th,20th,...,100th... step
    if i%DISPLAY_STEP == 0 or (i+1) == TRAINING_EPOCHS:
        
        train_accuracy = accuracy.eval(feed_dict={X:batch_xs, 
                                                  Y_gt: batch_ys,
                                                  drop_conv: DROPOUT_CONV, 
                                                  drop_hidden: DROPOUT_HIDDEN})       
        if(VALIDATION_SIZE):
            validation_accuracy = accuracy.eval(feed_dict={ X: validation_images[0:BATCH_SIZE], 
                                                            Y_gt: validation_labels[0:BATCH_SIZE],
                                                            drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})                                  
            print('training_accuracy / validation_accuracy => %.2f / %.2f for step %d'%(train_accuracy, validation_accuracy, i))
            
            validation_accuracies.append(validation_accuracy)
            
        else:
             print('training_accuracy => %.4f for step %d'%(train_accuracy, i))
        train_accuracies.append(train_accuracy)
        
        # increase DISPLAY_STEP
        if i%(DISPLAY_STEP*10) == 0 and i:
            DISPLAY_STEP *= 10
    # train on batch
    sess.run(train_op, feed_dict={X: batch_xs, Y_gt: batch_ys, drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})


# check final accuracy on validation set  
if(VALIDATION_SIZE):
    validation_accuracy = accuracy.eval(feed_dict={X: validation_images, 
                                                   Y_gt: validation_labels,
                                                   drop_conv: DROPOUT_CONV, drop_hidden: DROPOUT_HIDDEN})
    print('validation_accuracy => %.4f'%validation_accuracy)

# read test data from CSV file 
test_images = pd.read_csv('./input/test.csv').values
test_images = test_images.astype(np.float)

# convert from [0:255] => [0.0:1.0]
test_images = np.multiply(test_images, 1.0 / 255.0)

print('test_images({0[0]},{0[1]})'.format(test_images.shape))


# predict test set
#predicted_lables = predict.eval(feed_dict={X: test_images, keep_prob: 1.0})

# using batches is more resource efficient
predicted_lables = np.zeros(test_images.shape[0])
for i in range(0,test_images.shape[0]//BATCH_SIZE):
    predicted_lables[i*BATCH_SIZE : (i+1)*BATCH_SIZE] = predict.eval(feed_dict={X: test_images[i*BATCH_SIZE : (i+1)*BATCH_SIZE], drop_conv: 1.0, drop_hidden: 1.0})


# save results
np.savetxt('submission.csv', 
           np.c_[range(1,len(test_images)+1),predicted_lables], 
           delimiter=',', 
           header = 'ImageId,Label', 
           comments = '', 
           fmt='%d')

sess.close()
```



