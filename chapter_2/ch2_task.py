import tensorflow as tf
import random

num_examples = 100000
num_classes = 50


def input_values():
    """Generates 100,000 input values with byte representations of numbers between 0 and 2^20"""
    # create binary representation of the first 2^20 numbers (=1,048,576)
    multiple_values = [map(int, '{0:050b}'.format(i)) for i in range(2 ** 20)]
    # shuffle the numbers so that similar values are not next to each other (important to prevent bias during training)
    random.shuffle(multiple_values)
    final_values = []
    # slice our 2^20 numbers for the first 100,000 entries
    for value in multiple_values[:num_examples]:
        temp = []
        # copy the values to our final_values array
        for number in value:
            temp.append([number])
        final_values.append(temp)
    return final_values


def output_values(inputs):
    """returns a list of one-hot encoded representations of the inputs"""
    final_values = []
    for value in inputs:
        output_values = [0 for _ in range(num_classes)]
        count = 0
        for i in value:
            count += i[0]
        if count < num_classes:
            output_values[count] = 1
        final_values.append(output_values)
    return final_values


def generate_data():
    """generate the input and output values"""
    inputs = input_values()
    return inputs, output_values(inputs)


# we train in batches as that requires less memory and is faster
# tensor = generalization of vectors and matrices; represented as n-dimensional arrays of base datatypes
def train_model():
    # None means that the tensor can decide on that dimension
    X = tf.placeholder(tf.float32, shape=[None, num_classes, 1])
    Y = tf.placeholder(tf.float32, shape=[None, num_classes])

    # describes the dimensionality of the memory state and the output and therefore the learning capacity
    num_hidden_units = 24

    # weights and biases can be modified during training, their type and shape cannot be changed
    weights = tf.Variable(tf.truncated_normal([num_hidden_units, num_classes]))
    biases = tf.Variable(tf.truncated_normal([num_classes]))

    # we could add a custom activation function with the param 'activation', defaults to tanh
    rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=num_hidden_units)
    # produce network output
    outputs1, state = tf.nn.dynamic_rnn(rnn_cell, inputs=X, dtype=tf.float32)
    outputs = tf.transpose(outputs1, [1, 0, 2])

    last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)
    prediction = tf.matmul(last_output, weights) + biases

    # softmax emphasizes the largest value and suppresses values significantly below the max value
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=prediction)
    total_loss = tf.reduce_mean(loss)

    # used to optimize the loss function
    learning_rate = 0.001
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)

    # initialize batch size
    batch_size = 1000
    number_of_batches = int(num_examples / batch_size)
    # how many times the model should loop through the training set
    epoch = 100
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        X_train, y_train = generate_data()
        for epoch in range(epoch):
            iter = 0
            for _ in range(number_of_batches):
                training_x = X_train[iter:iter + batch_size]
                training_y = y_train[iter:iter + batch_size]
                iter += batch_size
                _, current_total_loss = sess.run([optimizer, total_loss], feed_dict={X: training_x, Y: training_y})
                # if the value of the loss function decreases, the weights and biases are successfully modified
                print("Epoch:", epoch, "Iteration:", iter, "Loss:", current_total_loss)
                print("________________________")

        test_example = [
            [[1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1],
             [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0], [0], [1], [1], [0], [1], [1], [1], [0], [1], [0],
             [0], [1], [1], [0], [1], [1], [1], [0]]]
        prediction_result = sess.run(prediction, {X: test_example})
        # find the index of the largest number
        largest_number_index = prediction_result[0].argsort()[-1:][::-1]

        print("Predicted sum: ", largest_number_index, "Actual sum:", 30)
        print("The predicted sequence parity is ", largest_number_index % 2, "and it should be")

# with epoch = 100
# Epoch: 99 Iteration: 100000 Loss: 0.11412848

# with epoch = 200
# Epoch: 199 Iteration: 100000 Loss: 0.06389284
