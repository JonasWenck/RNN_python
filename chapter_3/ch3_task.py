import collections
import os

import numpy as np
import tensorflow as tf

# ignore CPU warning as we use GPU anyway (
# https://stackoverflow.com/questions/47068709/your-cpu-supports-instructions-that-this-tensorflow-binary-was-not
# -compiled-to-u)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_words(file_name):
    """returns an array of all words from the given text file"""
    with open(file_name) as file:
        all_lines = file.readlines()
    lines_without_spaces = [x.strip() for x in all_lines]
    words = []
    for line in lines_without_spaces:
        words.extend(line.split())
    words = np.array(words)
    return words


def build_dictionary(words):
    # tuple with the words and their amounts of occurrence ordered from high to low
    most_common_words = collections.Counter(words).most_common()
    # dictionary of type <word, amount>
    word2id = dict((word, id) for (id, (word, _)) in enumerate(most_common_words))
    # dictionary of type <amount, word>
    id2word = dict((id, word) for (id, (word, _)) in enumerate(most_common_words))
    return most_common_words, word2id, id2word


words = get_words("the_hunger_games_chapter_1_to_8.txt")
most_common_words, word2id, id2word = build_dictionary(words)
# number of different words in our training data
number_of_different_words = len(most_common_words)

# each section in our encoded dataset will contain 20 encoded words
section_length = 20


def input_output_values(words):
    """creates the one-hot-encoded inputs and outputs for our model"""
    input_values = []
    output_values = []
    num_sections = 0
    for i in range(len(words) - section_length):
        # input_values are appended with arrays of the next 20 words
        input_values.append((words[i: i + section_length]))
        # output_values are appended with the 21st word so the word after each 20 word section
        output_values.append(words[i + section_length])
        num_sections += 1

    # input array is of dimensions num_sections (meaning how many section of 20 words we have),
    # 20 (the number of words in each section) and the amount of different words in the training data
    # with this structure, for each section, we can encode the index of each word in that section
    # all indices are initialized with 0, the index of each word is later set to 1
    one_hot_inputs = np.zeros((num_sections, section_length, number_of_different_words))
    # for the outputs, we need the dimensions num_sections (as we have as many outputs as sections) and amount of
    # different words in the training data to store the index of the output word for each section
    one_hot_outputs = np.zeros((num_sections, number_of_different_words))

    for s_index, section in enumerate(input_values):
        for w_index, word in enumerate(section):
            # at the current section, for the current word in that section, set the index of that word (over all words
            # in the training data to 1
            one_hot_inputs[s_index, w_index, word2id[word]] = 1.0
        # for the current section, set the index of the output word
        one_hot_outputs[s_index, word2id[output_values[s_index]]] = 1.0

    return one_hot_inputs, one_hot_outputs


# store the encoded input and output values to global variables
training_X, training_y = input_output_values(words)

# learning rate describes how fast we change the weights in our network
learning_rate = 0.001
# numbers of element per batch
batch_size = 512
# 1 step/iteration = picking one batch and performing forward and backwards propagation, update weights and biases
number_of_iterations = 100000
# number of units in any RNN cell
number_hidden_units = 1024

# X holds training data at the current batch
X = tf.placeholder(tf.float32, shape=[batch_size, section_length, number_of_different_words])
# y holds the predicted data at the current batch
y = tf.placeholder(tf.float32, shape=[batch_size, number_of_different_words])

# use normal distribution for weights and biases
weights = tf.Variable(tf.random.truncated_normal([number_hidden_units, number_of_different_words]))
biases = tf.Variable(tf.random.truncated_normal([number_of_different_words]))

gru_cell = tf.contrib.rnn.GRUCell(num_units=number_hidden_units)

outputs, state = tf.nn.dynamic_rnn(gru_cell, inputs=X, dtype=tf.float32)
# transpose from [batch_size, section_length, number_of_different words] to
# [section_length, batch_size, number_of_different words]
outputs = tf.transpose(outputs, perm=[1, 0, 2])
last_output = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

prediction = tf.matmul(last_output, weights) + biases
# calculate the loss by comparing with the predicted data in y
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=prediction)
total_loss = tf.reduce_mean(loss)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss=total_loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # this makes sure that the right batch is extracted from the training data
    iter_offset = 0

    # periodically save the model locally
    saver = tf.train.Saver()

    # perform the training iterations
    for iter in range(number_of_iterations):
        length_X = len(training_X)

        if length_X != 0:
            iter_offset = iter_offset % length_X

        if iter_offset <= length_X - batch_size:
            training_X_batch = training_X[iter_offset: iter_offset + batch_size]
            training_y_batch = training_y[iter_offset: iter_offset + batch_size]
        else:
            add_from_the_beginning = batch_size - (length_X - iter_offset)
            training_X_batch = np.concatenate((training_X[iter_offset: length_X], X[0: add_from_the_beginning]))
            training_y_batch = np.concatenate((training_y[iter_offset: length_X, y[0: add_from_the_beginning]]))
            iter_offset = add_from_the_beginning

    _, training_loss = sess.run([optimizer, total_loss], feed_dict={X: training_X_batch, y: training_y_batch})
    print("Loss:", training_loss)
    saver.save(sess, 'ckpt/model', global_step=iter)

print("Done with training!")

starting_sentence = ('I plan to make the world a better place because I love seeing how people grow and do in '
                     'their lives')

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    model = tf.train.latest_checkpoint('ckpt')
    saver = tf.train.Saver()
    saver.restore(sess, model)

    generated_text = starting_sentence
    words_in_starting_sentence = starting_sentence.split()
    test_X = np.zeros((batch_size, section_length, number_of_different_words))
    for index, word in enumerate(words_in_starting_sentence[:-1]):
        if index < section_length:
            test_X[0, index, word2id[word]] = 1

    _ = sess.run(prediction, feed_dict={X: test_X})

    test_last_X = np.zeros((1, 1, number_of_different_words))
    test_last_X[0, 0, word2id[words_in_starting_sentence[-1]]] = 1
    test_next_X = np.reshape(np.concatenate((test_X[0, 1:], test_last_X[0])),
                             (1, section_length, number_of_different_words))


    def prediction_to_one_hot(prediction):
        zero_array = np.zeros(np.shape(prediction))
        zero_array[np.argmax(prediction)] = 1
        return zero_array


    for i in range(1000):
        # the text_next_X has the shape (1, 20, 6257) but our X needs the shape (512, 20, 6257) so we repeat 512
        # times to achieve the required shape
        test_next_X_repeated = np.repeat(test_next_X, 512, axis=0)
        test_prediction = prediction.eval({X: test_next_X_repeated})[0]
        next_word_one_hot = prediction_to_one_hot(test_prediction)
        next_word = id2word[np.argmax(next_word_one_hot)]
        generated_text += " " + next_word + " "
        test_next_X = np.reshape(
            np.concatenate((test_next_X[0, 1:], np.reshape(next_word_one_hot, (1, number_of_different_words)))),
            (1, section_length, number_of_different_words))
        print("Generated text: ", generated_text)
