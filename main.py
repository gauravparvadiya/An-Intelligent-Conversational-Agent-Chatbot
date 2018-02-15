import preprocessing
import tensorflow as tf
import model
import time

# load data
lines, conversation_lines = preprocessing.load_data()

#generate questions and answer from data
questions, answers = preprocessing.generate_question_answer_from_data(lines=lines,conv_lines=conversation_lines)

#clean questions
cleaned_questions = preprocessing.clean_questions(questions=questions)

#clean answers
cleaned_answers = preprocessing.clean_answers(answers=answers)

#filter short and long answers
short_questions, short_answers = preprocessing.filter_long_short_questions_answers(questions=cleaned_questions,
                                                                                   answers=cleaned_answers,
                                                                                   min_length=2,
                                                                                   max_length=20)

#convert vocabulary to int seqence
questions_vocab_to_int, answers_vocab_to_int = preprocessing.question_answer_vocab_to_int(questions=short_questions,
                                           answers=short_answers,
                                           threshold=10)

questions_vocab_to_int,answers_vocab_to_int,short_questions,short_answers=preprocessing.add_tokens(questions_vocab_to_int=questions_vocab_to_int,
                                                                                                   answers_vocab_to_int=answers_vocab_to_int,
                                                                                                   short_questions=short_questions,
                                                                                                   short_answers=short_answers)


#convert question answer to int question answer
questions_int, answers_int = preprocessing.convert_to_int(questions_vocab_to_int=questions_vocab_to_int,
                                                          answers_vocab_to_int=answers_vocab_to_int,
                                                          short_questions=short_questions,
                                                          short_answers=short_answers)


#sort questions and answers
sorted_questions,sorted_answers = preprocessing.sort_question_answer(questions=questions_int,
                                   answers=answers_int,
                                   max_length=20)

print("Preprocessing done...")

epochs = 100
batch_size = 128
rnn_size = 512
num_layers = 2
encoding_embedding_size = 512
decoding_embedding_size = 512
learning_rate = 0.005
learning_rate_decay = 0.9
min_learning_rate = 0.0001
keep_probability = 0.75


tf.reset_default_graph()

sess = tf.InteractiveSession()

input_data = tf.placeholder(tf.int32, [None, None], name='input')
targets = tf.placeholder(tf.int32, [None, None], name='targets')
lr = tf.placeholder(tf.float32, name='learning_rate')
keep_prob = tf.placeholder(tf.float32, name='keep_prob')
sequence_length = tf.placeholder_with_default(20, None, name='sequence_length')

# Find the shape of the input data for sequence_loss
input_shape = tf.shape(input_data)

# Create the training and inference logits
train_logits, inference_logits = model.seq2seq_model(
    tf.reverse(input_data, [-1]), targets, keep_prob, batch_size, sequence_length, len(answers_vocab_to_int),
    len(questions_vocab_to_int), encoding_embedding_size, decoding_embedding_size, rnn_size, num_layers,
    questions_vocab_to_int)


tf.identity(inference_logits, 'logits')

with tf.name_scope("optimization"):
    # Loss function
    cost = tf.contrib.seq2seq.sequence_loss(
        train_logits,
        targets,
        tf.ones([input_shape[0], sequence_length]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)


train_valid_split = int(len(sorted_questions)*0.15)

# Split the questions and answers into training and validating data
train_questions = sorted_questions[train_valid_split:]
train_answers = sorted_answers[train_valid_split:]

valid_questions = sorted_questions[:train_valid_split]
valid_answers = sorted_answers[:train_valid_split]

print(len(train_questions))
print(len(valid_questions))

display_step = 100  # Check training loss after every 100 batches
stop_early = 0
stop = 5  # If the validation loss does decrease in 5 consecutive checks, stop training
validation_check = ((len(train_questions)) // batch_size // 2) - 1  # Modulus for checking validation loss
total_train_loss = 0  # Record the training loss for each display step
summary_valid_loss = []  # Record the validation loss for saving improvements in the model

checkpoint = "best_model.ckpt"

sess.run(tf.global_variables_initializer())

for epoch_i in range(1, epochs + 1):
    for batch_i, (questions_batch, answers_batch) in enumerate(
            model.batch_data(valid_questions, valid_answers, batch_size, questions_vocab_to_int, answers_vocab_to_int)):
        start_time = time.time()
        _, loss = sess.run(
            [train_op, cost],
            {input_data: questions_batch,
             targets: answers_batch,
             lr: learning_rate,
             sequence_length: answers_batch.shape[1],
             keep_prob: keep_probability})

        total_train_loss += loss
        end_time = time.time()
        batch_time = end_time - start_time

        if batch_i % display_step == 0:
            print('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'
                  .format(epoch_i,
                          epochs,
                          batch_i,
                          len(train_questions) // batch_size,
                          total_train_loss / display_step,
                          batch_time * display_step),flush=True)
            total_train_loss = 0

        if batch_i % validation_check == 0 and batch_i > 0:
            total_valid_loss = 0
            start_time = time.time()
            for batch_ii, (questions_batch, answers_batch) in enumerate(
                    model.batch_data(valid_questions, valid_answers, batch_size,questions_vocab_to_int,answers_vocab_to_int)):
                valid_loss = sess.run(
                    cost, {input_data: questions_batch,
                           targets: answers_batch,
                           lr: learning_rate,
                           sequence_length: answers_batch.shape[1],
                           keep_prob: 1})
                total_valid_loss += valid_loss
            end_time = time.time()
            batch_time = end_time - start_time
            avg_valid_loss = total_valid_loss / (len(valid_questions) / batch_size)
            print('Valid Loss: {:>6.3f}, Seconds: {:>5.2f}'.format(avg_valid_loss, batch_time),flush=True)

            # Reduce learning rate, but not below its minimum value
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate

            summary_valid_loss.append(avg_valid_loss)
            if avg_valid_loss <= min(summary_valid_loss):
                print('New Record!',flush=True)
                stop_early = 0
                saver = tf.train.Saver()
                saver.save(sess, checkpoint)

            else:
                print("No Improvement.",flush=True)
                stop_early += 1
                if stop_early == stop:
                    break

    if stop_early == stop:
        print("Stopping Training.",flush=True)
        break

