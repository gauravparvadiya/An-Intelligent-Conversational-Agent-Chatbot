import tensorflow as tf
import numpy as np


def process_encoding_input(target_data, vocab_to_int, batch_size):
    '''Remove the last word id from each batch and concat the <GO> to the begining of each batch'''
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], vocab_to_int['<GO>']), ending], 1)

    return dec_input


def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    '''Create the encoding layer'''
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
    enc_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)
    _, enc_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=enc_cell,
                                                   cell_bw=enc_cell,
                                                   sequence_length=sequence_length,
                                                   inputs=rnn_inputs,
                                                   dtype=tf.float32)
    return enc_state


def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, sequence_length, decoding_scope,
                         output_fn, keep_prob, batch_size):
    '''Decode the training data'''

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])

    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(attention_states,
                                             attention_option="bahdanau",
                                             num_units=dec_cell.output_size)

    train_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                     att_keys,
                                                                     att_vals,
                                                                     att_score_fn,
                                                                     att_construct_fn,
                                                                     name="attn_dec_train")

    train_pred, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                              train_decoder_fn,
                                                              dec_embed_input,
                                                              sequence_length,
                                                              scope=decoding_scope)
    train_pred_drop = tf.nn.dropout(train_pred, keep_prob)
    return output_fn(train_pred_drop)


def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id, end_of_sequence_id,
                         maximum_length, vocab_size, decoding_scope, output_fn, keep_prob, batch_size):
    '''Decode the prediction data'''

    attention_states = tf.zeros([batch_size, 1, dec_cell.output_size])
    att_keys, att_vals, att_score_fn, att_construct_fn = \
        tf.contrib.seq2seq.prepare_attention(attention_states,
                                             attention_option="bahdanau",
                                             num_units=dec_cell.output_size)

    infer_decoder_fn = tf.contrib.seq2seq.attention_decoder_fn_inference(output_fn,
                                                                         encoder_state[0],
                                                                         att_keys,
                                                                         att_vals,
                                                                         att_score_fn,
                                                                         att_construct_fn,
                                                                         dec_embeddings,
                                                                         start_of_sequence_id,
                                                                         end_of_sequence_id,
                                                                         maximum_length,
                                                                         vocab_size,
                                                                         name="attn_dec_inf")
    infer_logits, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(dec_cell,
                                                                infer_decoder_fn,
                                                                scope=decoding_scope)

    return infer_logits


def decoding_layer(dec_embed_input, dec_embeddings, encoder_state, vocab_size, sequence_length, rnn_size,
                   num_layers, vocab_to_int, keep_prob, batch_size):
    '''Create the decoding cell and input the parameters for the training and inference decoding layers'''

    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob=keep_prob)
        dec_cell = tf.contrib.rnn.MultiRNNCell([drop] * num_layers)

        weights = tf.truncated_normal_initializer(stddev=0.1)
        biases = tf.zeros_initializer()
        output_fn = lambda x: tf.contrib.layers.fully_connected(x,
                                                                vocab_size,
                                                                None,
                                                                scope=decoding_scope,
                                                                weights_initializer=weights,
                                                                biases_initializer=biases)

        train_logits = decoding_layer_train(encoder_state,
                                            dec_cell,
                                            dec_embed_input,
                                            sequence_length,
                                            decoding_scope,
                                            output_fn,
                                            keep_prob,
                                            batch_size)
        decoding_scope.reuse_variables()
        infer_logits = decoding_layer_infer(encoder_state,
                                            dec_cell,
                                            dec_embeddings,
                                            vocab_to_int['<GO>'],
                                            vocab_to_int['<EOS>'],
                                            sequence_length - 1,
                                            vocab_size,
                                            decoding_scope,
                                            output_fn, keep_prob,
                                            batch_size)

    return train_logits, infer_logits


def seq2seq_model(input_data, target_data, keep_prob, batch_size, sequence_length, answers_vocab_size,
                  questions_vocab_size, enc_embedding_size, dec_embedding_size, rnn_size, num_layers,
                  questions_vocab_to_int):
    '''Use the previous functions to create the training and inference logits'''

    enc_embed_input = tf.contrib.layers.embed_sequence(input_data,
                                                       answers_vocab_size + 1,
                                                       enc_embedding_size,
                                                       initializer=tf.random_uniform_initializer(0, 1))
    enc_state = encoding_layer(enc_embed_input, rnn_size, num_layers, keep_prob, sequence_length)

    dec_input = process_encoding_input(target_data, questions_vocab_to_int, batch_size)
    dec_embeddings = tf.Variable(tf.random_uniform([questions_vocab_size + 1, dec_embedding_size], 0, 1))
    dec_embed_input = tf.nn.embedding_lookup(dec_embeddings, dec_input)

    train_logits, infer_logits = decoding_layer(dec_embed_input,
                                                dec_embeddings,
                                                enc_state,
                                                questions_vocab_size,
                                                sequence_length,
                                                rnn_size,
                                                num_layers,
                                                questions_vocab_to_int,
                                                keep_prob,
                                                batch_size)
    return train_logits, infer_logits


def pad_sentence_batch(sentence_batch, vocab_to_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [vocab_to_int['<PAD>']] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def batch_data(questions, answers, batch_size, questions_vocab_to_int, answers_vocab_to_int):
    for batch_i in range(0, len(questions) // batch_size):
        start_i = batch_i * batch_size
        questions_batch = questions[start_i:start_i + batch_size]
        answers_batch = answers[start_i:start_i + batch_size]
        pad_questions_batch = np.array(pad_sentence_batch(questions_batch, questions_vocab_to_int))
        pad_answers_batch = np.array(pad_sentence_batch(answers_batch, answers_vocab_to_int))
        yield pad_questions_batch, pad_answers_batch
