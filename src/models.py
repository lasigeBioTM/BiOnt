import keras
from keras.layers import Input
from keras.layers import Concatenate
from keras.layers import Bidirectional
from keras.layers import Dropout
from keras.layers import GlobalMaxPooling1D
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from keras.layers import LSTM
from keras.layers import Dense
from keras.models import Model
from keras.optimizers import RMSprop
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras import backend as K
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

# Input Parameters (Model Architecture)
n_classes = 2
max_ancestors_length = 20
max_sentence_length = 20
embedding_size = 200
dropout_1 = 0.5
wordnet_embedding_size = 47
LSTM_units = 100
ontology_embedding_size = 100
sigmoid_units = 10


# --------------------------------------------------------------
#                          WORD VECTORS
# --------------------------------------------------------------

def get_words_channel_conc(words_input, embedding_matrix):
    """

    :param words_input:
    :param embedding_matrix:
    :return:
    """

    concatenate = keras.layers.concatenate([words_input[0], words_input[1]], axis = -1)
    e_words = embedding_matrix
    e_words = e_words(concatenate)

    words_lstm = Bidirectional(LSTM(LSTM_units, input_shape = (max_sentence_length, embedding_size), return_sequences = True, name = 'words_lstm'))(e_words)

    words_lstm = Dropout(dropout_1)(words_lstm)
    words_pool = GlobalMaxPooling1D()(words_lstm)

    return words_pool


# --------------------------------------------------------------
#                  CONTENATION OF ANCESTORS
# --------------------------------------------------------------

def get_ontology_concat_channel(ontology_input, id_to_index):
    """

    :param ontology_input:
    :param id_to_index:
    :return:
    """

    e_ancestors_left = Embedding(len(id_to_index), ontology_embedding_size, input_length = max_ancestors_length, trainable = True)
    e_ancestors_left.build((None,))
    e_ancestors_left = e_ancestors_left(ontology_input[0])
    e_ancestors_left = Dropout(dropout_1)(e_ancestors_left)

    e_ancestors_right = Embedding(len(id_to_index), ontology_embedding_size, input_length = max_ancestors_length, trainable = True)
    e_ancestors_right.build((None,))
    e_ancestors_right = e_ancestors_right(ontology_input[1])
    e_ancestors_right = Dropout(dropout_1)(e_ancestors_right)

    attention_rnn_left = LSTM(LSTM_units, input_shape = (max_ancestors_length, ontology_embedding_size), return_sequences = True)(e_ancestors_left)
    attention_rnn_right = LSTM(LSTM_units, input_shape = (max_ancestors_length, ontology_embedding_size), return_sequences = True)(e_ancestors_right)

    ancestors_pool_left = GlobalMaxPooling1D()(attention_rnn_left)
    ancestors_pool_right = GlobalMaxPooling1D()(attention_rnn_right)

    return ancestors_pool_left, ancestors_pool_right


# --------------------------------------------------------------
#          COMMON ANCESTORS (IF SAME TYPE OF ANCESTORS)
# -------------------------------------------------------------

def get_ontology_common_channel(ontology_input, id_to_index):
    """

    :param ontology_input:
    :param id_to_index:
    :return:
    """

    e_ancestors = Embedding(len(id_to_index),  ontology_embedding_size, input_length = max_ancestors_length*2, trainable = True)
    e_ancestors.build((None,))
    e_ancestors = e_ancestors(ontology_input)
    e_ancestors = Dropout(0.5)(e_ancestors)

    ancestors_lstm = LSTM(LSTM_units, input_shape = (max_ancestors_length*2,  ontology_embedding_size), return_sequences = True)(e_ancestors)

    ancestors_pool = GlobalMaxPooling1D()(ancestors_lstm)

    return ancestors_pool


# --------------------------------------------------------------
#                             MODEL
# --------------------------------------------------------------

def get_model(embedding_matrix, channels, wordnet_emb, id_to_index):
    """

    :param embedding_matrix:
    :param channels:
    :param wordnet_emb:
    :param id_to_index:
    :return:
    """

    inputs = []
    pool_layers = []

    if 'words' in channels:
        words_input_left = Input(shape = (max_sentence_length,), name = 'left_words')
        words_input_right = Input(shape = (max_sentence_length,), name = 'right_words')

        inputs += [words_input_left, words_input_right]

        words_pool = get_words_channel_conc((words_input_left, words_input_right), embedding_matrix)

        pool_layers += [words_pool]

    if 'wordnet' in channels:
        wordnet_left = Input(shape = (max_sentence_length,), name = 'left_wordnet')
        wordnet_right = Input(shape = (max_sentence_length,), name = 'right_wordnet')

        inputs += [wordnet_left, wordnet_right]

        e_wn_left = Embedding(len(wordnet_emb), wordnet_embedding_size, input_length = max_sentence_length, trainable = True, name = 'wn_emb_left')
        e_wn_left.build((None,))
        e_wn_left = e_wn_left(wordnet_left)
        e_wn_left = SpatialDropout1D(0.5)(e_wn_left)

        e_wn_right = Embedding(len(wordnet_emb), wordnet_embedding_size, input_length = max_sentence_length, trainable = True, name = 'wn_emb_right')
        e_wn_right.build((None,))
        e_wn_right = e_wn_right(wordnet_right)
        e_wn_right = SpatialDropout1D(0.5)(e_wn_right)

        wn_lstm_left = LSTM(LSTM_units, input_shape = (max_sentence_length, wordnet_embedding_size), name = 'wn_lstm_left', return_sequences = True)(e_wn_left)
        wn_lstm_right = LSTM(LSTM_units, input_shape = (max_sentence_length, wordnet_embedding_size), name = 'wn_lstm_right', return_sequences = True)(e_wn_right)

        wn_pool_left = GlobalMaxPooling1D()(wn_lstm_left)
        wn_pool_right = GlobalMaxPooling1D()(wn_lstm_right)

        pool_layers += [wn_pool_left, wn_pool_right]

    if 'concatenation_ancestors' in channels:
        # Uses just one chain without LSTM (order is always the same)
        ancestors_input_left = Input(shape = (max_ancestors_length,), name = 'left_ancestors')
        ancestors_input_right = Input(shape = (max_ancestors_length,), name = 'right_ancestors')

        inputs += [ancestors_input_left, ancestors_input_right]

        ancestors_pool_left, ancestors_pool_right = get_ontology_concat_channel((ancestors_input_left, ancestors_input_right), id_to_index)
        pool_layers += [ancestors_pool_left, ancestors_pool_right]

    if 'common_ancestors' in channels:
        ancestors_common = Input(shape = (max_ancestors_length * 2,), name = 'common_ancestors')
        inputs += [ancestors_common]
        ancestors_pool = get_ontology_common_channel(ancestors_common, id_to_index)
        pool_layers.append(ancestors_pool)

    if len(pool_layers) > 1:
        concatenate = keras.layers.concatenate(pool_layers, axis = -1)
    else:
        concatenate = pool_layers[0]

    final_hidden = Dense(sigmoid_units, activation = 'sigmoid', name = 'hidden_layer')(concatenate)

    output = Dense(n_classes, activation = 'softmax', name = 'output')(final_hidden)

    model = Model(inputs = inputs, outputs = [output])

    model.compile(loss = 'categorical_crossentropy', # options: categorical | binary_crossentropy
                  optimizer = RMSprop(0.0001),  # options: Adam(0.001) | SGD(0.1)
                  # sample_weight_mode = None,  # optional
                  # weighted_metrics = [recall],  # optional
                  metrics = ['accuracy', precision, recall, f1])

    print(model.summary())

    return model


# --------------------------------------------------------------
#                            METRICS
# --------------------------------------------------------------

def precision(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred[...,1:], 0, 1)))
    p = true_positives / (predicted_positives + K.epsilon())

    return p

def recall(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    true_positives = K.sum(K.round(K.clip(y_true[...,1:] * y_pred[...,1:], 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true[...,1:], 0, 1)))
    r = true_positives / (possible_positives + K.epsilon())

    return r


def f1(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    precision_v = precision(y_true, y_pred)
    recall_v = recall(y_true, y_pred)

    return (2.0 * precision_v * recall_v) / (precision_v + recall_v + K.epsilon())
