import sys
import random
import os

# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from keras.utils.np_utils import to_categorical
#from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import Callback, LambdaCallback, ModelCheckpoint
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt

# To run on GPU
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.compat.v1.Session(config = config)
set_session(sess)  # set this TensorFlow session as the default session for Keras

import ontology_preprocessing
from parse_xml import sst_light_directory
import models
from models import n_classes, max_ancestors_length, max_sentence_length
import results

# Input Parameters
dict_type_entity_ontology = {'DRUG': 'ontology_preprocessing.load_chebi()',
                             'PHENOTYPE': 'ontology_preprocessing.load_hpo()',
                             'GENE': 'ontology_preprocessing.load_go()',
                             'DISEASE': 'ontology_preprocessing.load_doid()'}
models_directory = 'models'
data_directory = 'data'
temporary_directory = 'temp/'
results_directory = 'results/'
n_epochs = 100
batch_size = 34
validation_split = 0.2


# --------------------------------------------------------------
#                   PREPROCESSING XML DATA
# --------------------------------------------------------------

def get_xml_data(dirs, name_to_id, synonym_to_id):
    """Generate data instances for the sentences of the corpus

    :param dirs: list of directories to be scanned for XML files
    :param name_to_id:
    :param synonym_to_id:
    :return: column vectors where each element corresponds to a label or a sequence of values of a particular
    data instance.
    """

    # Import function to process a directory with XML files
    from parse_xml import get_sdp_instances

    # Import function to process a directory with Biocreative V CDR files
    # from parse_biocreative_v_cdr import get_sdp_instances

    labels = []
    left_instances = []  # word indexes
    right_instances = []
    shared_ancestors = []  # ontology IDs (shared = common)
    left_ancestors = []
    right_ancestors = []
    left_wordnet = []  # wordnet IDs
    right_wordnet = []
    all_pos_gv = set()  # anti positive governors
    all_neg_gv = set()
    classes = np.empty((0,))

    for directory in dirs:
        if not os.path.isdir(directory):
            print('{} does not exist!'.format(directory))
            sys.exit()

        dir_labels, dir_instances, dir_classes, dir_common, dir_ancestors, dir_wordnet, \
        neg_gv, pos_gv = get_sdp_instances(directory, name_to_id, synonym_to_id, parser = 'spacy')

        dir_classes = np.array(dir_classes)
        labels += dir_labels
        left_instances += dir_instances[0]
        right_instances += dir_instances[1]
        shared_ancestors += dir_common
        left_ancestors += dir_ancestors[0]
        right_ancestors += dir_ancestors[1]
        left_wordnet += dir_wordnet[0]
        right_wordnet += dir_wordnet[1]
        classes = np.concatenate((classes, dir_classes), axis = 0)

        all_pos_gv.update(pos_gv)
        all_neg_gv.update(neg_gv)

    return labels, (left_instances, right_instances), classes, shared_ancestors, (left_ancestors, right_ancestors), (left_wordnet, right_wordnet)


# --------------------------------------------------------------
#                    PREPROCESS ENTITIES IDS
# --------------------------------------------------------------

def preprocess_ids(x_data, id_to_index, max_len):
    """Process a sequence of ontology:IDs, so an embedding index is not necessary

    :param x_data:
    :param id_to_index:
    :param max_len:
    :return: matrix to be used as training data
    """

    data = []
    for i, seq in enumerate(x_data):

        idxs = []

        for d in seq:
            if d and d.startswith('CHEBI') or d.startswith('HP') or d.startswith('GO') or d.startswith('DOID'):

                if d.replace('_', ':') not in id_to_index:  # 18 / 20 cases not in the ontology (probably the annotations are not updated)
                    pass

                else:
                    idxs = [id_to_index[d.replace('_', ':')]]

        data.append(idxs)

    data = pad_sequences(data, maxlen = max_len)

    return data


# --------------------------------------------------------------
#                          WORD VECTORS
# --------------------------------------------------------------

def get_w2v(file_name = '{}/PubMed-w2v.bin'.format(data_directory)):
    """Open Word2Vec file using Gensim package

    :param file_name:
    :return: word vectors in KeyedVectors Gensim object
    """

    word_vectors = KeyedVectors.load_word2vec_format(file_name, binary = True)  # C text format

    return word_vectors

def preprocess_sequences(x_data, embeddings_index):
    """Replace words in x_data with index of word in embeddings_index and pad sequence

    :param x_data: list of sequences of words (sentences)
    :param embeddings_index: word -> index in embedding matrix
    :return: matrix to be used as training data
    """

    data = []

    for i, seq in enumerate(x_data):

        idxs = []

        for w in seq:
            if w.lower() in embeddings_index.vocab:
                idxs.append(embeddings_index.vocab[w.lower()].index)

        if None in idxs:
            print(seq, idxs)

        data.append(idxs)

    data = pad_sequences(data, maxlen = max_sentence_length, padding = 'post')

    return data


# --------------------------------------------------------------
#                        WORDNET CLASSES
# --------------------------------------------------------------

def get_wordnet_indexes():
    """Get the WordNet classes considered by SST, ignoring BI tags

    :return: embedding_indexes: tag -> index
    """

    embedding_indexes = {}

    with open('{}/DATA/WNSS_07.TAGSET'.format(sst_light_directory), 'r') as f:
        lines = f.readlines()

        i = 0

        for l in lines:
            if l.startswith('I-'):
                continue

            embedding_indexes[l.strip().split('-')[-1]] = i
            i += 1

    return embedding_indexes

def preprocess_sequences_glove(x_data, embeddings_index):
    """Replace words in x_data with index of word in embeddings_index and pad sequence

    :param x_data: list of sequences of words (sentences)
    :param embeddings_index: word -> index in embedding matrix
    :return: matrix to be used as training data
    """

    data = []

    for i, seq in enumerate(x_data):

        idxs = [embeddings_index.get(w) for w in seq if w in embeddings_index]

        if None in idxs:
            print(seq, idxs)

        data.append(idxs)

    data = pad_sequences(data, maxlen = max_sentence_length)

    return data

# --------------------------------------------------------------
#                  CONTENATION OF ANCESTORS
# --------------------------------------------------------------

def concatenation_ancestors(pair_type, x_subpaths_train, list_order):
    """

    :param pair_type:
    :param x_subpaths_train:
    :param list_order:
    :return:
    """

    entity_left = pair_type.split('-')[0]
    entity_right = pair_type.split('-')[1]

    if entity_left == entity_right:

        is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = eval(dict_type_entity_ontology[entity_left])
        id_to_index_entity_left = id_to_index_entity_right = id_to_index

    else:

        is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index_entity_left = eval(dict_type_entity_ontology[entity_left])
        is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index_entity_right = eval(dict_type_entity_ontology[entity_right])
        id_to_index = {**id_to_index_entity_left, **id_to_index_entity_right}

    x_ids_left = preprocess_ids(x_subpaths_train[0], id_to_index_entity_left, max_ancestors_length)
    x_ids_right = preprocess_ids(x_subpaths_train[1], id_to_index_entity_right, max_ancestors_length)

    return  x_ids_left[list_order],  x_ids_right[list_order], id_to_index


# --------------------------------------------------------------
#          COMMON ANCESTORS (IF SAME TYPE OF ANCESTORS)
# --------------------------------------------------------------

def common_ancestors(pair_type, x_ancestors_train, list_order):
    """

    :param pair_type:
    :param x_ancestors_train:
    :param list_order:
    :return:
    """

    entity_left = pair_type.split('-')[0]

    is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index = eval(dict_type_entity_ontology[entity_left])
    x_ancestors = preprocess_ids(x_ancestors_train, id_to_index, max_ancestors_length * 2)

    return x_ancestors[list_order], id_to_index


# --------------------------------------------------------------
#                        TRAINING METRICS
# --------------------------------------------------------------

def write_plots(history, model_name, pair_type):
    """Write plots regarding model training

    :param history: history object returned by fit function
    :param model_name: name of model to be used as part of filename
    :param pair_type:
    """

    plt.figure()
    plt.plot(history.history['f1'])
    plt.plot(history.history['val_f1'])
    plt.title('model eval')
    plt.ylabel('score')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.savefig(results_directory + pair_type + '/' + '{}_acc.png'.format(model_name))

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.savefig(results_directory + pair_type + '/' + '{}_loss.png'.format(model_name))


class Metrics(Callback):
    """
    Implementation of P, R and F1 metrics for fit function callback
    """

    def __init__(self, labels, words, n_inputs, **kwargs):
        self.labels = labels
        self.words_left = words[0]
        self.words_right = words[1]
        self.n_inputs = n_inputs
        self._val_f1 = 0
        self._val_recall = 0
        self._val_precision = 0

        super(Metrics, self).__init__()

    def on_train_begin(self, logs = {}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs = {}):

        val_predict = (np.asarray(self.model.predict([self.validation_data[i] for i in range(self.n_inputs)],))).round()
        val_targ = self.validation_data[self.n_inputs]

        self._val_f1 = f1_score(val_targ[..., 1:], val_predict[..., 1:], average = 'macro')
        self._val_recall = recall_score(val_targ[..., 1:], val_predict[..., 1:], average = 'macro')
        self._val_precision = precision_score(val_targ[..., 1:], val_predict[..., 1:], average = 'macro')
        _confusion_matrix = confusion_matrix(val_targ.argmax(axis = 1), val_predict.argmax(axis = 1))

        self.val_f1s.append(self._val_f1)
        self.val_f1s.append(self._val_f1)
        self.val_recalls.append(self._val_recall)
        self.val_precisions.append(self._val_precision)

        s = 'Predicted not false: {}/{}\n{}\n'.format(len([x for x in val_predict if np.argmax(x) != 0]), len([x for x in val_targ if x[0] < 0.5]),_confusion_matrix)

        print('\n{} VAL_f1:{:6.3f} VAL_p:{:6.3f} VAL_r{:6.3f}\n'.format(s, self._val_f1, self._val_precision, self._val_recall),)


# --------------------------------------------------------------
#                        JOINED CHANNELS
# --------------------------------------------------------------

def join_channels(model_name, pair_type, channels, y_train, train_labels, n_classes, x_words_train, x_wordnet_train, x_subpaths_train, x_ancestors_train, id_to_index):
    """

    :param model_name:
    :param pair_type:
    :param channels:
    :param y_train:
    :param train_labels:
    :param n_classes:
    :param x_words_train:
    :param x_wordnet_train:
    :param x_subpaths_train:
    :param x_ancestors_train:
    :param id_to_index:
    :return:
    """

    # Remove and replace previous model files
    if os.path.isfile('{}/{}.json'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name)):
        os.remove('{}/{}.json'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name))

    if os.path.isfile('{}/{}.h5'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name)):
        os.remove('{}/{}.h5'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name))

    y_train = to_categorical(y_train, num_classes = n_classes)

    list_order = np.arange(len(y_train))
    print()
    print(list_order)

    random.seed(1)
    random.shuffle(list_order)
    y_train = y_train[list_order]

    train_labels = train_labels[list_order]
    print('\nTraining order:', list_order)

    inputs = {}
    n_inputs = 0

    if 'words' in channels:  ##### 1ST CHANNEL
        word_vectors = get_w2v()
        w2v_layer = word_vectors.get_keras_embedding(train_embeddings = False)

        x_words_left = preprocess_sequences(x_words_train[0], word_vectors)
        x_words_right = preprocess_sequences(x_words_train[1], word_vectors)

        del word_vectors

        inputs['left_words'] = x_words_left[list_order]
        inputs['right_words'] = x_words_right[list_order]

        n_inputs += 2

    else:
        w2v_layer = None

    if 'wordnet' in channels:  ##### 2ND CHANNEL
        wn_index = get_wordnet_indexes()

        x_wn_left = preprocess_sequences_glove(x_wordnet_train[0], wn_index)
        x_wn_right = preprocess_sequences_glove(x_wordnet_train[1], wn_index)

        inputs['left_wordnet'] = x_wn_left[list_order]
        inputs['right_wordnet'] = x_wn_right[list_order]

        n_inputs += 2

    else:
        wn_index = None

    if 'concatenation_ancestors' in channels:  ##### 3RD CHANNEL

        x_left, x_right, id_to_index = concatenation_ancestors(pair_type, x_subpaths_train, list_order)
        inputs['left_ancestors'] = x_left
        inputs['right_ancestors'] = x_right

        n_inputs += 2

    if 'common_ancestors' in channels:  ##### 4RT CHANNEL

        if  pair_type.split('-')[0] == pair_type.split('-')[1]:  # only for same type entities, for instance, DRUG-DRUG
            x_common, id_to_index = common_ancestors(pair_type, x_ancestors_train, list_order)
            inputs['common_ancestors'] = x_common

            n_inputs += 1

        else:
            print('The type of the entities participating in the pair is different, it is not possible to use the common_ancestors channel.')

    if 'concatenation_ancestors' or 'common_ancestors' in channels:
        pass

    else:
        id_to_index = None

    # Model
    model = models.get_model(w2v_layer, channels, wn_index, id_to_index)

    del id_to_index
    del wn_index

    # Serialize model to JSON
    model_json = model.to_json()

    with open('{}/{}.json'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name), 'w') as json_file:
        json_file.write(model_json)

    metrics = Metrics(train_labels, x_words_train, n_inputs)

    checkpoint = ModelCheckpoint(filepath='{}/{}.h5'.format(models_directory + '/' + pair_type.replace('-', '_').lower(), model_name), verbose=1, save_best_only=True)
    #class_weight = {0: 5.,
                    #1: 1.}
    history = model.fit(inputs, {'output': y_train}, validation_split=validation_split, epochs=n_epochs, batch_size=batch_size,
                        verbose=2, callbacks=[metrics, checkpoint]) #, class_weight=class_weight)

    write_plots(history, model_name, pair_type.replace('-', '_').lower())

    print('Saved model to disk.')


# --------------------------------------------------------------
#                        TEST / PREDICT
# --------------------------------------------------------------

def predict(model_name, corpus_name, gold_standard, channels, test_labels, x_words_test, x_wn_test, x_subpaths_test, x_ancestors_test, id_to_index):
    """

    :param model_name:
    :param corpus_name:
    :param gold_standard:
    :param channels:
    :param test_labels:
    :param x_words_test:
    :param x_wn_test:
    :param x_subpaths_test:
    :param x_ancestors_test:
    :param id_to_index:
    :return:
    """

    inputs = {}

    if 'words' in channels:  ##### 1ST CHANNEL
        word_vectors = get_w2v()

        x_words_test_left = preprocess_sequences([['entity'] + x[1:] for x in x_words_test[0]], word_vectors)
        x_words_test_right = preprocess_sequences([x[:-1] + ['entity'] for x in x_words_test[1]], word_vectors)

        del word_vectors

        inputs['left_words'] = x_words_test_left
        inputs['right_words'] = x_words_test_right

    if 'wordnet' in channels:  ##### 2ND CHANNEL
        wn_index = get_wordnet_indexes()

        x_wordnet_test_left = preprocess_sequences_glove(x_wn_test[0], wn_index)
        x_wordnet_test_right = preprocess_sequences_glove(x_wn_test[1], wn_index)

        del wn_index

        inputs['left_wordnet'] = x_wordnet_test_left
        inputs['right_wordnet'] = x_wordnet_test_right

    if 'concatenation_ancestors' in channels:  ##### 3RD CHANNEL

        x_ids_left = preprocess_ids(x_subpaths_test[0], id_to_index, max_ancestors_length)
        x_ids_right = preprocess_ids(x_subpaths_test[1], id_to_index, max_ancestors_length)

        inputs['left_ancestors'] = x_ids_left
        inputs['right_ancestors'] = x_ids_right

    if 'common_ancestors' in channels:  ##### 4RT CHANNEL

        x_ancestors = preprocess_ids(x_ancestors_test, id_to_index, max_ancestors_length * 2)

        inputs['common_ancestors'] = x_ancestors

    del id_to_index

    # Load JSON
    json_file = open('{}/{}.json'.format(models_directory + '/' + corpus_name.replace('-', '_'), model_name), 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model = tf.keras.models.model_from_json(loaded_model_json)

    # Load weights
    loaded_model.load_weights('{}/{}.h5'.format(models_directory + '/' + corpus_name.replace('-', '_'), model_name))

    print('Loaded model {}/{} from disk.'.format(models_directory + '/' + corpus_name.replace('-', '_'), model_name))

    scores = loaded_model.predict(inputs)

    # Write results to file
    from parse_xml import pair_type_to_label

    with open(results_directory + corpus_name.replace('-', '_') + '/{}_{}_results.txt'.format(model_name, corpus_name.replace('-', '_')), 'w') as f:
        f.write('\t'.join(['Entity_1', 'Entity_2', 'Predicted_class\n']))

        for i, pair in enumerate(test_labels):
            f.write('\t'.join((pair[0], pair[1], pair_type_to_label[(np.argmax(scores[i]))] + '\n')))


    # XML Results
    model_results = results.model_results_xml(results_directory + corpus_name.replace('-', '_') + '/{}_{}_results.txt'.format(model_name, corpus_name.replace('-', '_')), gold_standard)

    # Biocreative V CDR
    # model_results = results.model_results_cdr(results_directory + corpus_name.replace('-', '_') + '/{}_{}_results.txt'.format(model_name, corpus_name.replace('-', '_')),  gold_standard)

    model_precision = model_results[0]
    model_recall = model_results[1]
    model_f_measure = model_results[2]

    print('Precision:', model_precision)
    print('Recall:', model_recall)
    print('F-Measure:', model_f_measure)


# --------------------------------------------------------------
#                              RUN
# --------------------------------------------------------------


def main():
    """

    :return:
    """

    type_of_action = sys.argv[1]  # preprocess, train or test
    pair_type = sys.argv[2]  # DRUG-DRUG, GENE-DISEASE, GENE-DRUG, etc.


    ##### PREPROCESS #####
    if type_of_action == 'preprocess':
        preprocess_what = sys.argv[3]  # train, test, etc.
        corpora_directory = [sys.argv[4]]  # input directory

        name_to_id = {}
        synonym_to_id = {}
        if pair_type.split('-')[0] == 'DRUG' or pair_type.split('-')[-1] == 'DRUG':
            is_a_graph_chebi, name_to_id_chebi, synonym_to_id_chebi, id_to_name_chebi, id_to_index_chebi = ontology_preprocessing.load_chebi('{}/chebi.obo'.format(data_directory))
            name_to_id = {**name_to_id, **name_to_id_chebi}
            synonym_to_id = {**synonym_to_id, **synonym_to_id_chebi}

        if pair_type.split('-')[0] == 'PHENOTYPE' or pair_type.split('-')[-1] == 'PHENOTYPE':
            is_a_graph_hpo, name_to_id_hpo, synonym_to_id_hpo, id_to_name_hpo, id_to_index_hpo = ontology_preprocessing.load_hpo('{}/hp.obo'.format(data_directory))
            name_to_id = {**name_to_id, **name_to_id_hpo}
            synonym_to_id = {**synonym_to_id, **synonym_to_id_hpo}

        if pair_type.split('-')[0] == 'GENE' or pair_type.split('-')[-1] == 'GENE':
            is_a_graph_go, name_to_id_go, synonym_to_id_go, id_to_name_go, id_to_index_go = ontology_preprocessing.load_go('{}/go.obo'.format(data_directory))
            name_to_id = {**name_to_id, **name_to_id_go}
            synonym_to_id = {**synonym_to_id, **synonym_to_id_go}

        if pair_type.split('-')[0] == 'DISEASE' or pair_type.split('-')[-1] == 'DISEASE':
            is_a_graph_doid, name_to_id_doid, synonym_to_id_doid, id_to_name_doid, id_to_index_doid = ontology_preprocessing.load_doid('{}/doid.obo'.format(data_directory))
            name_to_id = {**name_to_id, **name_to_id_doid}
            synonym_to_id = {**synonym_to_id, **synonym_to_id_doid}

        train_labels, x_train, y_train, x_train_ancestors, x_train_subpaths, x_train_wordnet = get_xml_data(corpora_directory, name_to_id, synonym_to_id)

        del name_to_id
        del synonym_to_id

        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + '_y.npy', y_train)
        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + '_labels.npy', train_labels)
        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + '_x_words.npy', x_train)
        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + '_x_wordnet.npy', x_train_wordnet)
        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + '_x_subpaths.npy', x_train_subpaths)
        np.save(temporary_directory + pair_type.replace('-', '_').lower() + '/' + preprocess_what + "_x_ancestors.npy", x_train_ancestors)


    ##### TRAIN #####
    elif type_of_action == 'train':
        model_name = sys.argv[3]  # model_1, model_2 etc.
        channels = sys.argv[4:]  # channels to use or string with only one channel

        y_train = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_y.npy', allow_pickle=True)
        train_labels = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_labels.npy', allow_pickle=True)
        x_words_train = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_words.npy', allow_pickle=True)
        x_wordnet_train = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_wordnet.npy', allow_pickle=True)
        x_subpaths_train = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_subpaths.npy', allow_pickle=True)
        x_ancestors_train = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_ancestors.npy', allow_pickle=True)

        join_channels(model_name, pair_type, channels, y_train, train_labels, n_classes, x_words_train, x_wordnet_train, x_subpaths_train, x_ancestors_train, max_ancestors_length)


    ##### TEST / PREDICT #####
    elif type_of_action == 'test':
        model_name = sys.argv[3]  # model_1, model_2 etc.
        gold_standard = sys.argv[4]  # path to gold standard
        channels = sys.argv[5:]  # channels to use or string with only one channel

        id_to_index = {}
        if pair_type.split('-')[0] == 'DRUG' or pair_type.split('-')[-1] == 'DRUG':
            is_a_graph_chebi, name_to_id_chebi, synonym_to_id_chebi, id_to_name_chebi, id_to_index_chebi = ontology_preprocessing.load_chebi('{}/chebi.obo'.format(data_directory))
            id_to_index = {**id_to_index, **id_to_index_chebi}

        if pair_type.split('-')[0] == 'PHENOTYPE' or pair_type.split('-')[-1] == 'PHENOTYPE':
            is_a_graph_hpo, name_to_id_hpo, synonym_to_id_hpo, id_to_name_hpo, id_to_index_hpo = ontology_preprocessing.load_hpo('{}/hp.obo'.format(data_directory))
            id_to_index = {**id_to_index, **id_to_index_hpo}

        if pair_type.split('-')[0] == 'GENE' or pair_type.split('-')[-1] == 'GENE':
            is_a_graph_go, name_to_id_go, synonym_to_id_go, id_to_name_go, id_to_index_go = ontology_preprocessing.load_go('{}/go.obo'.format(data_directory))
            id_to_index = {**id_to_index, **id_to_index_go}

        if pair_type.split('-')[0] == 'DISEASE' or pair_type.split('-')[-1] == 'DISEASE':
            is_a_graph_doid, name_to_id_doid, synonym_to_id_doid, id_to_name_doid, id_to_index_doid = ontology_preprocessing.load_doid('{}/doid.obo'.format(data_directory))
            id_to_index = {**id_to_index, **id_to_index_doid}

        test_labels = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_labels.npy', allow_pickle=True)
        x_words_test = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_words.npy', allow_pickle=True)
        x_wordnet_test = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_wordnet.npy', allow_pickle=True)
        x_subpaths_test = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_subpaths.npy', allow_pickle=True)
        x_ancestors_test = np.load(temporary_directory + pair_type.replace('-', '_').lower() + '/' + type_of_action + '_x_ancestors.npy', allow_pickle=True)

        predict(model_name, pair_type.lower(), gold_standard, channels, test_labels, x_words_test, x_wordnet_test, x_subpaths_test, x_ancestors_test, id_to_index)

        del id_to_index

    else:
        print('The type of action was not properly defined, it has to be preprocess, train or test.')


if __name__ == '__main__':
    main()
