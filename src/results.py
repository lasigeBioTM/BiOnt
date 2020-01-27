import os

import xml.etree.ElementTree as ET

from parse_biocreative_v_cdr import get_new_offsets_sentences


# --------------------------------------------------------------
#                      MODEL XML STATISTICS
# --------------------------------------------------------------

def model_results_xml(test_results_file, directory_test):
    """Final micro metrics for the model test

    :param test_results_file: file with test results
    :param directory_test
    :return: floats that correspond to the metrics precision, recall and F-measure
    """

    results = open(test_results_file, 'r')
    results.readline()  # skip header
    results_lines = results.readlines()

    dict_results = {}

    for result in results_lines:
        line_elements = result.split('\t')

        if line_elements[-1][:-1] == 'no_relation':
            dict_results[(line_elements[0], line_elements[1])] = 'false'

        elif line_elements[-1][:-1] != 'no_relation':
            dict_results[(line_elements[0], line_elements[1])] = 'true'

    dict_test = dict_results.copy()

    for f in os.listdir(directory_test):

        tree = ET.parse(directory_test + '/' + f)
        root = tree.getroot()

        for sentence in root:
            all_pairs = sentence.findall('pair')

            # Important: in the XML files the type of relation must be explicitly indicated here
            if len(all_pairs) > 0:  # skip sentences without pairs

                for pair in all_pairs:

                    if pair.get('ddi'):
                        dict_test[(pair.get('e1'), pair.get('e2'))] = pair.get('ddi')

                    elif pair.get('relation'):
                        dict_test[(pair.get('e1'), pair.get('e2'))] = pair.get('relation')

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    for result_pair, relation in dict_results.items():

        if dict_test[result_pair] == 'true' and relation != 'false':
            true_positive += 1

        elif dict_test[result_pair] == 'false' and relation != 'false':
            false_positive += 1

        elif dict_test[result_pair] == 'true' and relation == 'false':
            false_negative += 1

        else:
            true_negative += 1

    print('\nTrue Positive:', true_positive)
    print('False Negative:', false_negative)
    print('False Positive:', false_positive)
    print('True Negative:', true_negative, '\n')

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (false_negative + true_positive)
    f_measure = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_measure


# --------------------------------------------------------------
#               MODEL BIOCREATIVE V CDR STATISTICS
# --------------------------------------------------------------

def model_results_cdr(test_results_file, directory_test):
    """Final micro metrics for the model test

    :param test_results_file: file with test results
    :param directory_test
    :return: floats that correspond to the metrics precision, recall and F-measure
    """

    results = open(test_results_file, 'r')
    results.readline()  # skip header
    results_lines = results.readlines()

    dict_results = {}

    for result in results_lines:
        line_elements = result.split('\t')
        dict_results[(line_elements[0], line_elements[1])] = line_elements[-1][:-1]

    dict_test = dict_results.copy()

    for input_file in os.listdir(directory_test):

        input = open(directory_test + '/' + input_file, 'r')
        input_blocks = input.read().split('\n\n')
        input.close()


        for block in input_blocks:

            if len(block.split('\n')) > 3:
                entities_per_sentence = get_new_offsets_sentences(block)

                for sentence, entities_sentence in entities_per_sentence.items():

                    all_entities = {}
                    for entity in entities_sentence:
                        all_entities[sentence[0] + '.e' + str(entity[0])] = entity[1:]

                    all_pairs = {}

                    for line in block.split('\n')[2:]:
                        if 'CID' in line:
                            if all_pairs.get(line.split('\t')[2]):
                                all_pairs[line.split('\t')[2]].append(line.split('\t')[3])

                            else:
                                all_pairs[line.split('\t')[2]] = []
                                all_pairs[line.split('\t')[2]].append(line.split('\t')[3])

                    for result, true_relation in dict_test.items():
                        if all_entities.get(result[0]):
                            if (all_entities[result[0]][-1] in all_pairs and all_entities[result[1]][-1] in all_pairs[all_entities[result[0]][-1]]) or \
                               (all_entities[result[1]][-1] in all_pairs and all_entities[result[0]][-1] in all_pairs[all_entities[result[1]][-1]]):

                                dict_test[result] = 'effect'

                            elif true_relation == 'effect' and ((all_entities[result[0]][-1] in all_pairs and all_entities[result[1]][-1] not in all_pairs[all_entities[result[0]][-1]]) or
                                                                (all_entities[result[1]][-1] in all_pairs and all_entities[result[0]][-1] not in all_pairs[all_entities[result[1]][-1]]) or
                                                                all_entities[result[0]][-1] not in all_pairs or all_entities[result[1]][-1] not in all_pairs):

                                dict_test[result] = 'no_relation'

    true_positive = 0
    false_positive = 0
    false_negative = 0
    true_negative = 0

    # for result_pair, relation in dict_results.items():
    #     print(result_pair, relation, dict_test[result_pair])

    for result_pair, relation in dict_results.items():

        if dict_test[result_pair] == 'effect' and relation == 'effect':
            true_positive += 1

        elif dict_test[result_pair] == 'no_relation' and relation == 'effect':
            false_positive += 1

        elif dict_test[result_pair] == 'effect' and relation == 'no_relation':
            false_negative += 1

        else:
            true_negative += 1

    print('\nTrue Positive:', true_positive)
    print('False Negative:', false_negative)
    print('False Positive:', false_positive)
    print('True Negative:', true_negative, '\n')

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (false_negative + true_positive)
    f_measure = 2 * ((precision * recall) / (precision + recall))

    return precision, recall, f_measure

