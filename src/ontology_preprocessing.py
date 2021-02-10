import os
import pickle
import atexit
import random

from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import obonet  # transforms OBO serialized ontologies in networks, source: https://github.com/dpavot/obonet
import networkx  # helps in the above


# --------------------------------------------------------------
#                 LOAD CHEBI (ENTITY TYPE: DRUG)
# --------------------------------------------------------------

def load_chebi(path = 'http://purl.obolibrary.org/obo/chebi.obo'):
    """

    :param path:
    :return:
    """

    print('\nLoading the ChEBI Ontology from {}...'.format(path))

    graph = obonet.read_obo(path)
    graph.add_node('BE:00000', name = 'ROOT')  # biomedical entity general root concept
    graph.add_edge('CHEBI:24431', 'BE:00000', edgetype = 'is_a')  # chemical_entity

    graph = graph.to_directed()
    is_a_graph = networkx.MultiDiGraph([(u, v, d) for u, v, d in graph.edges(data = True) if d['edgetype'] == 'is_a'])

    id_to_name = {}
    name_to_id = {}
    for id_, data in graph.nodes(data = True):
        try:
            id_to_name[id_] = data['name']
            name_to_id[data['name']] = id_
        except KeyError:
            pass

    id_to_index = {e: i + 1 for i, e in enumerate(graph.nodes())}  # ids should start on 1 and not 0

    id_to_index[''] = 0
    synonym_to_id = {}

    print('Synonyms to IDs...')

    for n in graph.nodes(data = True):

        for syn in n[1].get('synonym', []):

            syn_name = syn.split('"')

            if len(syn_name) > 2:
                syn_name = syn.split('"')[1]
                synonym_to_id.setdefault(syn_name, []).append(n[0])

    print('Done:', len(name_to_id), 'IDs with', len(synonym_to_id), 'synonyms.')

    return is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index


# --------------------------------------------------------------
#                 LOAD HPO (ENTITY TYPE: PHENOTYPE)
# --------------------------------------------------------------

def load_hpo(path = 'http://purl.obolibrary.org/obo/hp.obo'):
    """

    :param path:
    :return:
    """

    print('\nLoading the Human Phenotype Ontology from {}...'.format(path))

    graph = obonet.read_obo(path)
    graph.add_node('BE:00000', name = 'ROOT')  # biomedical entity general root concept
    graph.add_edge('HP:0000118', 'BE:00000', edgetype = 'is_a')  # human_phenotypic_abnormality

    graph = graph.to_directed()

    is_a_graph = networkx.MultiDiGraph([(u, v, d) for u, v, d in graph.edges(data = True) if d['edgetype'] == 'is_a'])

    id_to_name = {}
    name_to_id = {}
    for id_, data in graph.nodes(data=True):
        try:
            id_to_name[id_] = data['name']
            name_to_id[data['name']] = id_
        except KeyError:
            pass

    id_to_index = {e: i + 1 for i, e in enumerate(graph.nodes())}  # ids should start on 1 and not 0

    id_to_index[''] = 0
    synonym_to_id = {}

    print('Synonyms to IDs...')

    for n in graph.nodes(data = True):

        for syn in n[1].get('synonym', []):

            syn_name = syn.split('"')

            if len(syn_name) > 2:
                syn_name = syn.split('"')[1]
                synonym_to_id.setdefault(syn_name, []).append(n[0])

    print('Done:', len(name_to_id), 'IDs with', len(synonym_to_id), 'synonyms.')

    return is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index


# --------------------------------------------------------------
#                  LOAD GO (ENTITY TYPE: GENE)
# --------------------------------------------------------------

def load_go(path = 'http://purl.obolibrary.org/obo/go.obo'):
    """

    :return:
    """

    print('\nLoading the Gene Ontology from {}...'.format(path))

    graph = obonet.read_obo(path)
    graph.add_node('BE:00000', name = 'ROOT')  # biomedical entity general root concept
    graph.add_edge('GO:0008150', 'BE:00000', edgetype = 'is_a')  # biological_process

    graph = graph.to_directed()

    is_a_graph = networkx.MultiDiGraph([(u, v, d) for u, v, d in graph.edges(data = True) if d['edgetype'] == 'is_a'])

    id_to_name = {id_: data['name'] for id_, data in graph.nodes(data = True) if 'namespace' in data if data['namespace'] == 'biological_process'}  # only biological_process
    name_to_id = {data['name']: id_ for id_, data in graph.nodes(data = True) if 'namespace' in data if data['namespace'] == 'biological_process'}  # only biological_process

    id_to_index = {e: i + 1 for i, e in enumerate(graph.nodes())}  # ids should start on 1 and not 0

    id_to_index[''] = 0
    synonym_to_id = {}

    print('Synonyms to IDs...')

    for n in graph.nodes(data = True):

        for syn in n[1].get('synonym', []):

            syn_name = syn.split('"')

            if len(syn_name) > 2:
                syn_name = syn.split('"')[1]
                synonym_to_id.setdefault(syn_name, []).append(n[0])

    print('Done:', len(name_to_id), 'IDs with', len(synonym_to_id), 'synonyms.')

    return is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index


# --------------------------------------------------------------
#                LOAD DOID (ENTITY TYPE: DISEASE)
# --------------------------------------------------------------

def load_doid(path = 'https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo'):
    """

    :param path:
    :return:
    """

    print('\nLoading the Human Disease Ontology from {}...'.format(path))

    graph = obonet.read_obo(path)
    graph.add_node('BE:00000', name = 'ROOT')  # biomedical entity general root concept
    graph.add_edge('DOID:4', 'BE:00000', edgetype = 'is_a')  # disease

    graph = graph.to_directed()

    is_a_graph = networkx.MultiDiGraph([(u, v, d) for u, v, d in graph.edges(data = True) if d['edgetype'] == 'is_a'])

    id_to_name = {}
    name_to_id = {}
    for id_, data in graph.nodes(data=True):
        try:
            id_to_name[id_] = data['name']
            name_to_id[data['name']] = id_
        except KeyError:
            pass

    id_to_index = {e: i + 1 for i, e in enumerate(graph.nodes())}  # ids should start on 1 and not 0

    id_to_index[''] = 0
    synonym_to_id = {}

    print('Synonyms to IDs...')

    for n in graph.nodes(data = True):

        for syn in n[1].get('synonym', []):

            syn_name = syn.split('"')

            if len(syn_name) > 2:
                syn_name = syn.split('"')[1]
                synonym_to_id.setdefault(syn_name, []).append(n[0])

    print('Done:', len(name_to_id), 'IDs with', len(synonym_to_id), 'synonyms.')

    return is_a_graph, name_to_id, synonym_to_id, id_to_name, id_to_index


# --------------------------------------------------------------
#                        MAP TO ONTOLOGY
# --------------------------------------------------------------

def map_to_ontology(temporary_directory, text, name_to_id, synonym_to_id):
    """Get best entity name for given text

    :param temporary_directory:
    :param text: input text
    :param name_to_id:
    :param synonym_to_id:
    :return:
    """

    used_syn = False
    name, identifier = random.choice(list(name_to_id.items()))
    type_entity = identifier[0]

    cache_file = None
    chebi_cache = {}
    hpo_cache = {}
    go_cache = {}
    doid_cache = {}

    if type_entity == 'C':
        cache_file = temporary_directory + 'chebi_cache.pickle'
        if os.path.isfile(cache_file):
            chebi_cache = pickle.load(open(cache_file, 'rb'))

        else:
            chebi_cache = {}

    elif type_entity == 'H':
        cache_file = temporary_directory + 'hpo_cache.pickle'
        if os.path.isfile(cache_file):
            hpo_cache = pickle.load(open(cache_file, 'rb'))

        else:
            hpo_cache = {}

    elif type_entity == 'G':
        cache_file = temporary_directory + 'go_cache.pickle'
        if os.path.isfile(cache_file):
            go_cache = pickle.load(open(cache_file, 'rb'))

        else:
            go_cache = {}

    elif type_entity == 'D':
        cache_file = temporary_directory + 'doid_cache.pickle'
        if os.path.isfile(cache_file):
            doid_cache = pickle.load(open(cache_file, 'rb'))

        else:
            doid_cache = {}

    if text in name_to_id or text in synonym_to_id:
        return text, used_syn

    elif text in chebi_cache and type_entity == 'C':
        return chebi_cache[text], used_syn

    elif text in hpo_cache and type_entity == 'H':
        return hpo_cache[text], used_syn

    elif text in go_cache and type_entity == 'G':
        return go_cache[text], used_syn

    elif text in doid_cache and type_entity == 'D':
        return doid_cache[text], used_syn

    entity = process.extractOne(text, name_to_id.keys(), scorer = fuzz.token_sort_ratio)

    if entity[1] < 70:
        entity_syn = process.extract(text, synonym_to_id.keys(), limit = 10, scorer = fuzz.token_sort_ratio)

        if entity_syn[0][1] > entity[1]:
            used_syn = True
            entity = entity_syn[0]

    if type_entity == 'C':
        chebi_cache[text] = entity[0]
        pickle.dump(chebi_cache, open(cache_file, 'wb'))

    elif type_entity == 'H':
        hpo_cache[text] = entity[0]
        pickle.dump(hpo_cache, open(cache_file, 'wb'))

    elif type_entity == 'G':
        go_cache[text] = entity[0]
        pickle.dump(go_cache, open(cache_file, 'wb'))

    elif type_entity == 'D':
        doid_cache[text] = entity[0]
        pickle.dump(doid_cache, open(cache_file, 'wb'))

    return entity[0], used_syn
