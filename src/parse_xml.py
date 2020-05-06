import os
import sys
import logging

from itertools import product
import xml.etree.ElementTree as ET
from subprocess import PIPE, Popen
import en_core_web_sm
import networkx as nx
import string

import ontology_preprocessing

sys.path.append('bin/DiShIn/')  # access bin/DiShIn/
from ssmpy import ssm

# Input Parameters
sst_light_directory = 'bin/sst-light-0.4/'
temporary_directory = 'temp/'

NO_RELATION = 0
EFFECT = 1
MECHANISM = 2
ADVICE = 3
INT_TYPE = 4

label_to_pair_type = {'effect': EFFECT, 'mechanism': MECHANISM, 'advise': ADVICE, 'int': INT_TYPE}
pair_type_to_label = {v: k for k, v in label_to_pair_type.items()}
pair_type_to_label[NO_RELATION] = 'no_relation'
biomedical_entity_to_ontology = {'gene': 'G', 'phenotype': 'H', 'drug': 'C', 'disease': 'D'}

# To exclude (Needs Update)
neg_gv_list = {'cerubidine', 'trial', '5-fu', 'total', 'multivitamins', 'elemental', 'nitroprusside', 'chlortetracycline', 'transdermal', 'altered', 'promethazine', 'ml', 'fluoroquinolones', 'cephalothin_sodium', 'amiloride', 'tambocor', 'blocking_agents', 'immunosuppressives', 'weight', 'than', 'nabumetone', 'entacapone', 'fexofenadine', 'cytosine_arabinoside', 'drug', 'metaclopramide', 'divalproex_sodium', 'desloratadine', 'database', 'hydantoins', 'benazepril', 'amoxicillin', 'restricted', 'tendency', 'iron_supplements', 'azathioprine', 'exist', 'imidazole', 'half', 'anxiolytics', 'regimen', 'angiotensin-_converting_enzyme_(ace)_inhibitors', 'uroxatral', 'cefoperazone', 'other', 'wshow', 'andusing', '12', 'dobutamine', 'addiction', '500', 'potential', 'lead', 'eliminated', 'transferase', 'leflunomide', 'digitalis_preparations', 'stadol_ns', 'desbutyl_levobupivacaine', 'glibenclamide', 'vinblastine', 'aripiprazole', 'appear', 'oxidase', 'blunt', 'seriraline', 'bedtime', 'arimidex', 'dextromethorphan', 'lanoxin', 'cabergoline', 'oxacillin', 'naprosyn', 'users', 'iloprost', 'local', 'trifluoperazine', 'cefmenoxime', 'plaquenil', 'excess', 'chlorpromazine', 'misused', 'antibiotic', 'involving', 'stanozolol', 'antimycobacterial', 'zdv', 'antidiabetic_products', 'chlorothiazide', 'orlistat', 'bleomycin', 'latamoxef', 'somatostatin_analog', 'slows', 'alternatives', 'make', 'atenolol', 'corresponding', 'seen', 'l50', 'ribavirin', 'dynacirc', 'coumarin_derivatives', 'glyceryl_trinitrate', 'propofol', 'tacrine', 'mepron', 'excreted', 'examining', 'triflupromazine', 'iron_supplement', 'deltasone', 'amlodipine', 'nandrolone', 'antidiabetic_agents', 'antipsychotic_drugs', 'pa', 'containing_compounds', 'er', 'trimethoprim', 'glycoprotein', 'calcitriol', 'multiple', 'angiotensin_ii_receptor_antagonists', 'coa_reductase_inhibitor', 'nonsteroidal_antiinflammatories', 'infused', 'fluvastatin', 'reversible', 'mycophenolate', 'fell', 'vitamin_b3', 'maoi_antidepressants', '-treated', 'aeds', 'induction', 'hypoglycemic_agents', 'antifungal', 'salicylic_acid', 'gabapentin', 'fibrates', 'carvedilol', 'neuromuscular_blocking_agents', 'mesoridazine', 'require', 'fibrinogen', 'predispose', 'anakinra', 'somatostatin_analogs', 'magnesium_hydroxide_antacids', 'pregnancy', ';', 'therefore', 'antiarrhythmic_agents', 'surgery', 'conversion', 'monoamine_oxidase_inhibitor', 'serum', 'cardiac_glycosides', 'fosphenytoin', 'adrenergic_receptor_blockers', 'detected', 'grepafloxacin', 'systemic_corticosteroids', 'nucleoside_reverse_transcriptase_inhibitors', 'divalproex', 'thiocyanate', 'metrizamide', 'included', 'immunosuppressants', 'terbutaline', 'mycophenolate_mofetil', 'modify', 'blocker', 'valsartan', 'sulfoxone', 'distribution', 'famciclovir', 'minutes', 'chelating', 'immunosuppressive_drugs', 'accelerate', 'thrombolytic_agents', 'twice', 'promazine', 'bactrim', 'psychotropic_drugs', 'borne', 'novoseven', 'hivid', 'cromolyn_sodium', 'converting_enzyme_(ace)_inhibitors', 'cleared', 'transport', 'oruvail', 'experience', 'depletion', 'synkayvite', 'chlorthalidone', 'cyp1a2', 'produces', 'hypoglycemia', 'pegasys', 'diagnostic', 'mixing', 'oxc', 'hydroxyurea', 'and/or', 'requiring', 'mtx', 'lithium_carbonate', 'fibric_acid_derivatives', 'rifapentine', 'furafylline', 'dihydropyridine_calcium_antagonists', 'intensified', 'withdrawal', 'ameliorate', 'levonorgestrol', 'rofecoxib', 'ganglionic', 'anaprox', 'hiv_protease_inhibitors', 'studied', 'phenobarbitol', 'threohydrobupropion', 'antithyroid_drugs', 'alg', 'intoxication', 'anagrelide', 'assessed', 'nothiazines', 'terminated', 'coa_reductase_inhibitors', 'ticlopidine', 'cefazolin', 'cyp3a4', 'oxcarbazepine', 'hypokalemia', 'yielded', 'descarboethoxyloratadine', 'oxandrolone', 'leads', 'tranexamic_acid', 'dexmedetomidine', 'pancuronium', 'antacid', 'resorcinol', 'going', 'lenalidomide', 'influence', 'modified', 'pyrantel', 'droperidol', 'replacement', 'benzylpenicillin', 'acting_beta2-agonists', 'n=29', 'sequence', 'utilize', 'gram', 'interferences', 'nicotinic_acid', 'influenced', 'examples', 'min', 'salicylate', 'sulfur', 'keppra', 'iodoquinol', 'hours', 'trimeprazine', 'vitamin_d2', 'tolerated', 'procarbazine', 'volunteers', 'anions', 'increasing', 'etretinate', 'p450', 'nafcillin', 'cyp2c9', 'considered', 'prednisone', 'zofran', 'drawn', 'isradipine', 'lodosyn', 'substrates', 'orencia', 'debrisoquin', 'indicate', 'peginterferon', 'fortified', 'sulfisoxazole', 'tranylcypromine', 'antacid_products', 'antipsychotic_agents', 'antidiabetic_drugs', 'sucralfate', 'hemostasis', 'medrol', 'aminoglutethimide', 'clotrimazole', 'propanolol', 'monotherapy', 'irinotecan', 'identified', '/', 'somatrem', 'acetophenazine', 'gold', 'dirithromycin', 'sympathomimetics', 'erbitux', 'catalyzed', 'indanavir', 'ergonovine', 'lowered', 'infusion', 'combination', 'linezolid', 'substrate', 'differences', 'lowers', 'concomitant', 'nondepolarizing', 'meq', 'sparfloxacin', 'parameters', 'r', 'adjustments', 'prednisolone_sodium_succinate', 'nimodipine', 'tolerance', 'motrin', 'pill', 'sulfadoxine', 'mayuse', 'occurred', 'ci', 'flucoxacillin', 'metoclopramide', 'rifamipicin', 'responsive', 'cycles', 'trials', 'loop_diuretics', 'exhibits', 'folic_acid', 'ceftazidime', 'h2_antagonists', 'lansoprazole', 'escitalopram', 'methylprednisolone', 'antidepressant', 'accounts', 'vitamin_d3', 'gestodene', 'blocking_drugs', 'contribution', 'substances', 'tranylcypromine_sulfate', 'ritanovir', 'nizatidine', 'ingesting', 'buride', 'wthionamide', 'pravastatin', 'gleevec', 'index', 'tikosyn', 'cefotetan', 'antipsychotic_medications', 'aralen', 'performed', 'phenelzine', 'plicamycin', 'possibility', 'betablockers', 'isoenzymes', 'diphenylhydantoin', 'propatenone', 'eproxindine', 'alone', 'determined', 'evaluated', 'profiles', 'bioavailabillty', 'protamine', 'hyperreflexia', 'vitamin_a', 'vitamin_k_antagonists', 'medicine', 'cytokines', 'hydrocodone', 'vs.', 'methylthiotetrazole', 'tested', 'insert', 'antiacid', 'an', 'differ', 'invalidate', 'antiemetics', 'mellaril', 'dosed', 'range', 'bepridil', 'activated_prothrombin_complex_concentrates', 'inactivate', 'exercised', 'etomidate', 'vecuronium', 'coronary_vasodilators', 'dependent', 'anticholinesterases', 'prochlorperazine', 'r-', 'oxymetholone', 'aprepitant', 'ics', 'iressa', 'mephenytoin', 'ramipril', 'novum', 'medication', 'contains', 'diminished', 'activate', 'lam', 'sterilize', 'methandrostenolone', 'antipyrine', 'hydralazine', 'celecoxib', 'hydramine', 'exists', 'antipyretics', 'adenocard', 'besides', 'alpha-', 'cinacalcet', 'demonstrate', 'lomefloxacin', 'cephalothin', 'prolixin', 'concentrates', 'tests', 'analyses', 'proton_pump_inhibitors', 'mean', 'maintained', 'interferon', 'anticholinergic_agents', 'phenformin', 'failed', 'utilization', 'codeine', 'pediapred', 'isosorbide_dinitrate', 'oxaprozin', 'calcium_channel_antagonists', 'magnesium_sulfate', 'nonsteroidal_antiinflammatory_drugs', 'albuterol', 'prazosin', 'replacing', 'expanders', 'showed', 'hypercalcemia', 'benzothiadiazines', 'aza', 'humira', 'aminopyrine', 'cefamandole_naftate', '1/35', 'tolazoline', 'channel_blockers', 'thyroid_hormones', 'orudis', 'selegiline', 'analgesics', 'antagonists', 'ganglionic_blocking_agents', 'antagonism', 'pseudoephedrine', 'calcium_channel_blocking_drugs', 'oxide', 'chemotherapeutic_agents', 'cations', 'tend', 'undergo', 'includes', 'butazone', 'peak', 'sulfonamide', 'enzymes', '%', 'gabitril', 'acarbose', 'simvastatin', 'mixed', 'ethionamide', 'a', 'cyp2d6', 'ergot', 'metabolites', 'interrupted', 'carmustine', 'antianxiety_drugs', 'about', 'decarboxylase_inhibitor', 'hctz', 'advil', 'isosorbide_mononitrate', 'naltrexone', 'experienced', 'niacin', 'potassium_chloride', 'andtolbutamide', 'established', 'streptomycin', 'circulating', 'components', 'induces', 'dihydropyridine_derivative', 'caution', 'clonidine', 'piroxicam', 'phenylpropanolamine', 'label', 'indicated', 'pharmacokinetics', 'im', 'potassium_sparing_diuretics', 'adrenocorticoids', 'ocs', 'penicillin', 'conducted', 'desethylzaleplon', 'felbatol', 'nitrates', 'reviewed', 'smx', 'disease', 'cream', 'control', 'adefovir_dipivoxil', 'ethotoin', 'corticosteroid', 'voltaren', 'antivirals', 'protease_inhibitor', 'furazolidone', 'estrogen', 'investigated', 'mix', 'dapsone_hydroxylamine', 'cefamandole', 'mitotane', 'poisoning', 'metoprolol', 'dopa_decarboxylase_inhibitor', 'incombination', 'nisoldipine', 'diltiazem_hydrochloride', 'adjustment', 'tnf_blocking_agents', 'etodolac', 'phenelzine_sulfate', 'minus', 'formed', 'lower', 'show', 'cardiovasculars', 'sympathomimetic_bronchodilators', 'nitrofurantoin', 'calcium_channel_blocking_agents', 'oxymetazoline', 'neuroleptic', 'tetracyclic_antidepressants', 'steroid_medicine', 'arb', 'phenytoin_sodium', '5-dfur', 'bronchodilators', 'confirmed', 'among', 'sulphenazole', 'antiretroviral_nucleoside_analogues', 'binding', 'imatinib', 'cylates', 'plasmaconcentrations', 'acetohydroxamic_acid', 'inducing'}


# --------------------------------------------------------------
#                     GET SENTENCE ENTITIES
# --------------------------------------------------------------

def get_sentence_entities(base_dir, name_to_id, synonym_to_id, entity_id_max = 10):
    """

    :param base_dir:
    :param name_to_id:
    :param synonym_to_id:
    :param entity_id_max:
    :return:
    """

    entities = {}  # sentence_id:entities
    pair_entities = set()  # list of entities in relations

    for f in os.listdir(base_dir):

        tree = ET.parse(base_dir + '/' + f)
        root = tree.getroot()

        for sentence in root:
            sentence_id = sentence.get('id')
            sentence_text = sentence.get('text')
            sentence_entities = {}
            all_pairs = sentence.findall('pair')

            # Important: in the XML files the type of relation must be explicitly indicated here
            pos_pairs = {}
            for pair in all_pairs:

                if pair.get('ddi') == 'true':
                    # pos_pairs[(pair.get('e1'), pair.get('e2'))] = label_to_pair_type[pair.get('type')]
                    pos_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

                elif pair.get('relation') == 'true':
                    pos_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

                elif pair.get('pgr') == 'true':
                    pos_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

            for p in pos_pairs:
                pair_entities.add(p[0])
                pair_entities.add(p[1])

            if len(all_pairs) > 0:  # skip sentences without pairs

                for e in sentence.findall('entity'):
                    e_id = e.get('id')

                    if int(e_id.split('e')[-1]) > entity_id_max:
                        continue

                    sep_offsets = e.get('charOffset').split(';')
                    offsets = []

                    for o in sep_offsets:
                        start, end = o.split('-')
                        offsets.append(int(start))
                        offsets.append(int(end)+1)

                    e_text = e.get('text')
                    if offsets[0] == 0 and sentence_text[offsets[-1]] == ':':
                        logging.debug('Skipped title: {} -> {}'.format(e_text, sentence_text))

                    entity_name, used_syn = ontology_preprocessing.map_to_ontology(temporary_directory + base_dir.split('/')[1] + '/', e_text, name_to_id, synonym_to_id)
                    if entity_name in name_to_id:
                        entity_id = name_to_id[entity_name]

                    else:
                        entity_id = synonym_to_id[entity_name][0]

                    sentence_entities[e_id] = (offsets, e_text, entity_id)
                entities[sentence_id] = sentence_entities

    return entities, pair_entities


# --------------------------------------------------------------
#                     GET ENTITIES ANCESTORS
# --------------------------------------------------------------

def get_common_ancestors(id1, id2):
    """

    :param id1:
    :param id2:
    :return:
    """

    if id1.startswith('CHEBI'):
        ssm.semantic_base('bin/DiShIn/chebi.db')
    if id1.startswith('HP'):
        ssm.semantic_base('bin/DiShIn/hp.db')
    if id1.startswith('GO'):
        ssm.semantic_base('bin/DiShIn/go.db')
    if id1.startswith('DOID'):
        ssm.semantic_base('bin/DiShIn/doid.db')

    e1 = ssm.get_id(id1.replace(':', '_'))

    if id2.startswith('CHEBI'):
        ssm.semantic_base('bin/DiShIn/chebi.db')
    if id2.startswith('HP'):
        ssm.semantic_base('bin/DiShIn/hp.db')
    if id2.startswith('GO'):
        ssm.semantic_base('bin/DiShIn/go.db')
    if id2.startswith('DOID'):
        ssm.semantic_base('bin/DiShIn/doid.db')

    e2 = ssm.get_id(id2.replace(':', '_'))

    a = ssm.common_ancestors(e1, e2)
    # if a:
    #     print(id1, id2)
    #     print(e1, e2)
    #     print()
    #     print(a)
    #     b = [ssm.get_name(x) for x in a]
    #     print(b)
    #     print('\n\n\n')
    a = [ssm.get_name(x) for x in a]

    return a

def get_path_to_root(entity_id):
    """

    :param entity_id:
    :return:
    """

    if entity_id.startswith('CHEBI'):
        ssm.semantic_base('bin/DiShIn/chebi.db')
    if entity_id.startswith('HP'):
        ssm.semantic_base('bin/DiShIn/hp.db')
    if entity_id.startswith('GO'):
        ssm.semantic_base('bin/DiShIn/go.db')
    if entity_id.startswith('DOID'):
        ssm.semantic_base('bin/DiShIn/doid.db')

    e1 = ssm.get_id(entity_id.replace(':', '_'))

    a = ssm.common_ancestors(e1, e1)
    a = [ssm.get_name(x) for x in a]

    return a

def get_ancestors(sentence_labels, sentence_entities):
    """Obtain the path to lowest common ancestor of each entity of each pair and path from LCA to root

    :param sentence_labels: list of (e1, e2)
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :return: left and right paths to LCA
    """

    right_paths = []
    left_paths = []
    common_ancestors = []

    for p in sentence_labels:

        instance_ancestors = get_common_ancestors(sentence_entities[p[0]][2], sentence_entities[p[1]][2])
        left_path = get_path_to_root(sentence_entities[p[0]][2])
        right_path = get_path_to_root(sentence_entities[p[1]][2])

        # print('Common ancestors:', sentence_entities[p[0]][1:], sentence_entities[p[1]][1:], instance_ancestors)

        instance_ancestors = [i for i in instance_ancestors if i.startswith('CHEBI') or i.startswith('HP') or i.startswith('GO') or i.startswith('DOID')]
        left_path = [i for i in left_path if i.startswith('CHEBI') or i.startswith('HP') or i.startswith('GO') or i.startswith('DOID')]
        right_path = [i for i in right_path if i.startswith('CHEBI') or i.startswith('HP') or i.startswith('GO') or i.startswith('DOID')]

        common_ancestors.append(instance_ancestors)
        left_paths.append(left_path)
        right_paths.append(right_path)

    return common_ancestors, (left_paths, right_paths)


# --------------------------------------------------------------
#               PARSE CORPUS SENTENCES USING SPACY
# --------------------------------------------------------------

def prevent_sentence_segmentation(doc):
    """

    :param doc:
    :return:
    """

    for token in doc:

        # This will entirely disable spaCy's sentence detection
        token.is_sent_start = False

    return doc

nlp = en_core_web_sm.load(disable=['ner'])
nlp.add_pipe(prevent_sentence_segmentation, name='prevent-sbd', before='parser')

def parse_sentence_spacy(sentence_text, sentence_entities):
    """

    :param sentence_text:
    :param sentence_entities:
    :return:
    """

    # Use spacy to parse a sentence
    for e in sentence_entities:
        idx = sentence_entities[e][0]
        sentence_text = sentence_text[:idx[0]] + sentence_text[idx[0]:idx[1]].replace(' ', '_') + sentence_text[idx[1]:]

    # Clean text to make tokenization easier
    sentence_text = sentence_text.replace(';', ',')
    sentence_text = sentence_text.replace('*', ' ')
    sentence_text = sentence_text.replace(':', ',')
    sentence_text = sentence_text.replace(' - ', ' ; ')

    parsed = nlp(sentence_text)

    return parsed

def run_sst(base_dir, token_seq):
    """

    :param base_dir:
    :param token_seq:
    :return:
    """

    chunk_size = 500
    wordnet_tags = {}
    sent_ids = list(token_seq.keys())

    chunks = [sent_ids[i:i + chunk_size] for i in range(0, len(sent_ids), chunk_size)]

    for i, chunk in enumerate(chunks):
        sentence_file = open('{}/sentences_{}.txt'.format(temporary_directory + base_dir.split('/')[1], i), 'w')

        for sent in chunk:
            sentence_file.write("{}\t{}\t.\n".format(sent, '\t'.join(token_seq[sent])))

        sentence_file.close()
        sst_args = [sst_light_directory + 'sst', 'bitag',
                    '{}/MODELS/WSJPOSc_base_20'.format(sst_light_directory), '{}/DATA/WSJPOSc.TAGSET'.format(sst_light_directory),
                    '{}/MODELS/SEM07_base_12'.format(sst_light_directory), '{}/DATA/WNSS_07.TAGSET'.format(sst_light_directory),
                    '{}/sentences_{}.txt'.format(temporary_directory + base_dir.split('/')[1], i), '0', '0']

        p = Popen(sst_args, stdout = PIPE)
        p.communicate()

        with open('{}/sentences_{}.txt.tags'.format(temporary_directory + base_dir.split('/')[1], i)) as f:
            output = f.read()

        sstoutput = parse_sst_results(output)
        wordnet_tags.update(sstoutput)

    return wordnet_tags

def parse_sst_results(results):
    """

    :param results:
    :return:
    """

    sentences = {}
    lines = results.strip().split('\n')

    for l in lines:
        values = l.split('\t')
        wntags = [x.split(' ')[-1].split('-')[-1] for x in values[1:]]
        sentences[values[0]] = wntags

    return sentences

def parse_xml_sentences_spacy(base_dir, entities):
    """

    :param base_dir:
    :param entities:
    :return:
    """

    parsed_sentences = {}

    # First iterate all documents, and preprocess all sentences
    token_seq = {}

    for f in os.listdir(base_dir):

        logging.info('Parsing {}'.format(f))
        tree = ET.parse(base_dir + '/' + f)
        root = tree.getroot()

        for sentence in root:
            sentence_id = sentence.get('id')

            if len(sentence.findall('pair')) > 0:  # skip sentences without pairs

                parsed_sentence = parse_sentence_spacy(sentence.get('text'), entities[sentence_id])
                parsed_sentences[sentence_id] = parsed_sentence

                tokens = []

                for t in parsed_sentence:
                    tokens.append(t.text.replace(' ', '_').replace('\t', '_').replace('\n', '_'))

                token_seq[sentence_id] = tokens

    wordnet_tags = run_sst(base_dir, token_seq)

    return parsed_sentences, wordnet_tags

def get_network_graph_spacy(document):
    """Convert the dependencies of the spacy document object to a networkX graph

    :param document: spacy parsed document object
    :return: networkX graph object and nodes list
    """

    edges = []
    nodes = []

    # Ensure that every token is connected
    for s in document.sents:
        edges.append(('ROOT', '{0}-{1}'.format(s.root.lower_, s.root.i)))

    for token in document:
        nodes.append('{0}-{1}'.format(token.lower_, token.i))

        for child in token.children:
            edges.append(('{0}-{1}'.format(token.lower_, token.i), '{0}-{1}'.format(child.lower_, child.i)))

    return nx.Graph(edges), nodes

def get_head_tokens_spacy(base_dir, entities, sentence, positive_entities):
    """

    :param base_dir:
    :param entities: dictionary mapping entity IDs to (offset, text)
    :param sentence: sentence parsed by spacy
    :param positive_entities:
    :return: dictionary mapping head tokens word-idx to entity IDs
    """

    sentence_head_tokens_type_1 = {}
    sentence_head_tokens_type_2 = {}
    pos_gv = set()
    neg_gv = set()

    different_type_entities = set(base_dir.split('/')[1].split('_'))
    different_type_entities = list(different_type_entities)

    for eid in entities:
        offset = (entities[eid][0][0], entities[eid][0][-1])
        entity_tokens = sentence.char_span(offset[0], offset[1])

        i = 1
        while not entity_tokens and i + offset[1] < len(sentence.text) + 1:
            entity_tokens = sentence.char_span(offset[0], offset[1] + i)
            i += 1

        i = 0
        while not entity_tokens and offset[0] - i > 0:
            entity_tokens = sentence.char_span(offset[0] - i, offset[1])
            i += 1

        if not entity_tokens:
            logging.warning(
                ('No tokens found:', entities[eid], sentence.text, '|'.join([t.text for t in sentence])))
        else:
            head_token = '{0}-{1}'.format(entity_tokens.root.lower_,entity_tokens.root.i)

            if eid in positive_entities:
                pos_gv.add(entity_tokens.root.head.lower_)
            else:
                neg_gv.add(entity_tokens.root.head.lower_)

            if len(different_type_entities) == 1:
                if head_token in sentence_head_tokens_type_1:
                    logging.warning(('Head token conflict:', sentence_head_tokens_type_1[head_token], entities[eid]))

                sentence_head_tokens_type_1[head_token] = eid
                sentence_head_tokens_type_2[head_token] = eid

            elif len(different_type_entities) == 2:
                if head_token in sentence_head_tokens_type_1:
                    logging.warning(('Head token conflict:', sentence_head_tokens_type_1[head_token], entities[eid]))
                elif head_token in sentence_head_tokens_type_2:
                    logging.warning(('Head token conflict:', sentence_head_tokens_type_2[head_token], entities[eid]))

                if biomedical_entity_to_ontology[different_type_entities[0]] == entities[eid][2][0]:
                    sentence_head_tokens_type_1[head_token] = eid
                elif biomedical_entity_to_ontology[different_type_entities[1]] == entities[eid][2][0]:
                    sentence_head_tokens_type_2[head_token] = eid

    return sentence_head_tokens_type_1, sentence_head_tokens_type_2, pos_gv, neg_gv

def process_sentence_spacy(base_dir, sentence, sentence_entities, sentence_pairs, positive_entities, wordnet_tags = None, mask_entities = True, min_sdp_len = 0, max_sdp_len = 15):
    """Process sentence to obtain labels, instances and classes for a ML classifier

    :param base_dir:
    :param sentence: sentence processed by spacy
    :param sentence_entities: dictionary mapping entity ID to ((e_start, e_end), text, paths_to_root)
    :param sentence_pairs: dictionary mapping pairs of known entities in this sentence to pair types
    :param positive_entities:
    :param wordnet_tags:
    :param mask_entities:
    :param min_sdp_len:
    :param max_sdp_len:
    :return: labels of each pair (according to sentence_entities, word vectors and classes (pair types according to sentence_pairs)
    """

    left_word_vectors = []
    right_word_vectors = []
    left_wordnets = []
    right_wordnets = []
    classes = []
    labels = []

    graph, nodes_list = get_network_graph_spacy(sentence)
    sentence_head_tokens_type_1, sentence_head_tokens_type_2, pos_gv, neg_gv = get_head_tokens_spacy(base_dir, sentence_entities, sentence, positive_entities)

    entity_offsets = [sentence_entities[x][0][0] for x in sentence_entities]

    for (e1, e2) in product(sentence_head_tokens_type_1, sentence_head_tokens_type_2):

        if sentence_head_tokens_type_1.get(e1):
            if int(sentence_head_tokens_type_1[e1].split('e')[-1]) > int(sentence_head_tokens_type_2[e2].split('e')[-1]):
                e1, e2 = e2, e1
        else:
            if int(sentence_head_tokens_type_1[e2].split('e')[-1]) > int(sentence_head_tokens_type_2[e1].split('e')[-1]):
                e2, e1 = e1, e2

        if sentence_head_tokens_type_1.get(e1):
            e1_text = sentence_entities[sentence_head_tokens_type_1[e1]]
            e2_text = sentence_entities[sentence_head_tokens_type_2[e2]]

        else:
            e1_text = sentence_entities[sentence_head_tokens_type_1[e2]]
            e2_text = sentence_entities[sentence_head_tokens_type_2[e1]]

        if 'train' in base_dir:
            middle_text = sentence.text[e1_text[0][-1]:e2_text[0][0]]

            if middle_text.strip() in string.punctuation:
                continue

        try:
            sdp = nx.shortest_path(graph, source = e1, target = e2)

            if len(sdp) < min_sdp_len or len(sdp) > max_sdp_len:
                continue

            neg = False
            is_neg_gv = False
            for i, element in enumerate(sdp):
                token_text = element.split('-')[0]
                if (i == 1 or i == len(sdp) - 2) and token_text in neg_gv_list:
                    logging.info('Skipped gv {} {}:'.format(token_text, str(sdp)))


            if neg or is_neg_gv:
                continue

            vector = []
            wordnet_vector = []
            negations = 0
            head_token_position = None

            for i, element in enumerate(sdp):
                if element != 'ROOT':
                    token_idx = int(element.split('-')[-1])  # get the index of the token
                    sdp_token = sentence[token_idx]  # get the token object

                    if mask_entities and sdp_token.idx in entity_offsets:
                        vector.append('entity')
                    else:
                        vector.append(sdp_token.text)
                    if wordnet_tags:
                        wordnet_vector.append(wordnet_tags[token_idx])

                    head_token = '{}-{}'.format(sdp_token.head.lower_, sdp_token.head.i)  # get the key of head token

                    # Head token must not have its head in the path, otherwise that would be the head token
                    # In some cases the token is its own head
                    if head_token not in sdp or head_token == element:
                        head_token_position = i + negations

            if head_token_position is None:
                print('Head token not found:', e1_text, e2_text, sdp)
                sys.exit()
            else:
                left_vector = vector[:head_token_position+1]
                right_vector = vector[head_token_position:]
                left_wordnet = wordnet_vector[:head_token_position+1]
                right_wordnet = wordnet_vector[head_token_position:]

            left_word_vectors.append(left_vector)
            right_word_vectors.append(right_vector)
            left_wordnets.append(left_wordnet)
            right_wordnets.append(right_wordnet)

        except nx.exception.NetworkXNoPath:
            logging.warning('No path:', e1_text, e2_text, graph.nodes())
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])

        except nx.NodeNotFound:
            logging.warning(('Node not found:', e1_text, e2_text, e1, e2, list(sentence), graph.nodes()))
            left_word_vectors.append([])
            right_word_vectors.append([])
            left_wordnets.append([])
            right_wordnets.append([])

        if sentence_head_tokens_type_1.get(e1):
            labels.append((sentence_head_tokens_type_1[e1], sentence_head_tokens_type_2[e2]))
            if (sentence_head_tokens_type_1[e1], sentence_head_tokens_type_2[e2]) in sentence_pairs:
                classes.append(sentence_pairs[(sentence_head_tokens_type_1[e1], sentence_head_tokens_type_2[e2])])
            else:
                classes.append(0)
        else:
            labels.append((sentence_head_tokens_type_1[e2], sentence_head_tokens_type_2[e1]))
            if (sentence_head_tokens_type_1[e2], sentence_head_tokens_type_2[e1]) in sentence_pairs:
                classes.append(sentence_pairs[(sentence_head_tokens_type_1[e2], sentence_head_tokens_type_2[e1])])
            else:
                classes.append(0)

    return labels, (left_word_vectors, right_word_vectors), (left_wordnets, right_wordnets), classes, pos_gv, neg_gv


# --------------------------------------------------------------
#    PARSE CORPUS WITH SDP VECTORS FOR EACH RELATION INSTANCE
# --------------------------------------------------------------

def get_sdp_instances(base_dir, name_to_id, synonym_to_id, parser = 'spacy'):
    """Parse XML corpus, return vectors of SDP of each relation instance

    :param base_dir: directory containing the XML sentences
    :param name_to_id:
    :param synonym_to_id:
    :param parser:
    :return: labels (eid1, eid2), instances (vectors), classes (0/1), common ancestors, l/r ancestors, l/r wordnet
    """

    entities, positive_entities = get_sentence_entities(base_dir, name_to_id, synonym_to_id, entity_id_max = 20)

    if parser == 'spacy':
        parsed_sentences, wordnet_sentences = parse_xml_sentences_spacy(base_dir, entities)

    else:
        parsed_sentences = wordnet_sentences = None

    left_instances = []
    right_instances = []
    left_ancestors = []
    right_ancestors = []
    common_ancestors = []
    left_wordnet = []
    right_wordnet = []
    classes = []
    labels = []
    all_pos_gv = set()
    all_neg_gv = set()

    for f in os.listdir(base_dir):
        logging.info('Generating instances: {}'.format(f))

        tree = ET.parse(base_dir + '/' + f)
        root = tree.getroot()
        for sentence in root:
            sentence_id = sentence.get('id')

            # Important: in the XML files the type of relation must be explicitly indicated here
            sentence_pairs = {}
            for pair in sentence.findall('pair'):

                if pair.get('ddi') == 'true':
                    # sentence_pairs[(pair.get('e1'), pair.get('e2'))] = label_to_pair_type[pair.get('type')]
                    sentence_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

                elif pair.get('relation') == 'true':
                    sentence_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

                elif pair.get('pgr') == 'true':
                    sentence_pairs[(pair.get('e1'), pair.get('e2'))] = EFFECT

            if len(sentence.findall('pair')) > 0:  # skip sentences without pairs
                sentence_entities = entities[sentence_id]
                parsed_sentence = parsed_sentences[sentence_id]
                wordnet_sentence = wordnet_sentences[sentence_id]

                if parser == 'spacy':
                    sentence_labels, sentence_we_instances, sentence_wn_instances, sentence_classes, pos_gv, neg_gv = \
                        process_sentence_spacy(base_dir, parsed_sentence, sentence_entities, sentence_pairs, positive_entities, wordnet_sentence)

                else:
                    sentence_labels = sentence_we_instances = sentence_classes = sentence_wn_instances = pos_gv = neg_gv = None

                sentence_ancestors, sentence_subpaths = get_ancestors(sentence_labels, sentence_entities)

                labels += sentence_labels
                left_instances += sentence_we_instances[0]
                right_instances += sentence_we_instances[1]
                classes += sentence_classes
                common_ancestors += sentence_ancestors
                left_ancestors += sentence_subpaths[0]
                right_ancestors += sentence_subpaths[1]

                left_wordnet += sentence_wn_instances[0]
                right_wordnet += sentence_wn_instances[1]

                all_pos_gv.update(pos_gv)
                all_neg_gv.update(neg_gv)

    return labels, (left_instances, right_instances), classes, common_ancestors, (left_ancestors, right_ancestors), (left_wordnet, right_wordnet), all_neg_gv, all_pos_gv