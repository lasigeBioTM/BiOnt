#!/usr/bin/env bash

set -e

# $2: type_of_action
# $3: pair_type
# $4: preprocess_what
# $5: input_path

python3 src/ontologies_embeddings.py preprocess DRUG-DRUG train corpora/drug_drug/train
# python3 src/ontologies_embeddings.py preprocess DRUG-GENE train corpora/drug_gene/train
# python3 src/ontologies_embeddings.py preprocess DRUG-PHENOTYPE train corpora/drug_phenotype/train
# python3 src/ontologies_embeddings.py preprocess DRUG-DISEASE train corpora/drug_disease/train
# python3 src/ontologies_embeddings.py preprocess GENE-GENE train corpora/gene_gene/train
# python3 src/ontologies_embeddings.py preprocess GENE-PHENOTYPE train corpora/gene_phenotype/train
# python3 src/ontologies_embeddings.py preprocess GENE-DISEASE train corpora/gene_disease/train
# python3 src/ontologies_embeddings.py preprocess PHENOTYPE-PHENOTYPE train corpora/phenotype_phenotype/train
# python3 src/ontologies_embeddings.py preprocess PHENOTYPE-DISEASE train corpora/phenotype_disease/train
# python3 src/ontologies_embeddings.py preprocess DISEASE-DISEASE train corpora/disease_disease/train

python3 src/ontologies_embeddings.py preprocess DRUG-DRUG test corpora/drug_drug/test
# python3 src/ontologies_embeddings.py preprocess DRUG-GENE test corpora/drug_gene/test
# python3 src/ontologies_embeddings.py preprocess DRUG-PHENOTYPE test corpora/drug_phenotype/test
# python3 src/ontologies_embeddings.py preprocess DRUG-DISEASE test corpora/drug_disease/test
# python3 src/ontologies_embeddings.py preprocess GENE-GENE test corpora/gene_gene/test
# python3 src/ontologies_embeddings.py preprocess GENE-PHENOTYPE test corpora/gene_phenotype/test
# python3 src/ontologies_embeddings.py preprocess GENE-DISEASE test corpora/gene_disease/test
# python3 src/ontologies_embeddings.py preprocess PHENOTYPE-PHENOTYPE test corpora/phenotype_phenotype/test
# python3 src/ontologies_embeddings.py preprocess PHENOTYPE-DISEASE test corpora/phenotype_disease/test
# python3 src/ontologies_embeddings.py preprocess DISEASE-DISEASE test corpora/disease_disease/test

# $2: type_of_action
# $3: pair_type
# $4: model_name
# $6:: channels

python3 src/ontologies_embeddings.py train DRUG-DRUG model_name words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py train DRUG-GENE model words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py train DRUG-PHENOTYPE model words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py train DRUG-DISEASE model_sentences_no_2 words wordnet #concatenation_ancestors
# python3 src/ontologies_embeddings.py train GENE-GENE model words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py train GENE-PHENOTYPE model_no_3 words wordnet #concatenation_ancestors
# python3 src/ontologies_embeddings.py train GENE-DISEASE model words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py train PHENOTYPE-PHENOTYPE model words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py train PHENOTYPE-DISEASE model words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py train DISEASE-DISEASE model words wordnet concatenation_ancestors common_ancestors

# $2: type_of_action
# $3: pair_type
# $4: model_name
# $5: gold_standard
# $6:: channels

python3 src/ontologies_embeddings.py test DRUG-DRUG model_name corpora/drug_drug/test words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py test DRUG-GENE model corpora/drug_gene/test/ words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py test DRUG-PHENOTYPE model corpora/drug_phenotype/test/ words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py test DRUG-DISEASE model_sentences_no_2 corpora/drug_disease/test/ words wordnet #concatenation_ancestors
# python3 src/ontologies_embeddings.py test GENE-GENE model corpora/gene_gene/test/ words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py test GENE-PHENOTYPE model_no_3 corpora/gene_phenotype/test/ words wordnet #concatenation_ancestors
# python3 src/ontologies_embeddings.py test GENE-DISEASE model corpora/gene_disease/test/ words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py test PHENOTYPE-PHENOTYPE model corpora/phenotype_phenotype/test/ words wordnet concatenation_ancestors common_ancestors
# python3 src/ontologies_embeddings.py test PHENOTYPE-DISEASE model corpora/phenotype_disease/test/ words wordnet concatenation_ancestors
# python3 src/ontologies_embeddings.py test DISEASE-DISEASE model corpora/disease_disease/test/ words wordnet concatenation_ancestors common_ancestors
