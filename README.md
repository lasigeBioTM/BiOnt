# BiOnt: Deep Learning using Multiple Biomedical Ontologies for Relation Extraction

To perform relation extraction, our deep learning system, BiOnt, employs four types of biomedical ontologies, namely, the Gene Ontology, the Human Phenotype Ontology, the Human Disease Ontology, and the Chemical Entities of Biological Interest, regarding gene-products, phenotypes, diseases, and chemical compounds, respectively. 

Our academic paper which describes BiOnt in detail can be found [here](https://doi.org/10.1007/978-3-030-45442-5_46).

## Downloading Pre-Trained Weights

Available versions of pre-trained weights are as follows:

* [DRUG-DRUG](https://zenodo.org/records/10230879?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4ZTE5ZGUxLTQ3ZDEtNDIzZS05MTBhLWNmOGQ4NDc4Mzc4MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MGRlM2VhOTJiNjZiODk2ODQ0N2ZiOTJmMTRlN2Y5NCJ9.pMax9Vk9YV8CRUl-Ga2VxQXUVuXOkzcaW5NECrsm9doN1e5mizR3VVrXkAcDGLH5FjR642wcd_EqzUFSo28rnA)
* [DRUG-DISEASE](https://zenodo.org/records/10230879?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4ZTE5ZGUxLTQ3ZDEtNDIzZS05MTBhLWNmOGQ4NDc4Mzc4MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MGRlM2VhOTJiNjZiODk2ODQ0N2ZiOTJmMTRlN2Y5NCJ9.pMax9Vk9YV8CRUl-Ga2VxQXUVuXOkzcaW5NECrsm9doN1e5mizR3VVrXkAcDGLH5FjR642wcd_EqzUFSo28rnA)
* [GENE-PHENOTYPE](https://zenodo.org/records/10230879?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY4ZTE5ZGUxLTQ3ZDEtNDIzZS05MTBhLWNmOGQ4NDc4Mzc4MCIsImRhdGEiOnt9LCJyYW5kb20iOiI0MGRlM2VhOTJiNjZiODk2ODQ0N2ZiOTJmMTRlN2Y5NCJ9.pMax9Vk9YV8CRUl-Ga2VxQXUVuXOkzcaW5NECrsm9doN1e5mizR3VVrXkAcDGLH5FjR642wcd_EqzUFSo28rnA)

The training details are described in our academic paper.

## Getting Started

````
 cd bin/
 git clone git@github.com:lasigeBioTM/DiShIn.git
````

Use the Dockerfile to set up the rest of the experimental environment or the [BiOnt Image](https://hub.docker.com/r/dpavot/biont) available at Docker Hub. 
## Preparing Data

* $2: type_of_action
* $3: pair_type
* $4: preprocess_what
* $5: input_path

### Example:

````
 python3 src/ontologies_embeddings.py preprocess DRUG-DRUG train corpora/drug_drug/train
 python3 src/ontologies_embeddings.py preprocess DRUG-DRUG test corpora/drug_drug/test
````

For more options check **model.sh**.

## Train Model

* $2: type_of_action
* $3: pair_type
* $4: model_name
* $6:: channels

### Example:

````
 python3 src/ontologies_embeddings.py train DRUG-DRUG model_name words wordnet concatenation_ancestors common_ancestors
````

For more options check **model.sh**.

## Predict New Data

* $2: type_of_action
* $3: pair_type
* $4: model_name
* $5: gold_standard OR data_to_test
* $6:: channels

### Example:

````
 python3 src/ontologies_embeddings.py test DRUG-DRUG model_name corpora/drug_drug/test words wordnet concatenation_ancestors common_ancestors
````

For more options check **model.sh**.

## Reference

- Diana Sousa and Francisco M. Couto. 2020. BiOnt: Deep Learning using Multiple Biomedical Ontologies for Relation Extraction. In Jose J. et al. (eds) Advances in Information Retrieval. ECIR 2020. Lecture Notes in Computer Science, Volume 12036, pages 367-374. Springer, Cham.
