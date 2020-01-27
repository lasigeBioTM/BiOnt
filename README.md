# BiOnt: Deep Learning using Multiple Biomedical Ontologies for Relation Extraction

To perform relation extraction, our deep learning system, BiOnt, employs four types of biomedical ontologies, namely, the Gene Ontology, the Human Phenotype Ontology, the Human Disease Ontology, and the Chemical Entities of Biological Interest, regarding gene-products, phenotypes, diseases, and chemical compounds, respectively. 

Our academic paper which describes BiONT in detail was accepted for publication at the 42nd European Conference on Information Retrieval, and for now can be found [here](https://arxiv.org/abs/2001.07139).

## Downloading Pre-Trained Weights

Available versions of pre-trained weights are as follows:

* [DRUG-DRUG](https://drive.google.com/open?id=1q-180Sz6YGngswxmbYsQjetnXxI0Shps)
* [DRUG-DISEASE](https://drive.google.com/open?id=1IhqmQ9UGCZ-0pOIRU8FXGGxJZJ2b13rk)
* [GENE-PHENOTYPE](https://drive.google.com/open?id=1-b9prbiEuMAuR3bRIkXo_vYknIleiv1c)

Training details are described in our academic paper.

## Getting Started

````
 cd bin/
 git git clone git@github.com:lasigeBioTM/DiShIn.git
````

Use the Dockerfile to setup the rest of the experimental environment or the [BiOnt Image](https://hub.docker.com/r/dpavot/biont) available at Docker Hub.
   
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

## Reference

- Diana Sousa and Francisco M. Couto. 2020. BiOnt: Deep Learning using Multiple Biomedical Ontologies for Relation Extraction. 