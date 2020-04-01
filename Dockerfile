FROM nvidia/cuda:10.2-base
MAINTAINER Diana Sousa (dfsousa@lasige.di.fc.ul.pt)


WORKDIR /


# --------------------------------------------------------------
#                         GENERAL SET UP
# --------------------------------------------------------------

RUN apt-get update -y && apt-get install wget -y && apt-get install curl -y && apt-get install nano -y


# --------------------------------------------------------------
#                  COPY REPOSITORY DIRECTORIES                
# --------------------------------------------------------------

COPY bin/DiShIn/ bin/DiShIn/
COPY src/ src/
COPY corpora/ corpora/
COPY data/ data/
COPY models/ models/
COPY results/ results/
COPY temp/ temp/


# --------------------------------------------------------------
#               PYTHON LIBRARIES AND CONFIGURATION
# --------------------------------------------------------------

RUN apt-get update && apt-get install -y python3 python3-pip python3-dev &&  apt-get autoclean -y
#RUN apt-get update && apt-get install sqlite3 libsqlite3-dev -y
#RUN unlink /usr/bin/pip
RUN ln -s $(which pip3) /usr/bin/pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3
RUN pip3 install --upgrade setuptools
RUN pip3 install numpy==1.16.4
RUN pip3 install tensorflow-gpu
RUN pip3 install gensim==3.1.0
RUN pip3 install Keras==2.1.5
RUN pip3 install rdflib
RUN pip3 install sklearn==0.0
RUN pip3 install matplotlib
RUN apt-get update && apt-get install -y git && apt-get autoclean -y
RUN git clone https://github.com/dpavot/obonet
RUN cd obonet && python3 setup.py install
RUN pip3 install fuzzywuzzy==0.15.1
RUN pip3 install spacy==2.0.10
RUN pip3 install scipy==1.0.0
RUN pip3 install python-Levenshtein==0.12.0
RUN python3 -m spacy download en_core_web_sm


# --------------------------------------------------------------
#                GENIASS (REQUIRES RUBY AND MAKE)
# --------------------------------------------------------------

WORKDIR /bin
RUN wget -q http://www.nactem.ac.uk/y-matsu/geniass/geniass-1.00.tar.gz && \
    tar -xvzf geniass-1.00.tar.gz && \
    rm geniass-1.00.tar.gz
WORKDIR geniass
RUN apt-get update -y && apt-get install -y build-essential g++ make && make


# --------------------------------------------------------------
#                         SST LIGHT 0.4
# --------------------------------------------------------------

WORKDIR /bin
RUN apt-get update && apt-get install -y zip && apt-get autoclean -y
RUN wget -q https://sourceforge.net/projects/supersensetag/files/sst-light/sst-light-0.4/sst-light-0.4.tar.gz && \
    tar -xvzf sst-light-0.4.tar.gz && \
    rm sst-light-0.4.tar.gz
WORKDIR sst-light-0.4
#RUN apt-get update -y && make (error to solve)


# --------------------------------------------------------------
#                             DiShIn
# --------------------------------------------------------------

WORKDIR /bin
RUN wget -q ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/chebi.db
RUN wget -q www.geneontology.org/ontology/go.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/go.db
RUN wget -q http://purl.obolibrary.org/obo/hp.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/hp.db
RUN wget -q https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.owl
RUN wget -q http://labs.rd.ciencias.ulisboa.pt/dishin/do.db

#RUN git clone git@github.com:lasigeBioTM/DiShIn.git

WORKDIR /data

RUN wget -q ftp://ftp.ebi.ac.uk/pub/databases/chebi/ontology/chebi.obo
RUN wget -q http://purl.obolibrary.org/obo/hp.obo
RUN wget -q http://purl.obolibrary.org/obo/go.obo
RUN wget -q https://raw.githubusercontent.com/DiseaseOntology/HumanDiseaseOntology/master/src/ontology/doid.obo
RUN wget -q http://evexdb.org/pmresources/vec-space-models/PubMed-w2v.bin

RUN export CUDA_VISIBLE_DEVICES=0

WORKDIR /
