#!/usr/bin/env bash
wget http://data.dmlc.ml/models/imagenet/synset.txt
mkdir models
cd models
wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json
wget http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params
mv Inception-BN-0126.params Inception-BN-0000.params
wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-symbol.json
wget http://data.dmlc.ml/models/imagenet/resnet/152-layers/resnet-152-0000.params
cd ..
mkdir images
cd images
wget https://www.what-dog.net/Images/faces2/scroll001.jpg
mv scroll001.jpg image.jpg
cd ..