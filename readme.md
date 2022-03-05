# ViBERTgrid PyTorch

An Unofficial PyTorch implementation of *Lin et al. ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents. ICDAR, 2021.*

## Repo Structure

- data: data loaders
- model: ViBERTgrid net architecture
- pipeline: preprocessing, trainer, evaluation metrics
- utils: data visualize, dataset spilt
- train_*.py: main train scripts
- test_*.py: main validation scripts
- inference_*.py: inference scripts that contains postprocessing, give information extraction result

## Usage

### 1. Env Setting

```

pip install -r requirements.txt

```

### 2. Data Preprocessing

The following componets are required for the processed dataset:
- the original image [b, 3, h, w]
- class_labels [b, 1, h, w], h/w equal to original image's h/w, single channel, value at each pixel represents the class type of the text-region where the pixel locates in.
- pos_neg_labels [b, 1, h, w], h/w equal to original image's h/w, single channel, value at each pixel represents the pos_net type (see sec 3.3 in the paper) of the text-region where the pixel locates in. 
- label_csv, label file in csv format. Each row should contains text, left-coor, right-coor, top-coor, bot-coor, class type, pos_neg type information of each segment.

#### 2.1 ICDAR SROIE
It is worth noting that the labeling of the dataset will strongly affect the final result.
The original SROIE dataset (https://rrc.cvc.uab.es/?ch=13&com=tasks) only contains
text label of the key information. Coordinates, however, are necessary for constructing the grid, hence **we re-labeled the dataset to obtain the coordinates**. It is a pity that this re-labeld dataset cannot be made public for some reasons.

Here I provide another method for matching the coordinates through regular expression and cosine similarity, referring to (https://github.com/antoinedelplace/Chargrid). **The matching result is not satisfying and will cause 3~5 points decrease in the final F1 score**. 

```shell

python ./pipeline/sroie_data_preprocessing.py

```

We recommend re-labeling the dataset on your own as it contains around 1k images and will not take up a lot of time, or find out a better solution to match the coordinates.

#### 2.2 EPHOIE
The dataset can be obtained from (https://github.com/HCIILAB/EPHOIE). Unzip password will be provided after submitting the application form.  
EPHOIE provides label in txt format, you should firstly convert it into json format on your own. Then run the following command:

```shell

python ./pipeline/ephoie_data_preprocessing.py

```

### 3. Training

First you need to set up configurations. Example at example_config.yaml. Then run the following command. Replace * with SROIE or EPHOIE.

```shell

torchrun --nnodes 1 --nproc_per_node 2 ./train_*.py -c ./config/network_config.yaml

```

### 4. Inference

Under Construction

## Experiment Results

|Dataset|Configuration|# of Parameters|Precision|Recall|F1|
|--|--|--|--|--|--|
|SROIE|original paper, BERT-Base|142M|-|-|96.25|
|SROIE|original paper, RoBERTa-Base|147M|-|-|96.40|
|SROIE|BERT-Base uncased|141M|94.43|97.03|95.71|
|SROIE|RoBERTa-Base uncased|146M|-|-|-|
|EPHOIE|BERT-Base chinese|-|-|-|97.41|


## Differences in this reimplementation
- Some words in the author's custom dataset could be labeled with more than one field type tags, thus he design two classifiers at Word-level Field Type Classification Head and Auxiliary Semantic Segmentation Head (See the original paper for details). Since the problem mentioned above does not appear in the ICDAR SROIE dataset, for simplicity, I used a single cross-entropy loss for classification instead.

## Note and Implementation Details
- Due to source limitations, I used 2 NVIDIA TITAN X for training, which can only afford a batch size of 4 (2 on each GPU). The loss curve is not stable in this situation and may affect the performance.
- OHEM does not help in the case of small batch size, as it tends to back propagate local hard samples inside a batch, not global hard samples.
- My implementation of random sampling in auxiliary semantic segmentation head is not capatible with `torch.amp`. `amp` will be disabled when a random sampling list is detected.

