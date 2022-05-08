# ViBERTgrid PyTorch

An Unofficial PyTorch implementation of *Lin et al. ViBERTgrid: A Jointly Trained Multi-Modal 2D Document Representation for Key Information Extraction from Documents. ICDAR, 2021.*

## Repo Structure

- **data**: data loaders
- **model**: ViBERTgrid net architecture
- **pipeline**: preprocessing, trainer, evaluation metrics
- **utils**: data visualize, dataset spilt
- **deployment**: examples for inference and deployment
- **train_\***.py: main train scripts
- **eval_\***.py: evaluation scripts

## Usage

### 1. Env Setting

```

pip install -r requirements.txt

```

### 2. Data Preprocessing

The following componets are required for the processed dataset:
- the original image [bs, 3, h, w]
- label_csv, label file in csv format. Each row should contains text, left-coor, right-coor, top-coor, bot-coor, class type, pos_neg type information of each text segment.

#### 2.1 ICDAR SROIE
It is worth noting that the labeling of the dataset will strongly affect the final result.
The original SROIE dataset (https://rrc.cvc.uab.es/?ch=13&com=tasks) only contains
text label of the key information. Coordinates, however, are necessary for constructing the grid, hence **we re-labeled the dataset to obtain the coordinates**. Unfortunately this re-labeld version cannot be made public for some reasons.

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

#### 2.3 FUNSD
Images and labels can be found [here](https://guillaumejaume.github.io/FUNSD/).  
The FUNSD dataset contains two subtasks, `entity labeling` and `entity linking`. The ViBERTgrid model can only perform KIE on the first task, in which the text contents are labeled into 3 key types(header, question, answer).  Run the following commands to generate formatted labels.

```shell

python ./pipeline/funsd_data_preprocessing.py

```

### 3. Training

First you need to set up configurations. An example config file `example_config.yaml` is provided. Then run the following command. Replace * with SROIE, EPHOIE or FUNSD.

```shell

torchrun --nnodes 1 --nproc_per_node 2 ./train_*.py -c dir_to_config_file.yaml

```

### 4. Inference

Scripts for inference are provided in the `deployment` folder. run `inference_*` to get the KIE result in json format.

------

## Adaptation and Exploration in This Implementationi

### 1. Data Level
In the paper, the author applied classification on **word-level**, which predicts the key type of each word and join the words that belong to the same class as the final entity-level result. 

In fact, ViBERTgrid can work on any data level, like **line-level** or **char-level**. Choosing a proper data level may significantly boost the final score. According to our experiment result, **Line-level** is the most suitable choice for SROIE dataset, **char-level** for EPHOIE and **segment-level** for FUNSD.

### 2. CNN Backbone
The author of the paper used an ImageNet pretrained ResNet18-D to initialize the weights of CNN backbone. Pretrained weights of ResNet-D, however, cannot be found in PyTorch's model zoo. Hence we use an ImageNet pretrained ResNet34 instead.

CNN backbones can be changed by setting different values in the config file, supported backbones are shown below
- resnet_18_fpn
- resnet_34_fpn
- resnet_18_fpn_pretrained
- resnet_34_fpn_pretrained
- resnet_18_D_fpn
- resnet_34_D_fpn

### 3. Field Type Classification Head
> Some words could be labeled with more than one field type tags (similar to the nested named entity recognition task), we design two classifiers to input and perform field type classification for each word

To solve the problem mentioned above, the author designed a complicated two-stage classifier. We found that this classifier does not work well and hard to fine-tune. Since that the multi-label problem does not occur in SROIE, EPHOIE and FUNSD dataset, we use a one-stage multi-class classifier with multi-class cross entropy loss to replace the original design.

Experiment shows that an additional, independent key information binary classifier may imporve the final F1-score. The classifier indicates whether a text segment belongs to key information or not, which may boost the recall metric`.

### 5. Auxiliary Semantic Segmentation Head

In our case, the auxiliary semantic segmentation head does not help on both the SROIE and EPHOIE dataset. You can remove this branch by setting the `loss_control_lambda` to zero in cofiguration file.

### 6. Tag Mode

The model can directly predict the category of each text-line/char/segment, or predict the BIO tags under the restriction of a CRF layer. We found that the representative ability of ViBERTgrid is good enough and the direct prediction works best. Using a BIO tagging with CRF layers is unecessary and has a negative effect.

-----


## Experiment Results

| Dataset | Configuration                                       | # of Parameters | F1    |
| ------- | --------------------------------------------------- | --------------- | ----- |
| SROIE   | original paper, BERT-Base, ResNet18-D-pretrained    | 142M            | 96.25 |
| SROIE   | original paper, RoBERTa-Base, ResNet18-D-pretrained | 147M            | 96.40 |
| SROIE   | BERT-Base uncased, ResNet34-pretrained              | 151M            | 97.16 |
| EPHOIE  | BERT-Base chinese, ResNet34-pretrained              | 145M            | 96.55 |
| FUNSD   | BERT-Base uncased, ResNet34-pretrained              | 151M            | 87.63 |


## Note
- Due to source limitations, I used 2 NVIDIA TITAN X for training, which can only afford a batch size of 4 (2 on each GPU). The loss curve is not stable in this situation and may affect the performance.

