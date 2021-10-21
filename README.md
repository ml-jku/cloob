# CLOOB: Modern Hopfield Networks with InfoLOOB Outperform CLIP

_Andreas Fürst<sup>* 1</sup>,
Elisabeth Rumetshofer<sup>* 1</sup>,
Viet Tran<sup>1</sup>,
Hubert Ramsauer<sup>1</sup>,
Fei Tang<sup>3</sup>,
Johannes Lehner<sup>1</sup>,
David Kreil<sup>2</sup>,
Michael Kopp<sup>2</sup>,
Günter Klambauer<sup>1</sup>,
Angela Bitto-Nemling<sup>1</sup>,
Sepp Hochreiter<sup>1 2</sup>_

<sup>1</sup> ELLIS Unit Linz and LIT AI Lab, Institute for Machine Learning, Johannes Kepler University Linz, Austria  
<sup>2</sup> Institute of Advanced Research in Artificial Intelligence (IARAI)  
<sup>3</sup> HERE Technologies  
<sup>*</sup> Equal contribution

---

#### Detailed blog post on this paper at [this link](https://ml-jku.github.io/cloob).
#### The full paper is available [here](link).

---

# Implementation of CLOOB
This repository contains the implemenation of CLOOB used to obtain the results reported in the paper.
The implementation is based on [OpenCLIP](https://github.com/mlfoundations/open_clip), an open source implementation of OpenAI's [CLIP](https://arxiv.org/abs/2103.00020).


## Setup
We provide an 'environment.yml' file to set up a conda environment with all required packages.
Run the following command to clone the repository and create the environment.

```bash
# Clone repository and swtich into the directory
git clone https://github.com/ml-jku/cloob
cd cloob

# Create the environment and activate it
conda env create --file environment.yml
conda activate cloob

# Additionally, webdataset needs to be installed from git repo for pre-training on YFCC 
pip install git+https://github.com/tmbdev/webdataset.git

# Add the directory to the PYTHONPATH environment variable
export PYTHONPATH="$PYTHONPATH:$PWD/src"
```

## Data
For pre-training we use the two datasets supported by OpenCLIP, namely Conceptual Captions and YFCC.

### Conceptual Captions
OpenCLIP already provides a script to download and prepare the Conceptual Captions dataset, which contains 2.89M training images and 13k validation images.
First, download the [Conceptual Captions URLs](https://ai.google.com/research/ConceptualCaptions/download) and then run the script `gather_cc.py`.

```bash
python3 src/data/gather_cc.py path/to/Train_GCC-training.tsv path/to/Validation_GCC-1.1.0-Validation.tsv
```

### YFCC

We use the same subset of ~15M images from the [YFCC100M](http://mmcommons.org) dataset as CLIP. 
They provide a list of (line number, photo identifier, photo hash) of each image contained in this subset [here](https://openaipublic.azureedge.net/clip/data/yfcc100m_subset_data.tsv.bz2).

For more information see [YFCC100m Subset](https://github.com/openai/CLIP/blob/main/data/yfcc100m.md) on OpenAI's github.

### Downstream Tasks
In the paper we report results on several downstream tasks. 
Except for ImageNet we provide links to already pre-processed versions (where necessary) of the respective test set.

| Dataset       | Description | Official                                                                | Processed |
|---------------|-------------|-------------------------------------------------------------------------|-----------|
| Birdsnap      | This dataset contains images of North American bird species, however <br> our dataset is smaller than reported in CLIP as some samples are no longer available.            | [Link](http://thomasberg.org/datasets/birdsnap/1.1/birdsnap.tgz)        | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/birdsnap.zip)  |
| Country211    | This dataset was published in CLIP and is a small subset of the YFCC100m dataset. <br> It consists of photos that can be assigned to 211 countries via GPS coordinates. <br> For each country 200 photos are sampled for the training set and 100 for testing.            | [Link](https://github.com/openai/CLIP/blob/main/data/country211.md) | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/country211.zip)  |
| Flowers102    | Images of 102 flower categories commonly occuring in the United Kingdom were collected.<br> Several classes are very similar and there is a large variation in scale, pose and lighting.            | [Link](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)          | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/flowers102.zip)  |
| GTSRB         | This dataset was released for a challenge held at the IJCNN 2011. <br> The dataset contains images of german traffic signs from more than 40 classes.           | [Link](https://benchmark.ini.rub.de/gtsrb_news.html)                | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/gtsrb.zip)  |
| Stanford Cars | This dataset contains images of 196 car models at the level of make, <br> model and year (e.g. Tesla Model S Sedan 2012).            | [Link](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)       | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/stanford_cars.zip)  |
| UCF101        | The dataset has been created by extracting the middle frame from each video.            | [Link](https://www.crcv.ucf.edu/data/UCF101.php)                    | [Link](https://ml.jku.at/research/CLOOB/downloads/zeroshot_datasets/ucf101.zip)  |
| ImageNet      | This dataset spans 1000 object classes and contains 1,281,167 training images, <br> 50,000 validation images and 100,000 test images.            | [Link](https://image-net.org/download.php)                          | -  |
| ImageNet v2   | The ImageNetV2 dataset contains new test data for the ImageNet benchmark.            | [Link](https://github.com/modestyachts/ImageNetV2)                  | -  |


## Usage
In the following there is an example command for pretraining on CC with an effective batch size of 512 when used on 4 GPUs.

```bash
python -u src/training/main.py \
--train-data="<dataset-dir>/conceptual_captions/Train-GCC-training_output.csv" \
--val-data="<dataset-dir>/conceptual_captions/Validation_GCC-1.1.0-Validation_output.csv" \
--path-data="<dataset-dir>/conceptual_captions" \
--imagenet-val="<dataset-dir>/imagenet/val" \
--warmup 20000 \
--batch-size=128 \
--lr=1e-3 \
--wd=0.1 \
--lr-scheduler="cosine-restarts" \
--restart-cycles=10 \
--epochs=70 \
--method="cloob" \
--init-inv-tau=30 \
--init-scale-hopfield=8 \
--workers=8 \
--model="RN50" \
--dist-url="tcp://127.0.0.1:6100" \
--batch-size-eval=512
```

### Zeroshot evaluation of downstream tasks
We provide a [Jupyter notebook](src/notebooks/zeroshot.ipynb) to perform zeroshot evaluation with a trained model.


## LICENSE
MIT LICENSE
