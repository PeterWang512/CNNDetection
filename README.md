## Detecting CNN-Generated Images [[Project Page]](https://peterwang512.github.io/CNNDetection/)

**CNN-generated images are surprisingly easy to spot...for now**  
[Sheng-Yu Wang](https://peterwang512.github.io/), [Oliver Wang](http://www.oliverwang.info/), [Richard Zhang](https://richzhang.github.io/), [Andrew Owens](http://andrewowens.com/), [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/).
<br>In [ArXiv](https://arxiv.org/abs/1912.11035), 2019.

<img src='https://peterwang512.github.io/CNNDetection/images/teaser.png' width=1200>


## (1) Setup

### Install packages
- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`

### Download model weights
- Run `bash weights/download_weights.sh`


## (2) Quick start
 
```
# Model weights need to be downloaded.
python demo.py examples/real.png weights/blur_jpg_prob0.1.pth
python demo.py examples/fake.png weights/blur_jpg_prob0.1.pth
```

`demo.py` simply runs the model on a single image, and outputs the uncalibrated prediction.

## (3) Dataset
The testset evaluated in the paper can be downloaded [here](https://drive.google.com/file/d/1z_fD3UKgWQyOTZIBbYSaQ-hz4AzUrLC1/view?usp=sharing).

The zip file contains images from 13 CNN-based synthetic  algorithms, including the 12 testsets from the paper and images downloaded from whichfaceisreal.com. Images from each algorithm are stored in a separate folder. In each category, real images are in the `0_real` folder, and synthetic images are in the `1_fake` folder. 

Note: ProGAN, StyleGAN, StyleGAN2, CycleGAN testset contains multiple classes, which are stored in separate subdirectories.

A script for downloading the testset is as follows:

```
# Download the dataset
cd dataset
bash download_testset.sh
cd ..
```

## (4) Evaluation

After the testset and the model weights are downloaded, one can evaluate the models by running:

```
# Run evaluation script. Model weights need to be downloaded.
python eval.py
```

Besides print-outs, the results will also be stored in a csv file in the `results` folder. Configurations such as the path of the dataset, model weight are in `eval_config.py`, and one can modify the evaluation by changing the configurations. The following are the models' performances on the released set:

<b>[Blur+JPEG(0.1)]</b>

|Testset   |Accuracy|  AP |
|:--------:|:------:|:---:|
|ProGAN    |100.0%	|100.0%|
|StyleGAN  |87.1%	|99.6%|
|BigGAN    |70.2%	|84.5%|
|CycleGAN  |85.2%	|93.5%|
|StarGAN   |91.7%	|98.2%|
|GauGAN    |78.9%	|89.5%|
|CRN       |86.3%	|98.2%|
|IMLE      |86.2%	|98.4%|
|SITD      |90.3%	|97.2%|
|SAN       |50.5%	|70.5%|
|Deepfake  |53.5%	|89.0%|
|StyleGAN2 |84.4%	|99.1%|
|Whichfaceisreal|83.6%	|93.2%|



<b>[Blur+JPEG(0.5)]</b>

|Testset   |Accuracy|  AP  |
|:--------:|:------:|:----:|
|ProGAN    | 100.0%	|100.0%|
|StyleGAN  | 73.4%	|98.5% |
|BigGAN    | 59.0%	|88.2% |
|CycleGAN  | 80.8%	|96.8% |
|StarGAN   | 81.0%	|95.4% |
|GauGAN    | 79.3%	|98.1% |
|CRN       | 87.6%	|98.9% |
|IMLE      | 94.1%	|99.5% |
|SITD      | 78.3%	|92.7% |
|SAN       | 50.0%	|63.9% |
|Deepfake  | 51.1%	|66.3% |
|StyleGAN2 | 68.4%	|98.0% |
|Whichfaceisreal| 63.9%	|88.8% |



## (A) Acknowledgments

This repository borrows partially from the [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and the PyTorch [torchvision models](https://github.com/pytorch/vision/tree/master/torchvision/models) repositories. 

## (B) Citation, Contact

If you find this useful for your research, please consider citing this [bibtex](https://peterwang512.github.io/CNNDetection/bibtex.txt). Please contact Sheng-Yu Wang \<sheng-yu_wang at berkeley dot edu\> with any comments or feedback.

