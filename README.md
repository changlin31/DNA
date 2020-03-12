# DNA
This repository provides the evaluation code of our paper: [***Blockwisely Supervised Neural Architecture Search with Knowledge Distillation***](https://arxiv.org/abs/1911.13053).

## Our Trained Models 
- Our searched models have been trained from scratch and can be found in: https://drive.google.com/drive/folders/1Oqc2gq8YysrJq2i6RmPMLKqheGfB9fWH. 

- Here is a summary of our searched models:

    |    Model    |  FLOPs    |   Params |   Acc@1   |   Acc@5   |
    |:---------:|:---------:|:---------:|:---------:|:---------:|
    | DNA-a    |   348M     |	4.2M    |      77.1%    |       93.3%   |
    | DNA-b    |   394M     |	4.9M    |      77.5%    |       93.3%   |
    | DNA-c    |   466M     |	5.3M    |      77.8%    |       93.7%   |
    | DNA-d    |   611M     |	6.4M    |      78.4%    |       94.0%   |

## Usage
### 1. Requirements
- Install PyTorch ([pytorch.org](http://pytorch.org))
- Install third-party requirements
	- `pip install timm==0.1.14` We use this [Pytorch-Image-Models](https://github.com/rwightman/pytorch-image-models/) codebase to validate our models. 
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: [https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.shvalprep.sh](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh)
	- Only the validation set is needed in the evaluation process.
	
### 2. Train our models

**The Training Module is simplified from the repo: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)**

- Modify the `run_example.sh`: change data path and hyper-params  according to your requirements
-  `./run_example.sh`

### 3. Evaluate our models

- You can evaluate our models with the following command:\
    ```python validate.py PATH/TO/ImageNet/validation --model DNA_a --checkpoint PATH/TO/model.pth.tar```
    - ```PATH/TO/ImageNet/validation``` should be replaced by your validation data path.
    - ```--model``` : ```DNA_a``` can be replaced by ```DNA_b```, ```DNA_c```, ```DNA_d``` for our different models.
    - ```--checkpoint``` : Suggest the path of your downloaded checkpoint here.
	
## TODO
Training and Searching code will be released in the future.
