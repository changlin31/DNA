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

### 2. Searching
The code for supernet training, evaluation and searching is under `searching` directory.
- `cd searching`
#### i) Train & evaluate the block-wise supernet with knowledge distillation
- Modify datadir in `initialize/data.yaml` to your ImageNet path.
- Modify nproc_per_node in `dist_train.sh` to suit your GPU number. The default batch size is 64 for 8 GPUs, you can change batch size and learning rate in `initialize/train_pipeline.yaml`
- By default, the supernet will be trained sequentially from stage 1 to stage 6 and evaluate after each stage. This will take about 2 days on 8 GPUs with EfficientNet B7 being the teacher. Resuming from checkpoints is supported. You can also change `start_stage` in  `initialize/train_pipeline.yaml` to force start from a intermediate stage without loading checkpoint.
- `sh dist_train.sh`
#### ii) Search for the best architecture under constraint.
Our traversal search can handle a search space with 6 ops in each layer, 6 layers in each stage, 6 stages in total. A search process like this should finish in half an hour with a single cpu. To perform search over a larger search space, you can manually divide the search space or use other search algorithms such as Evolution Algorithms to process our evaluated architecture potential files.

- Copy the path to architecture potential files generated in step i) to `potential_yaml` in `process_potential.py`. Modify the constraint in `process_potential.py`.
- `python process_potential.py`

	
### 3. Retraining

**The retraining code is simplified from the repo: [pytorch-image-models](https://github.com/rwightman/pytorch-image-models)** and is under `retraining` directory.

- `cd retraining`

- Retrain our models or your searched models
    - Modify the `run_example.sh`: change data path and hyper-params according to your requirements
    - Add your searched model architecture to `model.py`. You can also use our searched and predefined DNA models.
    - `sh run_example.sh`

- You can evaluate our models with the following command:\
    ```python validate.py PATH/TO/ImageNet/validation --model DNA_a --checkpoint PATH/TO/model.pth.tar```
    - ```PATH/TO/ImageNet/validation``` should be replaced by your validation data path.
    - ```--model``` : ```DNA_a``` can be replaced by ```DNA_b```, ```DNA_c```, ```DNA_d``` for our different models.
    - ```--checkpoint``` : Suggest the path of your downloaded checkpoint here.
