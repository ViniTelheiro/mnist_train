# mnist_train
A simple Image Classification using lightning pytorch.

## install requirements:
You can install the requirements using the requirements.txt, as above:
```bash
pip install --upgrade pip | pip isntall -r requirements.txt

```
or using the uv:
```bash
uv sync 
```
## Train:
To train this model, just execute:
```bash
python train.py 
```
To change the training parameters, access the [config.yaml](./config.yaml) file
## Test:
After the train, you can test the model as:
```bash
python test.py --checkpoint <checkpoint_path>
```
Equal as the train, you can change the num_workers in the [config.yaml](./config.yaml) file


The results will be saved in the [Log](./log/) folder.

