# mnist_train
A simple Image Classification using lightning pytorch.

## install requirements:
```bash
pip install --upgrade pip | pip isntall -r requirements.txt

```
## Train:
To train this model, just execute:
```bash
python train.py 
```
If you want to change the size of the batches (default=32) or the num_workers (default=0) of the torch dataloader execute:
```bash
python train.py --batch_size <batch_size_value> --num_workers <num_workers_value>
```
## Test:
After the train, you can test the model as:
```bash
python test.py --checkpoint <checkpoint_path>
```
Equal as the train, you can change the num_workers value and the batch_size value for the test using

```bash
python test.py --checkpoint <checkpoint_path> --batch_size <batch_size_value> --num_workers <num_workers_value>
```
The results will be saved in the [Log](./log/) folder.

