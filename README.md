# Frontend-UI-Element-Detection-and-Classification
## Detection and Classification of UI Elements of Web pages and Apps from Wireframe Sketches


### The file and folder structure is given below:
<p>
Tensorflow Object Detection 
|
V
|-----Detection TF2.ipynb --> For detection

|-----Trainer TF2.ipynb --> For training

|-----addons --> LabelImg ->> Github Link to LabelImg 

|-----models ->> github TFOD master

|-----scripts

	|-----exporter_main_v2.py

	|-----generate_tfrecord.py

	|-----model_main_tf2.py
|-----workspace

	|-----training

		|-----annotations --> Contains labelmap and TF record files

		|-----exported-models --> Exported model after training

		|-----images --> Folders named train, test, eval and detected-images for their specific images. Each folder will have image and its corresponding PASCAL VOC formated .xml files.

		|-----pre-trained-models  --> Each folder containing downloaded pre-trained models ->> Download from TFOD Model Zoo

		|-----under-training-models --> Each folder containing customized pipeline config for the pretrained models and its checkpoints (Copy over the pipeline.config 	  file with necessary changes from either exported-models or pre-trained-models. See [Configure the Training Pipeline](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline))
</p>



In the anaconda prompt under the specific conda environment created for this project using requirements.txt: <a src='https://github.com/Somoy73/Frontend-UI-Element-Detection-and-Classification/blob/master/requirements.txt'>requirements</a>

For creating the train data TF record:
```
python scripts/generate_tfrecord.py -x ./workspace/training/images/train -l ./workspace/training/annotations/label_map.pbtxt -o ./workspace/training/annotations/train.record
```

For creating the test data TF record:
```
python scripts/generate_tfrecord.py -x ./workspace/training/images/test -l ./workspace/training/annotations/label_map.pbtxt -o ./workspace/training/annotations/test.record
```

For training the model:
```
python scripts/model_main_tf2.py --model_dir=./workspace/training/under-training-models/faster_rcnn --pipeline_config_path=./workspace/training/under-training-models/faster_rcnn/pipeline.config
```

For evaluating the model as it is getting trained, run the below command in a duplicated command prompt:
```
python model_main_tf2.py --model_dir=./workspace/training/under-training-models/faster_rcnn --pipeline_config_path=./workspace/training/under-training-models/faster_rcnn/pipeline.config --checkpoint_dir=./workspace/training/under-training-models/faster_rcnn
```

For exporting the model:<br/>
```
python scripts\exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./workspace/training/under-training-models/faster_rcnn/pipeline.config --trained_checkpoint_dir ./workspace/training/under-training-models/faster_rcnn/ --output_directory ./workspace/training/exported-models
```
Models must to exported or frozen for restarting training from where we left off. To use exported model, change the fine_tune_checkpoint directory to refer to the ckpt-0 of exported model.


For monitoring the training and/or evaluation job progress using TensorBoard during/after the process, run the below command in a duplicated command prompt: <br/>
```
tensorboard --logdir=./workspace/training/under-training-models/faster_rcnn
```


Reference:
For more detailed procedure explanation: [Training Custom Object Detector using Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)
