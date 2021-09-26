# Frontend-UI-Element-Detection-and-Classification
## Detection and Classification of UI Elements of Web pages and Apps from Wireframe Sketches


### The file and folder structure is given below:

```
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
 
 		          |-----images --> Folders named train, test, eval and 
                               detected-images for their specific images. 
                               Each folder will have image and its corresponding
                               PASCAL VOC formated .xml files.
 
 		          |-----pre-trained-models  --> Each folder containing downloaded pre-trained models ->> Download from TFOD Model Zoo
 
 		          |-----under-training-models --> Each folder containing customized pipeline config 
                                              for the pretrained models and its checkpoints
                                              (Copy over the pipeline.config file with necessary 
                                              changes from either exported-models or pre-trained-models.)

```

**Note**:

---

Some folders may be missing due to upload limitations. So, create them if they don't exist using the same folder names given above. For instance, 
under-training-models/faster_rcnn folder is missing. So create them, then copy over the pipeline.config file from exported-models/10200 and make necessary changes. **This needs to be done if you are planning to train.** 


See [Configure the Training Pipeline](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline) for figuring out what changes need to be done in pipeline.config file.


## Commands

In the anaconda prompt under the specific conda environment created for this project using [requirements.txt](https://github.com/Somoy73/Frontend-UI-Element-Detection-and-Classification/blob/master/requirements.txt):

**For creating the train data TF record**:
```
python scripts/generate_tfrecord.py -x ./workspace/training/images/train -l ./workspace/training/annotations/label_map.pbtxt -o ./workspace/training/annotations/train.record
```

**For creating the test data TF record**:
```
python scripts/generate_tfrecord.py -x ./workspace/training/images/test -l ./workspace/training/annotations/label_map.pbtxt -o ./workspace/training/annotations/test.record
```

**For training the model**:
```
python scripts/model_main_tf2.py --model_dir=./workspace/training/under-training-models/faster_rcnn --pipeline_config_path=./workspace/training/under-training-models/faster_rcnn/pipeline.config
```

**For evaluating the model as it is getting trained, run the below command in a duplicated command prompt**:
```
python model_main_tf2.py --model_dir=./workspace/training/under-training-models/faster_rcnn --pipeline_config_path=./workspace/training/under-training-models/faster_rcnn/pipeline.config --checkpoint_dir=./workspace/training/under-training-models/faster_rcnn
```

**For exporting the model**:<br/>
```
python scripts\exporter_main_v2.py --input_type image_tensor --pipeline_config_path ./workspace/training/under-training-models/faster_rcnn/pipeline.config --trained_checkpoint_dir ./workspace/training/under-training-models/faster_rcnn/ --output_directory ./workspace/training/exported-models
```
**Models must to exported or frozen for restarting training from where we left off. To use exported model for training, change the fine_tune_checkpoint directory to refer to the ckpt-0 of exported model.**


**For monitoring the training and/or evaluation job progress using TensorBoard during/after the process, run the below command in a duplicated command prompt**: <br/>
```
tensorboard --logdir=./workspace/training/under-training-models/faster_rcnn
```


### An example of an inferenced image
![Inferenced Image](https://github.com/Somoy73/Frontend-UI-Element-Detection-and-Classification/blob/master/workspace/training/images/detected_images/i292.jpg)

## Reference:

For more detailed procedure explanation: [Training Custom Object Detector using Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html)

Tensorflow 2 Object Detection: [Tensorflow Model Garden](https://github.com/tensorflow/models)

Tensorflow 2 Model Zoo : [Go there](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

Faster RCNN pre-trained Model: [Download from Tensorflow 2 Model Zoo](http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_inception_resnet_v2_640x640_coco17_tpu-8.tar.gz)

[LabelImg](https://github.com/tzutalin/labelImg): [Download LabelImg](https://github.com/tzutalin/labelImg/files/2638199/windows_v1.8.1.zip)
