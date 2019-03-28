# Oktoberfest Food Dataset
## Description
The data was aquired during Schanzer Almfest at Ingolstadt in 2018 by [IlassAG](https://www.ilass.com). As a part of a practical at the [Data Mining and Analytics Chair by Prof. Günnemann at TUM](https://www.kdd.in.tum.de) we've been given the task to count objects at checkout. Therefore we annotated the data with bounding boxes and classes to train an object detection network.
![Annotated image](images/example_annotated.png)

### Data distribution

Number of labels | Class
 --- | ---
458 | Bier
317 | Bier Mass
308 | Weissbier
221 | Cola
300 | Wasser
179 | Curry-Wurst
119 | Weisswein
101 | A-Schorle
154 | Jaegermeister
149 | Pommes
144 | Burger
129 | Williamsbirne
126 | Alm-Breze
74 | Brotzeitkorb
116 | Kaesespaetzle

![Distribution](images/stats.png)

## Benchmark
For training object detection models we've been using [tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). We trained several different approaches and got best results for an Single Shot Detector (SSD) with Feature Pyramid Networks (FPN). Our evaluation metric was the AUC score (as our goal was to count we used IoU=0) on a test set of 86 images. 

Approach | Backbone model | AUC score 	
 --- | --- | --- 
SSD | Mobilenet | 0.86
SSD + FPN | Mobilenet | 0.98
Faster RCNN (PyTorch Version) | VGG-16 | 0.95
RFCN | ResNet-101 | 0.965

## Code
We offer several pieces of code, for data inspection, conversion and labeling as well as evaluation and further usages which are very specific to our approaches. Please see the [code directory](code) for usage. 

## Authors
[Alexander Ziller](https://github.com/a1302z): Student of Robotics, Cognition & Intelligence (M.Sc.) at TUM \
Julius Hansjakob: Student of Informatics (M.Sc.) at TUM \
Vitalii Rusinov: Student of Informatics (M.Sc.) at TUM 

We also want to credit [Daniel Zügner](https://github.com/danielzuegner) for advising us any time and encouraging to publish this dataset. 

<!---
## Task
Throughout this practical we aimed to implement, test and evaluate approaches to solve the problem of counting food items at checkout. You'll find code we used in this repository. However, note that it is not straight forward to use as data and models are due to their storage size not apparent. 
## Structure
This repository is divided to several subfolder to maintain structure. 
1. [submission_julius_alex](submission_julius_alex) \
This folder contains most important scripts @ga78yah and @ga78veb created, including Annotation, Evaluation and further approaches. 
2. [scripts_from_tfrepo](scripts_from_tfrepo) \
This contains files from the [repository](https://github.com/tensorflow/models/tree/master/research/object_detection) we used for training models.
3. [miscellaneous](miscellaneous) \
Here is very different code that is not necessary or outdated, as well as some data examples.
4. [configs](configs) \
This folder contains configs that were used for training.
--->