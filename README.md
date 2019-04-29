# Oktoberfest Food Dataset
The data was aquired during Schanzer Almfest at Ingolstadt in 2018 by [IlassAG](https://www.ilass.com). As a part of a practical at the [Data Mining and Analytics Chair by Prof. Günnemann at TUM](https://www.kdd.in.tum.de) we've been given the task to count objects at checkout. Therefore we annotated the data with bounding boxes and classes to train an object detection network.
![Annotated image](images/example_annotated.png)

## Dataset Description
TODO
- usage
- wie man es herunterladen kann
- den datensatz irgendwo hochladen

### Data Distribution

Class | Images | Annotations | average quantity
 --- | --- | --- | ---
Bier | 300 | 436 | 1.45 
Bier Mass | 200 | 299 | 1.50 
Weissbier | 229 | 298 | 1.30 
Cola | 165 | 210 | 1.27 
Wasser | 198 | 284 | 1.43 
Curry-Wurst | 120 | 159 | 1.32 
Weisswein | 81 | 105 | 1.30 
A-Schorle | 90 | 98 | 1.09 
Jaegermeister | 43 | 152 | 3.53 
Pommes | 110 | 126 | 1.15 
Burger | 105 | 122 | 1.16 
Williamsbirne | 50 | 121 | 2.42 
Alm-Breze | 100 | 114 | 1.14 
Brotzeitkorb | 65 | 72 | 1.11 
Kaesespaetzle | 92 | 100 | 1.09 
Total | 1110 | 2696 | 2.43

![Images per class](images/images_per_class.png)
![Annotations per class](images/annotations_per_class.png)
![Items per image](images/items_per_image.png)
![Occurance heat map](images/Occurance_heatmap.png)

## Benchmark
For training object detection models we've been using [tensorflow's Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). We trained several different approaches and got best results for an Single Shot Detector (SSD) with Feature Pyramid Networks (FPN). Our evaluation metric was the area under the precision-recall curve on a test set of 86 images (as our goal was to count we ignored the localization). 

Approach | Backbone model | Area | Example precision@recall
 --- | --- | --- | ---
SSD | Mobilenet | 0.86 | 0.85@0.70
SSD + FPN | Mobilenet | 0.98 | 0.97@0.97
Faster RCNN (PyTorch Version) | VGG-16 | 0.95 | 0.90@0.92
RFCN | ResNet-101 | 0.965 | 0.90@0.95

TODO
- die pretrained models irgendwo hochladen

## Code
The [Evaluation](evaluation) folder contains Jupyter notebooks to evaluate the TensorFlow models.

With the [Preview](Preview.ipynb) notebook one can try out the pretrained TensorFlow models on arbitrary images.

The [CreateTFRecordFile](CreateTFRecordFile.ipynb) notebook contains code to convert the dataset in to the TFRecord file format so it can be used with the TensorFlow object detection library.

The [ShowAnnotations](ShowAnnotations.py) visualizes the bounding boxes of the dataset. Use 'n' for the next image, 'p' for the previous and 'q' to quit. 

## Authors
[Alexander Ziller](https://github.com/a1302z): Student of Robotics, Cognition & Intelligence (M.Sc.) at TUM \
[Julius Hansjakob](https://github.com/polarbart): Student of Informatics (M.Sc.) at TUM \
Vitalii Rusinov: Student of Informatics (M.Sc.) at TUM 

We also want to credit [Daniel Zügner](https://github.com/danielzuegner) for advising us any time and encouraging to publish this dataset. 
