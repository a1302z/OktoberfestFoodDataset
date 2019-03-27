# Description of Code
## Data
Following files are used for data creation, conversion, inspection and labeling
1. [Correlation](Correlation.ipynb) \
Creates links between checkout data and image data
2. [DatasetCreator](DatasetCreator.py), [ExtractThumbnails](ExtractThumbnails.py) and [CreateSmallerDataset](CreateSmallerDataset.py) \
Scripts that we used to create our datasets
3. [AnnotationTool](AnnotationToolBB.py), [CountAnnotations](CountAnnotations.py) and [ShowAnnotations](ShowAnnotations.py) \
Tools we used to annotate data
4. [Create-tfrecord-file](Create-tfrecord-file.ipynb), [CreateEvaluationData](CreateEvaluationData.ipynb) \
Convert annotated data so it can be used with the tensorflow object detection api or with our [new_test](new_test.py) script
5. [create-tfrecord-for-OID](create-tfrecord-for-OID.ipynb) \
Convert the Open Image Dataset V4 to the tfrecord format
## Evaluation
1. [forward_pass](forward_pass.py) \
Applies model to given data
2. [new_test](new_test.py), [eval_all](eval_all.py) \
Evaluates one given respectively all models in a folder
3. [Evaluation](Evaluation.ipynb) and [plot_class_evaluation](plot_class_evaluation2.ipynb) \
Show more enhanced evaluations in notebook
4. [SpeedMeasurement](SpeedMeasurement.ipynb) \
Evaluate inference time of models
## Further approaches
1. [Visualization by Annotation of Videos](VideoClassification.ipynb) \
To get better insights to performance of model we can apply a model to video data
2. [PseudoLabeling](PseudoLabeling.ipynb) and [LargePseudoLabeling](LargePseudoLabeling.py) \
As a simple semi-supervised learning approach we classified data with a trained model and thereby gained new annotated data. In the notebook first tests were performed as well as created data verifed. However it does not work for larger amount of data as it does assume infinite RAM. This problem was solved in LargePseudoLabeling which is classifying data chunk wise. 
3. [HPO](HPO.py) \
A script we used to do hyperparameter optimization automatically on multiple GPU's at the same time
4. [nms_utils](nms_utils.py) \
Helper functions to do non-maximum suppression