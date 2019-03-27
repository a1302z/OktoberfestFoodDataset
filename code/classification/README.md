At the beginning of our project we had the problem that the object detection model didn't recognize Weißbier.
We wanted to find out if this had to do something with our object detection model or if there was a general error with our data.

Therefore we trained a simple classification model on our data. This folder contains the code for this. The classification works almost perfectly (see https://wiki.tum.de/display/mllab/Classification).

Shortly afterwards we found out, that the problem was the umlaut in the name. 

Note: Vitaliy found the issue with dataset export script to everyone's surprise comparing two exported datasets in different formats, credit was given at the presentation. The classification script didn't help to do it (at all). 
Yet it was really useful to know we can distinguish Bier from Bier Mass.