In the zip file one can find the labels for the image and video datasets in the PASCAL_VOC format.

The complete dataset (images and labels) is available at

https://drive.google.com/open?id=1rgJUEFB4Cmbf9mQVdGPCHGiT4bvh_gDT (images)

https://drive.google.com/open?id=1mLIc1Ybs1rVwzMDuWMwxWUgl7spx2tBB (video)

If the links above are unavailable, one can recreate the dataset in the following way.

Download the images from https://mediatum.ub.tum.de/1487154

Create folder JPEGImages.

In the folders 'large' and 'small' the images should be of size 1200x675 and 1000x562, respectively.

In order to recreate the video dataset, one should extract frames corresponding to the annotation files from the provided video files. 

The annotation filename consists of the timestamp and the frame number separated by an underscore (timestamp_framenum).

First, the saturation of the video frames must be normalized. Then one should resize them to size 1000x562 pixels and place them into the folder JPEGImages.
