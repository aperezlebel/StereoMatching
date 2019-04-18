# Disparity calculation using Loopy Belief Propagation

## Context
The purpose of this work is to compute the disparity between two images (from which can be recovered the depth of field of the objects in the image).

## Input
The input are two images of the same scene taken from two different points. Be careful, it is assumed that the two points differ only by an horizontal translation.
Here are the two left and right tsukuba images used for the test :

![alt text](input/imL.png?raw=true)![alt text](input/imR.png?raw=true)

Try with your own images by replacing the ```imgL.png``` and ```imgR.png``` files in the ```input``` folder.

## Compute disparity

To compute the disparity map run :
```
python main.py
```

Don't forget to adjust your parameters at the beginning of ```main.py```.

Find your disparity map in the ```output``` folder. Here is an example of disparity map :

![alt txt](output/disparity_10.png?raw=true)

## Resource

This work has been made following the assignment.pdf file, proposed by the computer vision course of the ENPC school. See more details in it.
