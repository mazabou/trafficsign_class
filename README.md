# Traffic signs classification

This repository use multiple predefine model in keras, and try to fine tune them
on a traffic sign classification task.

## Requirement

 - Keras
 - numpy
 - matplotlib
 
## dataset

This script expect a dataset with the following structure:

    Dataset
    ├── Class1
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Class2
    │   ├── image1.jpg
    │   └── image2.jpg
    └── Class3
        ├── image1.jpg
        └── image2.jpg

## Usage

    python3 train.py <super-class-to-use> <dataset-path>
    
Where `super-class-to-use` is one of the first level key of `classes` dict in 
[train.py](train.py).

A lot of different parameters can be tuned on the command line call, to have a
complete description of them, please run:

    python3 train.py -h

The predefined super classes of `classes` dict are MUTCD code of US signs for super classes
`Rectangular`, `Diamond` and `Zebra`. For `RedRoundSign` it is the class names as used in
the TT100k traffic sign dataset.


## Future work

An other project, based on Siamese network, showed very promising results to tackle the 
problem of sign classification (low data sample for some class, lot of classes).
This project code is available at 
[https://github.com/DL-project-Fall2019/Siamese-traffic-signs](https://github.com/DL-project-Fall2019/Siamese-traffic-signs)

## Author

Nicolas Six



