## Intro
We propose the following computer vision framework for screw detection and related tool recommendation for end-of-life electronics disassembly: \
![framework](https://github.com/Xinyao-Zhang/Screw-Detection-and-Tool-Recommendation-for-Robotic-Disassembly/blob/528ab9892400b95a67f99d8ee1f7aa851b87a54b/framework.png)\
More details can be found in the article [Automatic screw detection and tool recommendation system for robotic disassembly](https://asmedigitalcollection.asme.org/manufacturingscience/article-abstract/145/3/031008/1148469/Automatic-Screw-Detection-and-Tool-Recommendation?redirectedFrom=fulltext).

## Dataset
The dataset includes three types of screws, Torx security screws on the desktop hard drive, Phillips screws on the back cover of a Dell laptop, and Pentalobe screws on the back cover of a Mac laptop. A total of 300 images are recorded at a resolution of 4000x6000 pixels and divided into an 80% training set and a 20% test set. The training dataset is then manually annotated by [LabelImg](https://github.com/qaprosoft/labelImg), a graphical image annotation tool. The rectangular bounding boxes only contain screws and are labeled as a single class named ‘0’, which are stored as .txt files in YOLO format. Specifically, the object coordinates are the x-y coordinates of the center of each bounding box relative to the width and height of each image.

## Citation
If you wish to cite the work, you may use the following:
```ruby
@article{10.1115/1.4056074,
    author = {Zhang, Xinyao and Eltouny, Kareem and Liang, Xiao and Behdad, Sara},
    title = "{Automatic Screw Detection and Tool Recommendation System for Robotic Disassembly}",
    journal = {Journal of Manufacturing Science and Engineering},
    volume = {145},
    number = {3},
    year = {2022},
    month = {12},
    issn = {1087-1357},
    doi = {10.1115/1.4056074},
    url = {https://doi.org/10.1115/1.4056074},
    note = {031008},
    eprint = {https://asmedigitalcollection.asme.org/manufacturingscience/article-pdf/145/3/031008/6953322/manu\_145\_3\_031008.pdf},
}
```
