# AugmentedReality

This is an implementation of an Augmented Reality application wherein we have placed several virtual object models in an existing scene as if they already exist in the real world.

In the project, I recovered the camera poses with two different approaches:
1. Perspective-N-Point (PnP) problem with coplanar assumption
2. Persepective-three-point (P3P) and the Procrustes problem

After retrieving the 3D relationship between the camera and world, arbitrary objects were placed in the scene.

### Input Scene ###
![Input scene](https://github.com/ayushgoel24/AugmentedReality/blob/master/input_image.jpg?raw=true)

### Output Scene ###
![Output scene](https://github.com/ayushgoel24/AugmentedReality/blob/master/output_scene.gif)
