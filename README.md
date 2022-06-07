# EfficientnetB0
Building an image classifier model using Efficientnet architechture.

Requirments:
1.Keras and tensorflow API --> used to build model architechture.
2.matplotlib --> used for visualization.
3.Tensorflow_datasets --> used for built-in datasets.
**from Tensorflow. keras.applications we import EfficientnetB0 arcitecture.

This model is used to predict data with a higher accuracy rate on any given dataset.
Efficientnet architecture had diffrent versions stating from B0 to B7
we are training a efficientnetBO architecture for our model
Builtin dataset is used in the architecture to tarin the model.
"standard_dogs" is a build-in dataset taken from " tensorflow_datasets" library which is used in the model training.
"standard_dogs dataset have 120 breeds of dogs from around the world. There are 20,580 images, out of which 12,000 are used for training and 8580for testing.
this dataset was loaded into model by using TFDS.load method.
we used both sequential and functional API inthis model.
Sequential layer is used in data augmentation where are functional APi is used in building the model.
we have used Adam optimizers and catagoricl cross entropy as loss function in model compilation.
we have taken the batch size of 64 in the model
later we call model.fit to fit the model and start training the model
