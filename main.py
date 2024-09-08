# load dependencies
import os # operating system functionalities
from pathlib import Path # for handling paths conveniently
import scipy.io # load the data according to the instructions in the data sheet
import numpy as np # images are numpy.ndarrays
import matplotlib.pyplot as plt # for plotting the images


# define paths
# path_input_dir = Path("/kaggle/input/dataset-of-bmode-fatty-liver-ultrasound-images")
path_data =  "dataset_liver_bmodes_steatosis_assessment_IJCARS.mat"

data = scipy.io.loadmat(path_data)

data.keys()
# access the 'data' field
data_array = data['data']

# access specific fields within the data array
ids = data_array['id']
classes = data_array['class']
fats = data_array['fat']
images = data_array['images']

# get class or rather data type of images
type(images)

# access the first image, save to object and preview data structure
first_image = images[0][10][3]

# plot the image
plt.figure(figsize=(9,9))
plt.imshow(first_image, cmap='gray')  # Use 'gray' for grayscale images
plt.axis('off')  # Hide axes for better visualization
plt.show()
