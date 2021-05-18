# Project 4 #

## Introduction 
  	This file includes python program solution for Project 4 for
	UMD ENPM 673 Spring 2021 batch. The folders consists of python
	program along with some gedata set to test.
	
	These files are executable:
    Part1.py
    Part2.py
		CNN_FISH.ipynb

  
## Requirements
       ***Important*****************************************************
       *To Test it on your data set, update images into data set folder*
       *****************************************************************
       
### To run this code following libraries are required
* Tensor Flow
* NumPy
* Pathlib
* SkLearn
* Pandas
* Keras
* Seaborn
* OpenCV

### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````
* NumPy
	````
	pip install numpy
	````
Except that, we have to use Google Colab or Kaggle to run the second problem and these cloud IDEs come with all the libraries instaleed.
	
### Running code in ubuntu
After changing the path of the video source file and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command
* Run the following command which will execute the Optical flow vector plot
````
python Part1.py
````
* Run the following command which will execute the code to get only moving objects
````
python Part2.py
````
* now, if you want run the CNN code, you will have to upload the data set into respective cloud based IDE and run upload the following notebook
````
CNN_FISH.ipynb
````
It is important to note that if both python files are in /inputs/fishdataset/FishDataset/FishDataset path
you have to change to the correct directory if it is not true.


### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the vdata set
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
  
### Maintainers ###
	Mrugesh Shah (mrugesh.shah92@gmail.com)

