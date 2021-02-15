# Homework 1 #

## Introduction 
  	This file includes python program solution for Homework 1 for
	UMD ENPM 673 Spring 2021 batch. The folders consists of python
	program along with some generated outputs as .png image. 
	
	These files are executable:
		Part_1.py
		Part_2.py
		Homography.py

	All other files are either pictures of generated output or the
	the media files.
## Requirements
       ***Important****************************************************************
       *Change the path of the video in cv2.VideoCapture() function in python file*
       ****************************************************************************
### To run this code following libraries are required
* OpenCV, 
* random, 
* matplotlib, 
* NumPy, 
* time.

### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````
* matplotlib
	````
	python -m pip install -U matplotlib
	````
* NumPy
	````
	pip install numpy
	````
	
### Running code in ubuntu
After changing the path of the video source file and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command
* Run the following command which will generate curve plot for video 1
````
python Part_1.py
````
* Run the following command which will generate curve plot for video 2
````
python Part_2.py
````
It is important to note that if both python files are in different directory
we have to change to the correct directory again.
* Run the following command to get the solution of problem 3
* This will generate homography matrix in the terminal
````
python Homography.py
````

### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the video
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
### Maintainers ###
	Mrugesh Shah (mrugesh.shah92@gmail.com)
