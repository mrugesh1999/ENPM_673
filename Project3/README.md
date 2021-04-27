# Project 2 #

## Introduction 
  	This file includes python program solution for Project 3 for
	UMD ENPM 673 Spring 2021 batch. The folders consists of python
	program along with some gedata set to test.
	
	These files are executable:
		main.py

  
## Requirements
       ***Important*****************************************************
       *To rest it on your data set, update images into data set folder*
       *****************************************************************
       
### To run this code following libraries are required
* OpenCV  
* NumPy
* SciPy
* time
* PIL

### Installation (For ubuntu 18.04) ###
* OpenCV
	````
	sudo apt install python3-opencv
	````
* NumPy
	````
	pip install numpy
	````
* SciPy
	````
 	pip install scipy
  ````
* PIL
  ````
  python3 -m pip install --upgrade Pillow
  ````
	
### Running code in ubuntu
After changing the path of the video source file and installing dependencies
Make sure that current working derectory is same as the directory of program
You can change the working derectory by using **cd** command
* Run the following command which will execute the code
````
python main.py
````
* Select the data set by entering the digit using number pad
````

Enter the dataset you want to execute (enter 1, 2 or 3)):
````
* now, if you want to check the disparity as well, press 1, Note that it will take long time to be executed
````
Do you want to get disparity and depth image (1:Yes, 2:No)?
````
It is important to note that if both python files are in different directory
we have to change to the correct directory if it is not true.


### Troubleshooting ###
	Most of the cases the issue will be incorrect file path.
	Double check the path by opening the properies of the vdata set
	and copying path directly from there.

	For issues that you may encounter create an issue on GitHub.
  
### Maintainers ###
	Mrugesh Shah (mrugesh.shah92@gmail.com)

