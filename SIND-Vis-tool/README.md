# SIND Dataset Python Tools


We provide a python program to import and visualize the sind data set to make it easy to observe!  

## Installation
The package required for the program to run is shown in the `requirements.txt` file, and you can install them by：`pip3 install -r requirements.txt`  
This Track Visualizer is tested with Python 3.6 and 3.8 but is very probably compatible with newer or slightly older releases.  
In addition to this, we recommend installing the [lanelet2](https://github.com/fzi-forschungszentrum-informatik/Lanelet2) module, which provides a convenient HD map API, and allows better visualization and ease of use of map information.  

## Usage
* copy/download the SIND dataset into the right place
  * copy/download the track files into the folder `data`, keep one folder per record, as in your download
  * your folder structure should look like in [File-Directory.md](https://github.com/SOTIF-AVLab/SinD/blob/main/doc/File-Directory.md)
* to visualize the data
  * run `./VisMain.py <data_path (default= ../Data)> <record_name (default= 8_02_1)>` from this folder directory to visualize the recorded data. 

## Module Description
### `DataReader.py`
This module allows to read either the tracks, static track info, traffic light states and recording meta info by respective function, and by calling `read_tracks_all(path)` to read a total recording info. 

### `VisMain.py`
This module uses the `intersection_visualizer.py` to create a gui to playback the provided recordings. In the visualization, traffic participants (vehicles and pedestrians) are presented as rectangular boxes. By clicking a rectangular box with the mouse, multiple graphs showing the changes of the parameters corresponding to the motion state of the traffic participants can pop up. 
The script has many different parameters, which are listed and explained in the `VisMain.py` file itself. The most 
important parameter is the `recording_name`, which specifies the recording to be loaded. 

<div align=center>
<img src="https://github.com/SOTIF-AVLab/SinD/blob/main/doc/Visualization.jpg" width =800><img src="https://github.com/SOTIF-AVLab/SinD/blob/main/doc/motion-parameters.jpg" width = 800>  
</div>  

