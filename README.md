# SIND
## Introduction
A key challenging scenario for autonomous driving is intersections, but there are currently no large-scale public trajectory datasets on signalized intersections. Motivated by this, we constructed the SIND dataset, which was collected at a signalized intersection in Tianjin, China. The SIND dataset is based on 4K video captured by drones, providing information including traffic participant trajectories, traffic light status, and high-definition maps.  A demo video of the dataset can be viewed on [Youtube](https://youtu.be/H9QSGqioYww) or [BiliBili](https://www.bilibili.com/video/BV1wN4y1F7Sc?share_source=copy_web&vd_source=07ce0c26f3e74d7ba5041a25df0072be)

<div align=center>
<img src="doc/SIND.jpg" width = 800>
</div>   

## Basic Statistics
SIND contains 7 hours of recording including over 13,000 traffic participants with 7 types,  HD maps and traffic light information are used to count traffic light violations by vehicles in them.  Clearly, SIND has a high proportion of vulnerable road users and frequent non-motor vehicle violations.  
<div align=center>
<img src="doc/Number and proportion of categories.png" width = 400><img src="doc/veh-traffic light violation.png" width = 400>  
</div>  

##  Access
You can get the project and a sample record by executing `git clone https://github.com/SOTIF-AVLab/SinD.git`. To access the full dataset， please contact us by e-mail:
hong_wang@tsinghua.edu.cn or 18975505069@163.com    
* The title of the email should be: __\[Apply for SinD\] name\_country(region)\_organization__    
Please describe your laboratory or department, research interest, and the purpose of the dataset in detail in the email content; After confirmation, we will send the complete data link as soon as possible(By default, a reply email containing a link will be sent to email address which you contact us). You are also welcome to further communicate with us

## Description of Format

SIND consists of 23 records, each of which contains 8-22 minutes of trajectories of traffic participants. In addition to the trajectories and motion state parameters of traffic participants, SIND also provides synchronized traffic light states and HD map. Each record contains the following <kbd>.csv</kbd> files:
For detailed format see [Format.md](Format.md#sdd)  

## Visualization  
For visualization see [SIND-Vis-tool](https://github.com/SOTIF-AVLab/SinD/tree/main/SIND-Vis-tool)

## Acknowledgements

Our visualization code is built upon the public code of the following papers:
* [ The ind dataset: A drone dataset of naturalistic road user trajectories at german intersections, IV'2020](https://github.com/ika-rwth-aachen/drone-dataset-tools)
* [Constructing a Highly Interactive Vehicle Motion Dataset, IROS'2019](https://github.com/interaction-dataset/interaction-dataset)

## Organization
<img src="doc/logo.png" width = 350>

- School of Vehicle and Mobility, Tsinghua University
- Tsinghua Intelligent Vehicle Design and Safety Research Institute
- Safety Of The Intended Functionality（SOTIF） Research Team
