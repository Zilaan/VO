#!/bin/bash

cd ~/Documents/Programmering/opencv/VO/odometry/results/

rm -rf *

cp -rf ~/Documents/Programmering/opencv/VO/odometry/res2_800/$2/data ~/Documents/Programmering/opencv/VO/odometry/results/

cd ~/Downloads/devkit/cpp
./eval_odometry $1 `cat ~/Documents/Programmering/opencv/VO/odometry/results/data/*.txt | wc -l`

rm -rf ~/Documents/Programmering/opencv/VO/odometry/res2_800/$2/*/
#find ~/Documents/Programmering/opencv/VO/odometry/res2_800/$2 -type d -exec rm -rf '{}' \;

cp -rf ~/Documents/Programmering/opencv/VO/odometry/results/* ~/Documents/Programmering/opencv/VO/odometry/res2_800/$2/
