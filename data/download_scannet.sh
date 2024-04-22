#!/bin/bash


# Image pairs from ScanNet. Evaluation protocol from SuperGlue paper (Sarlin et al.)
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500_spsg.h5
wget -N http://vision.maths.lth.se/viktor/posebench/relative/scannet1500-images.zip

mkdir scannet1500-images
unzip scannet1500-images.zip -d scannet1500-images
rm scannet1500-images.zip
