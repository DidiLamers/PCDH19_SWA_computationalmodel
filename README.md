# PCDH19_SWA_computationalmodel

## General Information  
Computational thalamocortical model generating SWA. Created at the OIST Computational Neuroscience Course 2018 with a lot of help from my tutor: 
Dr. Espen Hagen.



## Use in PCDH19 project
This model was developed as part of my PhD thesis on PCDH19 with the aim to model SWA, and test how PCDH19 mosaicism would disturb SWA.

## Description of the model

The model was created using the PyNEST simulator (Gewaltig et al. 2007). 

The model was incorporated exactly as described by Destexhe 2009. Briefly, it employs adaptive exponential IF neurons to create a cortical layer 
with regular spiking pyramidal cells (1600 cells) and fast spiking interneurons (400 cells) as well as a thalamic layer with thalamocortical neurons (100 
cells) and inhibitory thalamic reticular neurons (100 cells). I created a mosaic in this model by assigning all cortical cells to one of two groups: a
"PCDH19 positive" and "PCDH19 negative" group. Connectivity between but not within these groups was then reduced and effects on SWA monitored.

To compare output of this model with LFP data obtained experimentally, I simulated the LFP based on an adaptation of Mazzoni et al. 2015.

## Contributors
I have had a lot of help in developing this model from my tutor at the OCNC2018: Dr. Espen Hagen. 

## Dependencies
The numpy and matplotlib packages. 