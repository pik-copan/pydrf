# Package for analysing contagion dynamics in temporal social networks

This package still contains some code which was not used for the thesis. Most importantly are the files "cns_world.py" for loading the copenhagen data, "synthetic_world.py" for loading the models and "contagion_analysis.py" for calculating dose response functions.

## Classes
### World
The World-Class is a data wrapper for temporal social networks. Purpose of this class is to load the networks and bring them into a equal format.

Main Variables (defined in child-class):

* a_ij: adjency matrix
* op_nodes: traits of the node at each time point
* d_ij: like-mindedness (bool) (outdated)

Main Function (defined in child-class):
* load_world(...): loads all necessary data

#### Child-Classes "Synthetic World", "CNS World"
The child-classes "Synthetic World", "CNS World" are the specific classes for the different data sources:

* Synthetic World: For the different models like the adaptive voter model
* CNS World: For the Copenhagen Data

### Glasses

Collection of all tools in order to analyse the temporal social networks. The class itself is not ordered and contains a lot old code. More important: The code to analyse contagion dynamics is in the Child-Class "contagion analysis".

#### Child-Class "Contagion Analysis"

Class to calculate dose response functions.
1) Calculate the exposure using calc_exposure()
2) Calculate if nodes changed their trait after experiencing the exposure using opinion_change_per_exposure()
3) Plot the dose response funtion using plot_opinion_change_per_exposure_number()

Example application: CNS_Dose_Response_Function.ipynb

Main functions:
* run_contaigon_analysis()
* calc_exposure()
* opinion_change_per_exposure()
* plot_opinion_change_per_exposure_number()
