# pydrf
Python scripts for estimating dose response functions and performing surrogate analysis based on temporal network data

Provides code for reproducing the analysis reported in:

Donges*, J. F./Lochner*, J. H., Kitzmann, N. H., Heitzig, J., Lehmann, S., Wiedermann, M., & Vollmer, J. (2021). Dose-response functions and surrogate models for exploring social contagion in the Copenhagen Networks Study. preprint arXiv:2103.09496.
\* shared lead authorship.

Please see the above paper for a detailed mathematical description of the model (Methods section) and references to the relevant scientific literature.

Software DOI of original release along with publication:

<doi>

## Code Structure
Two types of temporal network data were analysed, (1) the data from the Copenhagen Networks Study (CNS) and (2) the data from the adaptive voter model (AVM). The code which is independent of the datatype is summarised in the package *world_viewer* (for details, see world_viewer/README.md). The belonging of the remaining code is indicated by the prefix CNS or AVM.

## Run Analysis

### Prepare Copenhagen Data
Data originating from: 
Sapiezynski, P., Stopczynski, A., Lassen, D. D. & Lehmann, S. Interaction data from the Copenhagen Networks Study. Sci. Data 6, 1–10 (2019).

1) Calculate contacts using the script: PreprocessRelations/BluetoothToEdge.ipynb
2) Calculate traits using the script: PreprocessOpinions/FitnessAsBehavior.ipynb

The following files will be created: 
* data/relations.pkl
* data/op_fitness.pkl

**Cumulative Distribution of the Past-Week Behavioural Function**
- Plot the cumulative distribution using the notebook: CNS_CDF.ipynb

**Sensitivity Analyses of Parameters**
- Analysis of characteristic time: CNS_Sensitivity_Characteristic_Time.ipynb
- Analysis of k^tilde_min: CNS_Sensitivity_k_tilde_min.ipynb

### Prepare Synthetic Data from Adaptive Voter Model

1) Generate ten model runs using the script: data/Synthetisch/avm_final_5k/run.sh
2) Calculate exposure for all runs using the script: run_avm_model.sh

### Generate Dose Response Functions 
Use the following jupyter notebook to generate the DRFs.
- CNS-Data: CNS_Dose_Response_Function.ipynb
- AVM-Data: AVM_Dose_Response_Function.ipynb 

### Surrogate Data Test

#### Generate the Surrogates
Use the python scripts in the folder *CreateSurrogates/* to generate the surrogates. The prefix indicates if the script belongs to the CNS-data or the AVM-data. 
For each null hypothesis there is one surrogate model resp. one script:

1. Null Hypothesis: CNS_surrogate_time-trait-complete.py
2. Null Hypothesis: CNS_surrogate_time-trait-complete_per_node.py
3. Null Hypothesis: CNS_surrogate_time-trait-complete_per_node_conserve_switches.py
4. Null Hypothesis: CNS_surrogate_trait.py
5. Null Hypothesis: CNS_surrogate_time-edges-complete.py
6. Null Hypothesis: CNS_surrogate_time-edges-complete_per_node_cons_indiv_degree.py

#### Generate Surrogate DRFs

To generate the DRFs for the six surrogates, run the following scripts:
- AVM_make_surrogate_drfs_1+2+5.py
- AVM_make_surrogate_drfs_3.py
- AVM_make_surrogate_drfs_4.py
- AVM_make_surrogate_drfs_6.py

For the CNS data use the corresponding scripts with the prefix "CNS".

#### Plotting the Surrogates

**AVM**
1. To compare the surrogate DRF with the AVM DRF, calculate the AVM DRF using: make_avm_spring_data.py
2. Plot the DRFs using: AVM_Surrogate_Plots.ipynb

**CNS**
- Plot the DRFs using: CNS_Surrogate_Plots.ipynb

