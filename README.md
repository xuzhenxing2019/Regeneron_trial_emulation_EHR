Introduction

This is the public code of the submitted manuscript "Systematic Evaluation of Supportive Care Drugs in Cancer Using High-Throughput Target Trial Emulation of Real-World Data". 



0. Create Environment and Install Libs:
   
  pip install -r requirements.txt


2. Run the Experiments (From data preprocessing to data analysis):
   ---01_determine_therapy_pres_adm_pro.py
   ---02_cohort_construction.py
   ---03_extract_drug_baseline_all.py
   ---04_concat_covariate.py
   ---05_extract_dod.py
   ---06_create_dataframe_drug.py
   ---07_emulate_PSM_sae.py
   ---08_analyze_PSM_dod_sae.py
   ---09_generate_cohort_characteristics.py 
