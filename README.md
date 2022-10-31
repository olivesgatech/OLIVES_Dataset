# OLIVES_Dataset

## Abstract
Clinical diagnosis of the eye is performed over multifarious data modalities including scalar clinical labels, vectorized biomarkers, two-dimensional fundus images, and three-dimensional Optical Coherence Tomography (OCT) scans. While the clinical labels, fundus images and OCT scans are instrumental measurements, the vectorized biomarkers are interpreted attributes from the other measurements. Clinical practitioners use all these data modalities for diagnosing and treating eye diseases like Diabetic Retinopathy (DR) or Diabetic Macular Edema (DME). Enabling usage of machine learning algorithms within the ophthalmic medical domain requires research into the relationships and interactions between these relevant data modalities. Existing datasets are limited in that: ($i$) they view the problem as disease prediction without assessing biomarkers, and ($ii$) they do not consider the explicit relationship among all four data modalities over the treatment period. In this paper, we introduce the Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset that addresses the above limitations. This is the first OCT and fundus dataset that includes clinical labels, biomarker labels, and time-series patient treatment information from associated clinical trials. The dataset consists of $1268$ fundus eye images each with $49$ OCT scans, and $16$ biomarkers, along with $3$ clinical labels and a disease diagnosis of DR or DME. In total, there are $96$ eyes' data averaged over a period of at least two years with each eye treated for an average of $66$ weeks and $7$ injections. OLIVES dataset has advantages in other fields of machine learning research including self-supervised learning as it provides alternate augmentation schemes that are medically grounded.

## Dataset
**Images and Labels**: https://doi.org/10.5281/zenodo.7105232

**Labels**:
There are two directories for the labels: full_labels and ml_centric labels. 

**Full labels** contain all the clinical inforamtion used in these studies for the associated studies of interest.

**ML Centric labels** are divided into two files: Biomarker_Clinical_Data_Images.csv and Clinical_Data_Images.xlsx. 

Biomarker_Clinical_Data_Images.csv contains full biomarker and clinical labels for the 9408 images that have this labeled biomarker information.

Clinical_Data_Images.xlsx has the BCVA, CST, Eye ID, and Patient ID for the 78000+ images that have clinical data. 

## Code Usage

**Self-Supervised Experiments**:
1. Go to the **Biomarker Interpretation with Contrastive Learning** directory and set the python path with: export PYTHONPATH=$PYTHONPATH:$PWD.
2. Train the backbone network with the supervised contrastive loss using the parameters specified in config/config_supcon.py \
a) Specify number of clinical labels to train with --num_methods parameter \
b) Specify which clinical labels to train with --method1, --method2, etc. \
c) An example of a script would be: \
python training_main/clinical_sup_contrast.py --dataset 'Prime_TREX_DME_Fixed' --num_methods 1 --method1 'bcva'
3. Train the appended linear using the parameters specified in config/config_linear.py \
a) Set the super flag to identify whether to use contrastively trained backbone (0), completely supervised (1), or fusion supervised (2). \
b) Set the multi flag to (1) in order to control whether multi-label classification is used and (0) otherwise. \
c) If not using multi-label classification, then set the biomarker flag to the biomarker of interest used in this study. \
d) An example of this script would be: 
python training_main/main_linear.py --dataset 'Prime' --multi 0 --super 0 --ckpt 'path_to_checkpoint file' --biomarker 'fluid_irf'


**Treatment Prediction with Fundus and OCT Experiments**:
1. Go to the treatment_pred_fundus_oct directory and set the python path with: export PYTHONPATH=$PYTHONPATH:$PWD. 
2. Generate the treatment labels using the clinical labels provided. In the paper, these treatment labels were defined with respect to an increase of BCVA on a week to week basis. Save these labels in a csv file and set the train and test path for the generated file within the config/config_linear.py file. 
3. Train the backbone network by setting the parameters of interest in the config/config_linear.py file. 
4. Set whether training utilizes Fundus or OCT Volumes of data. 
5. An example usage of this directory would be:
python training_main/main_linear.py --dataset 'Fundus_Treatment' --train_csv_path 'path to csv' --test_csv_path 'path to csv' --super 1 --epochs 50

## Links

**Associated Website**: https://ghassanalregib.info/

**Code Acknowledgement**: Code uses loss from https://github.com/HobbitLong/SupContrast.git.

## Citations

If you find the work useful, please include the following citation in your work:

>@inproceedings{prabhushankarolives2022,\
  title={OLIVES Dataset: Ophthalmic Labels for Investigating
Visual Eye Semantics},\
  author={Prabhushankar, Mohit and Kokilepersaud, Kiran and Logan, Yash-yee and Trejo Corona, Stephanie and AlRegib, Ghassan and Wykoff, Charles},\
  booktitle={Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 2 (NeurIPS Datasets and Benchmarks 2022) },\
  year={2022}\
}
