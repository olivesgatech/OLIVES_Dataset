# OLIVES_Dataset

## Abstract
Clinical diagnosis of the eye is performed over multifarious data modalities including scalar clinical labels, vectorized biomarkers, two-dimensional fundus images, and three-dimensional Optical Coherence Tomography (OCT) scans. While the clinical labels, fundus images and OCT scans are instrumental measurements, the vectorized biomarkers are interpreted attributes from the other measurements. Clinical practitioners use all these data modalities for diagnosing and treating eye diseases like Diabetic Retinopathy (DR) or Diabetic Macular Edema (DME). Enabling usage of machine learning algorithms within the ophthalmic medical domain requires research into the relationships and interactions between these relevant data modalities. Existing datasets are limited in that: ($i$) they view the problem as disease prediction without assessing biomarkers, and ($ii$) they do not consider the explicit relationship among all four data modalities over the treatment period. In this paper, we introduce the Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset that addresses the above limitations. This is the first OCT and fundus dataset that includes clinical labels, biomarker labels, and time-series patient treatment information from associated clinical trials. The dataset consists of $1268$ fundus eye images each with $49$ OCT scans, and $16$ biomarkers, along with $3$ clinical labels and a disease diagnosis of DR or DME. In total, there are $96$ eyes' data averaged over a period of at least two years with each eye treated for an average of $66$ weeks and $7$ injections. OLIVES dataset has advantages in other fields of machine learning research including self-supervised learning as it provides alternate augmentation schemes that are medically grounded.

## Dataset
Images: https://doi.org/10.5281/zenodo.6622145

**Labels**:
There are two directories for the labels: full_labels and ml_centric labels. 

**Full labels** contain all the clinical inforamtion used in these studies for the associated studies of interest.

**ML Centric labels** are divided into two files: Biomarker_Clinical_Data_Images.csv and Clinical_Data_Images.xlsx. 

Biomarker_Clinical_Data_Images.csv contains full biomarker and clinical labels for the 9408 images that have this labeled biomarker information.

Clinical_Data_Images.xlsx has the BCVA, CST, Eye ID, and Patient ID for the 78000+ images that have clinical data. 

## Code Usage

Self-Supervised Experiments:

Treatment Prediction with Fundus and OCT Experiments:
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
