# OLIVES_Dataset

## Abstract
Clinical diagnosis of the eye is performed over multifarious data modalities including scalar clinical labels, vectorized biomarkers, two-dimensional fundus images, and three-dimensional Optical Coherence Tomography (OCT) scans. While the clinical labels, fundus images and OCT scans are instrumental measurements, the vectorized biomarkers are interpreted attributes from the other measurements. Clinical practitioners use all these data modalities for diagnosing and treating eye diseases like Diabetic Retinopathy (DR) or Diabetic Macular Edema (DME). Enabling usage of machine learning algorithms within the ophthalmic medical domain requires research into the relationships and interactions between these relevant data modalities. Existing datasets are limited in that: ($i$) they view the problem as disease prediction without assessing biomarkers, and ($ii$) they do not consider the explicit relationship among all four data modalities over the treatment period. In this paper, we introduce the Ophthalmic Labels for Investigating Visual Eye Semantics (OLIVES) dataset that addresses the above limitations. This is the first OCT and fundus dataset that includes clinical labels, biomarker labels, and time-series patient treatment information from associated clinical trials. The dataset consists of $1268$ fundus eye images each with $49$ OCT scans, and $16$ biomarkers, along with $3$ clinical labels and a disease diagnosis of DR or DME. In total, there are $96$ eyes' data averaged over a period of at least two years with each eye treated for an average of $66$ weeks and $7$ injections. OLIVES dataset has advantages in other fields of machine learning research including self-supervised learning as it provides alternate augmentation schemes that are medically grounded.

## Dataset

## Code Usage
## Links

**Associated Website**: https://ghassanalregib.info/

**Code Acknowledgement**: Code uses loss from https://github.com/HobbitLong/SupContrast.git.

## Citations

If you find the work useful, please include the following citation in your work:

>@inproceedings{prabhushankarolives2022,\
  title={OLIVES Dataset: Ophthalmic Labels for Investigating
Visual Eye Semantics},\
  author={Prabhushankar, Mohit, Kokilepersaud, Kiran, Logan, Yash-yee, Trejo Corona, Stephanie, AlRegib, Ghassan and Wykoff, Charles},\
  booktitle={2022 },\
  year={2022}\
}
