# Computer Vision for Airborne Imaging of Antarctica: Creating a 75-Year Record of Surface Change
![Succesful matching results](results.png)
**You can view the full research paper** [here]([SURF 2024 FINAL REPORT.pdf](https://docs.google.com/document/d/1Rt2t64TUNOChu_n2d8sS8sAvSKOVvHRjNxbfG_iSQrM/edit?usp=sharing)).

**Author:** Sani Deshmukh, California Institute of Technology  
**Research Mentor:** James Dickson (Polar Geospatial Center Director, UMN)  
**Associate Mentor:** Professor Bethany Ehlmann (Caltech)

## Abstract
Airborne imaging of Antarctica exists dating back to 1947, but the data are not registered to the surface, preventing efficient comparisons between the surface 75 years ago and today. This project evaluates the efficacy of contemporary Computer Vision (CV) algorithms—SIFT, AKAZE, KAZE, SURF, and ORB—in registering high-resolution (<10 m/pixel) time-series airborne imagery to cartographically accurate elevation models.

## Approach
We employed an iterative feature-matching approach that aligns simulated terrains generated from a high-resolution Antarctic elevation model (REMA) with aerial images. Key steps include:

- **Feature Matching:** Fine-tuning CV algorithms to match features between aerial imagery and elevation models.
- **Outlier Detection:** Using advanced techniques to discard inaccurate matches.
- **Optimization:** Iteratively refining parameters to improve precision, particularly in handling shadows and disregarding snow and debris.

## Results
The SIFT and AKAZE algorithms demonstrated superior performance in matching aerial images, especially when similarly oriented relative to the base elevation model. This work is a successful proof-of-concept for georeferencing historic overflight imagery, extending the temporal analysis of Antarctic surface changes from a few decades to nearly a century.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Algorithms](#algorithms)
- [Results](#results)
- [Future Work](#future-work)
- [Contributors](#contributors)

## Installation
To run this project, install the following dependencies:

```bash
pip install opencv-python numpy remapy
```

## Usage
Clone the repository:

```bash
git clone https://github.com/yourusername/antarctica-cv.git
```

Run the main script:

```bash
python main.py --input ./data/airborne_images --model ./data/REMA_model
```

## Algorithms
The project uses the following algorithms for feature matching:

- **SIFT** (Scale-Invariant Feature Transform)
- **AKAZE**
- **KAZE**
- **SURF** (Speeded-Up Robust Features)
- **ORB** (Oriented FAST and Rotated BRIEF)

## Results
The project shows that **SIFT** and **AKAZE** outperform other algorithms in matching aerial images, especially when aligned with elevation models.

## Future Work
- Enhance outlier detection using machine learning techniques.
- Automate the feature alignment process across various orientations and conditions.
- Scale up to process the full Antarctic archive of aerial imagery.

## Contributors
- **Sani Deshmukh** - [GitHub](https://github.com/sani-deshmukh)
- **James Dickson** - Research Mentor
- **Bethany Ehlmann** - Associate Mentor
