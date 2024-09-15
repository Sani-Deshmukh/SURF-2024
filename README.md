Computer Vision for Airborne Imaging of Antarctica: Creating a 75-year Record of Surface Change 
Sani Deshmukh, California Institute of Technology
Research Mentor: 
Polar Geospatial Center Director James Dickson (UMN)
Associate Mentor: 
Professor Bethany Ehlmann (Caltech)

Abstract
Airborne imaging of Antarctica exists dating back to 1947, but the data are not registered to the surface, preventing efficient comparisons between the surface 75 years ago and today. This project examines the efficacy of various contemporary Computer Vision (CV) algorithms—namely SIFT, AKAZE, KAZE, SURF, and ORB—in registering high-resolution (<10 m/pixel) time-series airborne imagery to cartographically accurate elevation models. To address this, we employed an iterative feature-matching approach that aligns simulated terrains generated from a high-resolution Antarctic elevation model (REMA) with aerial images, enhancing the precision of feature matching by cumulatively retaining accurate matches and discarding outliers. Matching aerial images with an elevation model involved fine-tuning CV algorithm parameters, refining outlier detection techniques, and optimizing iteration and matching processes for matching on inconsistent shadows while disregarding the snow and debris outcrops. Our findings indicate that the SIFT and AKAZE algorithms demonstrate superior performance in matching aerial images, particularly when these images are similarly oriented relative to a larger base map elevation model. These results represent a successful proof-of-concept that will allow accurate georeferencing of historic overflight imagery and extend the temporal analysis of Antarctic surface changes from a few decades to nearly a century.
