# PerceptionChallenge
Application code for the perception team of Wisconsin Autonomous

answer.png:
-
<img src="answer.png" alt="Answer" width="400">


Methodology:
-
- Attempted to replicate sample answer.png image given originally.
- Researched different algorithms to determine which would be best suited for cone/color detection and learned how to use the relevant libraries used in code.
- First preprocessed the input image by converting it to the HSV color space and applying Gaussian blur to reduce noise when first coding.
- Applied color thresholding to isolate orange cones and used morphological transformations to refine the detected regions.
- Used contour detection to extract cone positions, which was then clustered into left and right boundaries using DBSCAN through Scikit-learn. 
- Also looked up methods for annotating boundary lines and implemented RANSAC regression to fit smooth curves through the detected cone boundaries, which was then drawn onto the original image.

Issues:
-
- Initially tried using K-Means clustering to separate the cones into left and right groups, but it sometimes failed when cones were unevenly distributed (as percieved by the algorithm). Attempted polynomial fitting, but it resulted in unstable curves when dealing with outliers (with other shades of color).
- Attempted to also use curves originally instead of straight lines to connect the cones together for boundary detection but failed when left side curved out of boundary and stopped connecting after three cones from the left (K-Means was used for this as well).
- Switching to DBSCAN helped improve clustering by handling noise points, and RANSAC regression provided a more robust curve-fitting method.
- Some issues remained, including false positives in cone detection and missing cones in cases of varying lighting such as the bright colors in the original picture that can cause confusion in boundary detection.

Libraries Used:
-
- OpenCV for image processing, NumPy for numerical operations, and Scikit-learn for clustering (DBSCAN) and regression (RANSAC).



