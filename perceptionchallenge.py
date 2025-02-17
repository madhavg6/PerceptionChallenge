import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor

"""Loads the image, converts it to HSV, and applies Gaussian Blur."""
def load_and_preprocess(image_path):
   image = cv2.imread(image_path)  # Reads  image from the file
   hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Converts image from BGR to HSV color space
   blurred = cv2.GaussianBlur(hsv, (5, 5), 0)  # Applies Gaussian Blur to reduce noise that would confuse image analysis
   return image, blurred

"""Detects orange cones using HSV thresholding and morphological transformations."""
def detect_cones(hsv):
   # Defines HSV range for detecting orange color
   lower_orange = np.array([10, 100, 100])
   upper_orange = np.array([25, 255, 255])
   mask = cv2.inRange(hsv, lower_orange, upper_orange)  # Creates binary mask for orange regions (cone color)

   # Morphological operations to remove noise and fill gaps
   kernel = np.ones((5, 5), np.uint8)
   mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Closes small holes in the detected areas
   mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Removes small noise

   return mask

"""Finds contours of detected cones and extracts centroid points."""
def find_cone_positions(mask):
   contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   cone_points = []

   for cnt in contours:
       x, y, w, h = cv2.boundingRect(cnt)  # Gets bounding box around each contour
       centroid = (x + w // 2, y + h // 2)  # Calculates centroid of the bounding box
       cone_points.append(centroid)  # Stores centroid as a cone position

   return cone_points

"""Uses DBSCAN clustering to separate cones into left and right boundaries."""
def cluster_cones(cone_points, image_width):
   if len(cone_points) < 4:  # If fewer than 4 cones are detected, return empty lists
       return [], []

   # Apply DBSCAN clustering to group nearby cone points
   clustering = DBSCAN(eps=50, min_samples=2).fit(cone_points)
   labels = clustering.labels_  # Gets cluster labels for each point

   left_cones, right_cones = [], []
   for i, label in enumerate(labels):
       if label == -1:  # Ignores noise points (label = -1)
           continue
       if cone_points[i][0] < image_width // 2:  # Classifies based on image width (left vs right)
           left_cones.append(cone_points[i])
       else:
           right_cones.append(cone_points[i])

   # Sorts cones based on their vertical (y-axis) position for better curve fitting
   return sorted(left_cones, key=lambda x: x[1]), sorted(right_cones, key=lambda x: x[1])

"""Fits a robust RANSAC regression curve to the cone positions."""
def fit_ransac_curve(points):
   if len(points) < 3:  # Not enough points to fit a curve
       return None

   x, y = zip(*points)  # Separates x and y coordinates
   model = RANSACRegressor()  # Initializes RANSAC regression model
   y = np.array(y).reshape(-1, 1)  # Reshapes y values for model input
   model.fit(y, x)  # Fits model with y as input and x as output

   def curve_func(y_vals):
       return model.predict(y_vals.reshape(-1, 1)).astype(int)  # Predicts x values for given y

   return curve_func

"""Draws smooth boundary lines using RANSAC regression."""
def draw_boundaries(image, left_cones, right_cones):
   height = image.shape[0]  # Gets image height
   y_values = np.linspace(0, height, num=100).astype(int)  # Generates y values for curve drawing

   left_curve = fit_ransac_curve(left_cones)  # Fit curve for left boundary
   right_curve = fit_ransac_curve(right_cones)  # Fit curve for right boundary

   def draw_curve(curve_func, color):
       if curve_func is not None:
           # Generates curve points
           curve_points = [(int(curve_func(y).item()), y) for y in y_values]
           for i in range(len(curve_points) - 1):
               cv2.line(image, curve_points[i], curve_points[i + 1], color, 3)  # Draws curve line

   boundary_color = (0, 165, 255)  # Orange color for both boundary lines
   draw_curve(left_curve, boundary_color)
   draw_curve(right_curve, boundary_color)

   return image

"""Main function to process the image and detect boundaries."""
def detect_and_draw(image_path, output_path="answer.png"):
   image, hsv = load_and_preprocess(image_path)  # Loads and preprocesses the image
   mask = detect_cones(hsv)  # Detects cones using color thresholding
   cone_points = find_cone_positions(mask)  # Finds centroid positions of cones

   if len(cone_points) < 4:  # Ensures enough cones are detected
       print("Not enough cones detected.")
       return

   left_cones, right_cones = cluster_cones(cone_points, image.shape[1])  # Clusters cones into left and right
   result_image = draw_boundaries(image, left_cones, right_cones)  # Draws boundary curves
   cv2.imwrite(output_path, result_image)  # Saves the result image
   print(f"Output saved to {output_path}")

# Runs detection
detect_and_draw("original.png")