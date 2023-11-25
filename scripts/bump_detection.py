import cv2
import numpy as np
import itertools
import pandas as pd
list_images = ['IMG_0757.jpg_patch.jpg']


#,'IMG_1240.jpg','IMG_1733.jpg','IMG_1734.jpg','IMG_1964.jpg'

def Show_image(label,image_sample):
	cv2.imshow(label,image_sample)

	

def endpoints(rho, theta): 
	a = np.cos(theta) 
	b = np.sin(theta) 
	x_0 = a * rho 
	y_0 = b * rho 
	x_1 = int(x_0 + 1000 * (-b)) 
	y_1 = int(y_0 + 1000 * (a)) 
	x_2 = int(x_0 - 1000 * (-b)) 
	y_2 = int(y_0 - 1000 * (a)) 
	return ((x_1, y_1), (x_2, y_2)) 


def hough_transform(): 
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale 
	kernel = np.ones((15, 15), np.uint8) 
	opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)  # Open (erode, then dilate) 
	#Show_image('opening',opening)
	edges = cv2.Canny(opening, 50, 150, apertureSize=3)  # Canny edge detection 
	#Show_image('edges',edges)

	lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)  # Hough line detection 
	hough_lines = [] 
 # Lines are represented by rho, theta; converted to endpoint notation 
	if lines is not None: 
		 for line in lines: 
			 hough_lines.extend(list(itertools.starmap(endpoints, line))) 
	return hough_lines 


def det(a, b):
	return a[0] * b[1] - a[1] * b[0]


# Find intersection point of two lines (not segments!)
def line_intersection(line1, line2):
    x_diff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    y_diff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(x_diff, y_diff)
    if div == 0:
        return None  # Lines don't cross
    d = (det(*line1), det(*line2))
    x = det(d, x_diff) / div
    y = det(d, y_diff) / div
    return x, y


# Find intersections between multiple lines (not line segments!)
def find_intersections(lines):
    intersections = []
    for i, line_1 in enumerate(lines):
        for line_2 in lines[i + 1:]:
            if not line_1 == line_2:
                intersection = line_intersection(line_1, line_2)
                if intersection:  # If lines cross, then add
                    intersections.append(intersection)
    return intersections

# Given intersections, find the grid where most intersections occur and treat as vanishing point
def find_vanishing_point(img,img_name, grid_size, intersections):
    # Image dimensions
	image_height = img.shape[0]
	image_width = img.shape[1]

    # Grid dimensions
	grid_rows = (image_height // grid_size) + 1
	grid_columns = (image_width // grid_size) + 1

    # Current cell with most intersection points
	max_intersections = 0
	best_cell = (0.0, 0.0)

	for i, j in itertools.product(range(grid_columns),range(grid_rows)):
		cell_left = i * grid_size
		cell_right = (i + 1) * grid_size
		cell_bottom = j * grid_size
		cell_top = (j + 1) * grid_size
		cv2.rectangle(img, (cell_left, cell_bottom), (cell_right, cell_top), (0, 0, 255), 10)

		current_intersections = 0  # Number of intersections in the current cell

		for x, y in intersections:
			if cell_left < x < cell_right and cell_bottom < y < cell_top:
				current_intersections += 1
        # Current cell has more intersections that previous cell (better)
			if current_intersections > max_intersections:
				max_intersections = current_intersections
				best_cell = ((cell_left + cell_right) / 2, (cell_bottom + cell_top) / 2)
				#print("Best Cell:", best_cell)

		if best_cell[0] != None and best_cell[1] != None:
			rx1 = int(best_cell[0] - grid_size / 2)
			ry1 = int(best_cell[1] - grid_size / 2)
			rx2 = int(best_cell[0] + grid_size / 2)
			ry2 = int(best_cell[1] + grid_size / 2)
			cv2.rectangle(img, (rx1, ry1), (rx2, ry2), (0, 255, 0), 10)
			outfile = '%s_Vanishing_box.jpg' % (img_name)
			cv2.imwrite(outfile, img)

	return best_cell


#########################################
list_images = ['IMG_0757.jpg_patch.jpg'] 
for img_name in list_images:
	img = cv2.imread(img_name)
	img = cv2.resize(img,(960,640))
	image = img
	image_1 = img
	hough_lines = hough_transform() 
	lineThickness = 1
	#print (hough_lines)
	for hough_line in hough_lines:
		#print (hough_line[0][1])
		cv2.line(img, (hough_line[0][0], hough_line[0][1]), (hough_line[1][0], hough_line[1][1]), (0,255,0), lineThickness)
	Show_image('image with lines',img)
	
	
	#intersection_points = find_intersections(hough_lines)
	
	#for point in intersection_points:
	#	cv2.circle(image, (int(point[0]),int(point[1])), 0, (0, 0, 255), -1)
	#Show_image('image with intersection points',image)
	

	#grid_size = 100
	#best_point = find_vanishing_point(image,img_name,grid_size, intersection_points)
	#print (int(best_point[0]),int(best_point[1]))
	#cv2.circle(image_1, (int(best_point[0]),int(best_point[1])), 10, (255, 0, 0), 10)
	#cv2.imshow("image with best intersection",image_1)
	
	#input()
	
	

	cv2.waitKey(0)


cv2.destroyAllWindows()
#########################################################################
	#df_intersection_points= pd.DataFrame(intersection_points)
	#length = df_intersection_points.shape[0]
	
	#import matplotlib.pyplot as plt
	
	# Plot
#	plt.scatter(df_intersection_points.iloc[:,0], df_intersection_points.iloc[:,1])
#	plt.title('Scatter plot')
#	plt.xlabel('x')
#	plt.ylabel('y')
#	plt.show()