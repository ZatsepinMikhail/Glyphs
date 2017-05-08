import numpy as np
import cv2

def get_distance( point1, point2 ):
	return ( point1[0][0] - point2[0][0] ) ** 2 + ( point1[0][1] - point2[0][1] ) ** 2

def get_distance_simple( point1, point2 ):
	return ( point1[0] - point2[0] ) ** 2 + ( point1[1] - point2[1] ) ** 2

def is_big_square( points, big_threshold_low, big_threshold_up ):
	distances = []
	for i in range( 0, len(points) ):
		distances.append( get_distance( points[i], points[ ( i + 1 ) % len( points ) ] ) )

	#print distances
	#print '-------'
	average_distance = sum( distances ) / len( distances )

	return ( max( distances ) / min( distances ) < 1.05 and average_distance > big_threshold_low and average_distance < big_threshold_up, average_distance ** 0.5 )

def get_big_squares( binarized_image ):
	
	height, width = binarized_image.shape
	big_square_threshold_low = ( width / 30 ) ** 2
	big_square_threshold_up = ( width / 10 ) ** 2

	all_contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	result_contours = []
	average_side_sizes = []
	for contour in all_contours:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		if len(approx) == 4:

			is_big_square_result = is_big_square( approx, big_square_threshold_low, big_square_threshold_up )
			if is_big_square_result[0]:
				result_contours.append( approx )
				average_side_sizes.append( is_big_square_result[1] )


	return (result_contours, average_side_sizes)


def add_suffix( filename, suffix ):
	dot_position = filename.find('.')
	return filename[:dot_position] + suffix + filename[dot_position:]


def find_squares( filename ):

	image = cv2.imread(filename, cv2.IMREAD_COLOR)
	grayscale_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

	#cv2.imwrite( 'res1.jpg', image )

	ret, binarized_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	height, width = binarized_image.shape

	#cv2.imwrite( 'res2.jpg', binarized_image )

	eroded_image = cv2.erode(binarized_image, np.ones((5, 5)))
	#cv2.imwrite( 'res3.jpg', dilated_image )

	big_square_contours, average_side_sizes = get_big_squares( eroded_image )

	#for contour in big_square_contours:
	#	print( contour )
	#	cv2.drawContours(image, [contour], 0, (0, 0, 255), -1)

	#cv2.imwrite( 'res4.jpg', image )

	assert( len( big_square_contours ) == 4 )
	side_size = sum( average_side_sizes ) / 4

	centers = []
	for contour in big_square_contours:
		sum_x = 0
		sum_y = 0
		for i in range(0, 4):
			sum_x += contour[i][0][0]
			sum_y += contour[i][0][1]
		centers.append( ( sum_x / 4, sum_y / 4 ) )

	# 0      1
	#
	#
	#
	# 2      3
	centers_sorted = np.zeros( ( 3, 2 ), dtype='f4' )
	for i in range( 4 ):
		if centers[i][0] < width / 2:
			if centers[i][1] < height / 2:
				centers_sorted[0] = centers[i]
			else:
				centers_sorted[2] = centers[i]
		else:
			if centers[i][1] < height / 2:
				centers_sorted[1] = centers[i]

	centers_rotated = np.copy( centers_sorted )
	centers_rotated[1] = ( centers_rotated[0][0] + get_distance_simple( centers_rotated[0], centers_rotated[1] ) ** 0.5 , centers_rotated[0][1] )
	centers_rotated[2] = ( centers_rotated[0][0], centers_rotated[0][1] + get_distance_simple( centers_rotated[0], centers_rotated[2] ) ** 0.5 )

	print centers_sorted
	print centers_rotated
	transform = cv2.getAffineTransform( centers_sorted[:3], centers_rotated )

	rotated_image = cv2.warpAffine( image, transform, ( width, height ) )

	cv2.imwrite( add_suffix(filename, '_rotated'), rotated_image )

	# crop
	new_width = centers_rotated[1][0] - centers_rotated[0][0]
	new_height = centers_rotated[2][1] - centers_rotated[0][1]

	left_upper_x = centers_rotated[0][0] + side_size / 2
	left_upper_y = centers_rotated[0][1] + side_size / 2
	right_down_x = left_upper_x + new_width - side_size
	right_down_y = left_upper_y + new_height - side_size

	cropped_image = rotated_image[left_upper_y : right_down_y, left_upper_x : right_down_x]
	cv2.imwrite( add_suffix(filename, '_cropped'), cropped_image )

find_squares( '1.jpg' )