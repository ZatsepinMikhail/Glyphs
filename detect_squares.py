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

def draw_vertical_line( image, position, color ):
	height, _, _ = image.shape
	for i in range( height ):
		image[i, position] = color

def draw_horizontal_line( image, position, color ):
	_, width, _ = image.shape
	for i in range( width ):
		image[position, i] = color

def crop_image( filename ):

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
	return cropped_image

image_name = '1.jpg'
image = crop_image( image_name )
grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, binarized_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite( add_suffix(image_name, '_cropped_binarized'), binarized_image )

vertical_histogram = cv2.reduce( binarized_image, 0, cv2.cv.CV_REDUCE_AVG)[0]
horizontal_histogram_raw = cv2.reduce( binarized_image, 1, cv2.cv.CV_REDUCE_AVG)
horizontal_histogram = [ x[0] for x in horizontal_histogram_raw ]

start_color = (255, 0, 0)
stop_color = (0, 255, 0)

# -------------------------------------------vertical histogram--------------------------------------------------------------------

horizontal_starts = []
horizontal_stops = []

acceptable_threshold = 0
min_distance_between_columns = 20
in_hieroglyphs = False
for i in range( image.shape[1] ):
	if vertical_histogram[i] < 255 - acceptable_threshold:
		if not in_hieroglyphs:
			in_hieroglyphs = True
			horizontal_starts.append(i)
			draw_vertical_line( image, i, start_color )
	elif in_hieroglyphs:
		in_hieroglyphs = False
		horizontal_stops.append(i)
		draw_vertical_line( image, i, stop_color )

horizontal_zone_sizes = []
stop_index = 0
for start_index in range( len( horizontal_starts ) ):
	while stop_index < len(horizontal_stops) and horizontal_stops[stop_index] < horizontal_starts[start_index]:
		stop_index = stop_index + 1
	if stop_index == len( horizontal_stops ):
		break
	else:
		horizontal_zone_sizes.append( horizontal_stops[stop_index] - horizontal_starts[start_index] )
		print horizontal_starts[start_index], horizontal_stops[stop_index], horizontal_stops[stop_index] - horizontal_starts[start_index]

horizontal_zone_sizes.sort()
column_width = horizontal_zone_sizes[len(horizontal_zone_sizes) / 2]
print 'zone number: ' + str(len(horizontal_zone_sizes))
print min(horizontal_zone_sizes)
print horizontal_zone_sizes[len(horizontal_zone_sizes) / 2]
print max(horizontal_zone_sizes)
print '\n----------------------------\n'
print horizontal_zone_sizes

# -------------------------------------------horizontal histogram-----------------------------------------------------------------

vertical_starts = []
vertical_stops = []

acceptable_threshold = 0
min_distance_between_columns = 20
in_hieroglyphs = False
for i in range( image.shape[0] ):
	if horizontal_histogram[i] < 255 - acceptable_threshold:
		if not in_hieroglyphs:
			in_hieroglyphs = True
			vertical_starts.append(i)
			draw_horizontal_line( image, i, start_color )
	elif in_hieroglyphs:
		in_hieroglyphs = False
		vertical_stops.append(i)
		draw_horizontal_line( image, i, stop_color )

vertical_zone_sizes = []
stop_index = 0
for start_index in range( len( vertical_starts ) ):
	while stop_index < len(vertical_stops) and vertical_stops[stop_index] < vertical_starts[start_index]:
		stop_index = stop_index + 1
	if stop_index == len( vertical_stops ):
		break
	else:
		vertical_zone_sizes.append( vertical_stops[stop_index] - vertical_starts[start_index] )
		print vertical_starts[start_index], vertical_stops[stop_index], vertical_stops[stop_index] - vertical_starts[start_index]

vertical_zone_sizes.sort()
print 'zone number: ' + str(len(vertical_zone_sizes))
print min(vertical_zone_sizes)
print vertical_zone_sizes[len(vertical_zone_sizes) / 2]
print max(vertical_zone_sizes)
print '\n----------------------------\n'
print horizontal_zone_sizes
raw_height = vertical_zone_sizes[len(vertical_zone_sizes) / 2]

# -------------------------------------------cut------------------------------------------------------------------------------------

columns = []
#for 

cv2.imwrite( add_suffix(image_name, '_histograms'), image )
