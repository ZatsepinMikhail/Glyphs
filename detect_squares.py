import numpy as np
import cv2

def get_distance( point1, point2 ):
	return ( point1[0][0] - point2[0][0] ) ** 2 + ( point1[0][1] - point2[0][1] ) ** 2

def get_distance_simple( point1, point2 ):
	return ( point1[0] - point2[0] ) ** 2 + ( point1[1] - point2[1] ) ** 2

def is_big_square( points, big_threshold ):
	distances = []
	for i in range( 0, len(points) ):
		distances.append( get_distance( points[i], points[ ( i + 1 ) % len( points ) ] ) )

	#print distances
	#print '-------'
	average_distance = sum( distances ) / len( distances )

	return max( distances ) / min( distances ) < 1.05 and average_distance > big_threshold

def get_big_squares( binarized_image ):
	
	height, width = binarized_image.shape
	big_square_threshold = ( width / 30 ) ** 2

	all_contours, _ = cv2.findContours(binarized_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	result_contours = []
	for contour in all_contours:
		approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
		if len(approx) == 4:
			if is_big_square( approx, big_square_threshold ):
				result_contours.append( approx )

	return result_contours

image = cv2.imread('rotated_grayscale 001.jpg', cv2.IMREAD_COLOR)
grayscale_image = cv2.imread('rotated_binarized 001.jpg', cv2.IMREAD_GRAYSCALE)

ret, binarized_image = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
height, width = binarized_image.shape

big_square_contours = get_big_squares( binarized_image )

#for contour in big_square_contours:
#	cv2.drawContours(image, [contour], 0, (0, 0, 255), -1)

assert( len( big_square_contours ) == 4 )

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

cv2.imwrite( 'img.jpg', rotated_image )
#cv2.imwrite( 'img.jpg', image[min( centers_y ) : max( centers_y ), min(centers_x) : max( centers_x)] )
