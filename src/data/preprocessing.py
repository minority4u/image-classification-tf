
import cv2
import logging


def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


def create_slides(image, slice_width, slice_height):
    (winW, winH) = (slice_width,slice_height)
    result = []
    # loop over the image pyramid
    #for resized in pyramid(image, scale=100):
        # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(image, stepSize=400, windowSize=(winW, winH)):
            # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = image.copy()
        #cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        crop_img = clone[y:y + winW, x:x + winH]
        result.append(crop_img)

    return result

