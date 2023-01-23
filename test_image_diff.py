import cv2


image1 = cv2.imread("./media/videos/screenshots/parisc-hostage-20230122184333622283/20230122184333653819.jpg")
image2 = cv2.imread("./media/videos/screenshots/parisc-hostage-20230122184333622283/20230122184333672243.jpg")
image3 = cv2.imread("./media/videos/screenshots/parisc-hostage-20230122184333622283/20230122184333699324.jpg")
image4 = cv2.imread("./media/videos/screenshots/parisc-hostage-20230122184333622283/20230122184333734304.jpg")
image5 = cv2.imread("./media/videos/screenshots/parisc-hostage-20230122184333622283/20230122184333778752.jpg")

def measure_image_difference(image1, image2):
    # Convert images to grayscale
    image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Calculate the absolute difference between the two images
    difference = cv2.absdiff(image1_gray, image2_gray)
    
    # Apply a threshold to the difference image to make it binary
    threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
    
    # Calculate the percentage of pixels that are different between the two images
    total_pixels = image1.shape[0] * image1.shape[1]
    different_pixels = cv2.countNonZero(threshold)
    difference_percentage = (different_pixels / total_pixels) * 100
    
    return difference_percentage

# Example usag
difference_1v2 = measure_image_difference(image1, image2)
difference_2v3 = measure_image_difference(image2, image3)
difference_3v4 = measure_image_difference(image3, image4)
difference_4v5 = measure_image_difference(image4, image5)

print("Difference percentage: {:.2f}%".format(difference_1v2))
print("Difference percentage: {:.2f}%".format(difference_2v3))
print("Difference percentage: {:.2f}%".format(difference_3v4))
print("Difference percentage: {:.2f}%".format(difference_4v5))