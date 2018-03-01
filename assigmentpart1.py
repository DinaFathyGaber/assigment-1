import numpy as np
import matplotlib.pyplot as plt
import mpldatacursor
import matplotlib.image as mpimg
import cv2
# Question 1,2,3 load 2 images, view them + Histogram
image1 = mpimg.imread("images.jpg")
image2 = mpimg.imread("cameraman1.png")

image2 = np.array( image2 , dtype = np.float32 )


print( image1.shape )
print( image2.shape )

plt.figure()
plt.imshow(image1)

plt.figure()
plt.gray()
plt.imshow(image2)

plt.figure()
plt.hist( image1.flatten() , 128 )

plt.figure()
plt.hist( image2.flatten() , 128 )
# Question 4: X+Y coordinates
image = mpimg.imread("some-pigeon.jpg")
fig, ax = plt.subplots()
ax.imshow(image, interpolation='none')

mpldatacursor.datacursor(hover=True, bbox=dict(alpha=1, fc='w'))

plt.show()

# Question 4: Pixel and window values


def draw_circle(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        intensity = (img[y, x][2]+img[y, x][1]+img[y, x][0])/3.0
        print("=============================")
        print("Intensity of pixel: {0}".format(intensity))
        if 5 < x < width-6 and 5 < y < height-6:
            box = img[y-5:y+6, x-5:x+6]
            b, g, r = cv2.split(box)
            box_intensity = (b + g + r)/3
            print("Mean: {0}, Standard deviation: {1}".format(np.mean(box_intensity), np.std(box_intensity)))
            cv2.imshow("ROI", box)


def rgb2gray(rgb_image):
    return np.dot(rgb_image[..., :3], [0.299, 0.587, 0.114])


img = cv2.imread("./images/peppers.png")
box = img[0:11, 0:11]
height, width, channels = img.shape
# display the image
cv2.namedWindow("Original Image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("Original Image", draw_circle)
cv2.imshow("Original Image", img)
cv2.namedWindow("ROI", cv2.WINDOW_NORMAL)
cv2.imshow("ROI", box)
cv2.waitKey(0)

# split into channels
rgb = cv2.split(img)

# list to select colors of each channel line
colors = ("b", "g", "r")

# create the histogram plot, with three lines, one for
# each color
plt.xlim([0, 256])
for(channel, c) in zip(rgb, colors):
    histogram = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plt.plot(histogram, color=c)

plt.xlabel("Color value")
plt.ylabel("Pixels")
plt.show()
# Question 5: roll


def multi_view(images):
    images_count = len(images)
    fig = plt.figure(figsize=(10, 20))
    for row in range(images_count):
        ax1 = fig.add_subplot(images_count, 1, row + 1)
        ax1.imshow(images[row])
        plt.show()


def scale(image):
    return 255 * (image - np.min(image)) / (np.max(image) - np.min(image))

images = mpimg.imread('some-pigeon.jpg')

gray_image = rgb2gray(images)

gray_image_v = np.roll(gray_image, 1, 0)
gray_image_h = np.roll(gray_image, 1, 1)

gray_image_gv = np.abs(gray_image - gray_image_v)
gray_image_gh = np.abs(gray_image - gray_image_h)

comb = tuple((gray_image, gray_image_gv, gray_image_gh))

multi_view(comb)

# Question 5: for loops
GradientX = np.zeros((gray_image.shape[0], gray_image.shape[1]))
GradientY = np.zeros((gray_image.shape[0], gray_image.shape[1]))
for y in range(gray_image.shape[0]):
    for x in range(gray_image.shape[1]):
        GradientY[y, x] = np.abs(gray_image[y, x] - gray_image[y - 1, x])
        GradientX[y, x] = np.abs(gray_image[y, x] - gray_image[y, x - 1])

fig, ax = plt.subplots()
ax.imshow(GradientX, interpolation='none')
fig, ax = plt.subplots()
ax.imshow(GradientY, interpolation='none')
plt.show()
# Of course for loop takes more time in python as it's an interpreted language
