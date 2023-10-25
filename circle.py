import cv2 as cv
import sys
from matplotlib import pyplot as plt
import numpy as np
import math
import csv
import os

sys.stdout = open(r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\output.txt.txt", "w")


def imageshow(img):
    cv.namedWindow(
        "output", cv.WINDOW_NORMAL
    )  # Create window with freedom of dimensions # Read image
    imS = cv.resize(img, (2000, 2000))  # Resize image
    cv.imshow("output", imS)  # Show image
    cv.waitKey(0)


name = "DBM-11-13.5 cm by 8.5 cm"
path = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\bina" + "temp" + ".csv"
ipath = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\ipath" + name + ".jpg"
gra = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\gra"
bina = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\bina"
cont = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\cont"
circ = r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\circ"
speciman_area = 17662
print(speciman_area)

img1 = cv.imread(r"C:\Users\singh\OneDrive\Desktop\MTP\practice\unedited.png")
if img1 is None:
    sys.exit("Could not read the image.")
# imageshow(img1)
# cv.namedWindow("output", cv.WINDOW_NORMAL) # Create window with freedom of dimensions # Read image
# imS = cv.resize(img1, (2000, 2000)) # Resize image
# cv.imshow("output", imS) # Show image
# cv.waitKey(0)
# plt.imshow(img1)

gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
figure_size = 3
# gray = cv.Laplacian(gray, cv.CV_16S, figure_size)
gray = cv.medianBlur(gray, figure_size)
# plt.imshow(gray)
# print(gray)
imageshow(gray)
# os.chdir(gra)
cv.imwrite(name + ".jpg", gray)
hist1 = cv.calcHist([gray], [0], None, [256], [0, 256])
# plt.hist(gray.ravel(),256,[0,256]); plt.show()
equ1 = cv.equalizeHist(gray)
clahe = cv.createCLAHE(clipLimit=12.0, tileGridSize=(8, 8))
equ1 = clahe.apply(gray)
equ1 = cv.bilateralFilter(equ1, 2, 200, 200)
imageshow(equ1)
hist = cv.calcHist([equ1], [0], None, [256], [0, 256])
# print("Histogram for automatic equalise histogram")
# plt.hist(equ1.ravel(),256,[0,256]);
# plt.show()
# plt.hist(gray.ravel(),256,[0,256]);
# plt.show()

ret, th = cv.threshold(equ1, 130, 255, cv.THRESH_BINARY)
kernel = np.ones((5, 5), np.uint8)
th = cv.erode(th, kernel, iterations=2)
# th = cv.dilate(th, kernel, iterations=1)
th = cv.medianBlur(th, figure_size)
# imageshow(th)
# plt.imshow(th)
ret1, white = cv.threshold(equ1, 0, 255, cv.THRESH_BINARY)
os.chdir(bina)
cv.imwrite(name + ".jpg", th)
# opening = cv.morphologyEx(gray, cv.MORPH_OPEN, kernel,iterations=1) #opening
# sure_bg = cv.erode(opening,kernel,iterations = 1)
# dist_tr = cv.distanceTransform(opening,cv.DIST_L2,0)
# ret1,th1 = cv.threshold(dist_tr,0.1*dist_tr.max(),255,0)
# th = np.uint8(th1)
# imageshow(th)
contours, hierarchy = cv.findContours(th, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
img2 = equ1.copy()
img3 = equ1.copy()
area = [0]
count = 1
for cot in contours:
    # M = cv.moments(cot)
    # print (M['m00'])
    # cx = int (M['m10']/M['m00'])
    # cy = int (M['m01']/M['m00'])
    # center = (cx,cy)
    epsilon = 0.0001 * cv.arcLength(cot, True)
    approx = cv.approxPolyDP(cot, epsilon, True)
    img = cv.drawContours(img2, [approx], -1, (0, 255, 50), 2)
    area1 = cv.contourArea(cot)
    # area.append(area1)
    (x, y), radius = cv.minEnclosingCircle(cot)
    center = (int(x), int(y))
    radius = int(radius)
    cv.circle(img3, center, radius, (0, 255, 50), 2)
    area.append((radius**2) * 3.14)
    count = count + 1
# imageshow(img2)
imageshow(img3)
os.chdir(cont)
cv.imwrite(name + ".jpg", img2)
os.chdir(circ)
cv.imwrite(name + ".jpg", img3)
area.sort()

# n_white_pix_a = np.sum(th==255) #voids area
# print('Number of white pixels aggregate:', n_white_pix_a)
dimensions = gray.shape
print(dimensions)
print("Height of image", dimensions[0])
print("width of image", dimensions[1])
# 5582.37
number_of_total_pixels = np.sum(white == 255)
number_of_white_pixels = np.sum(th == 255)
number_of_black_pixels = np.sum(th == 0)

# print("number_of_total_pixels->",number_of_total_pixels)

print("number_of_white_pixels->", number_of_white_pixels)
print("number_of_black_pixels->", number_of_black_pixels)
print("total number of pixels->", number_of_total_pixels)
ratio_of_arrgregates_to_voids = number_of_black_pixels / number_of_total_pixels
print("ratio_of_aggregates->", ratio_of_arrgregates_to_voids)

mm_square_per_pixel = speciman_area / number_of_total_pixels
area_of_a = number_of_white_pixels * mm_square_per_pixel
print("area of aggregate:", area_of_a)
final_area = [0]
for a in area:
    final_area.append(a * mm_square_per_pixel)
# print (final_area)
diameter = []
mass = []
cumulative_area = []

cumulative_percentage = []
sum = 0
for label in final_area:
    diameter.append(math.sqrt(label * 4 / 3.14))
    mass.append(label)
batch = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# print(diameter)
# print(mass)
total = 0
for i in range(0, len(area) + 1):
    curr_radius = diameter[i]
    m = mass[i]
    total = total + m
    a = 1
    if curr_radius < 26.5:
        batch[1] = batch[1] + a
    if curr_radius < 19:
        batch[2] = batch[2] + a
    if curr_radius < 13.2:
        batch[3] = batch[3] + a
    if curr_radius < 9.5:
        batch[4] = batch[4] + a
    if curr_radius < 4.75:
        batch[5] = batch[5] + a
    if curr_radius < 2.36:
        batch[6] = batch[6] + a
    if curr_radius < 1.18:
        batch[7] = batch[7] + a
    if curr_radius < 0.6:
        batch[8] = batch[8] + a
    if curr_radius < 0.3:
        batch[9] = batch[9] + a
    if curr_radius < 0.15:
        batch[10] = batch[10] + a
    if curr_radius < 0.075:
        batch[11] = batch[11] + a

total = batch[1]
for i in range(0, 12):
    batch[i] = (batch[i] / total) * 100


batch[0] = 100
print(batch)
seive = [36, 26.5, 19, 13.2, 9.5, 4.75, 2.36, 1.18, 0.6, 0.3, 0.15, 0.075]
# print(diameter)
cumulative_percentage = np.array(cumulative_percentage)
area = np.array(area)
# print (cumulative_percentage,area)
print(seive)
# print("HAHA")
# print(len(area),len(cumulative_percentage))
plt.plot(seive, batch)
plt.xlabel("area log scale")
plt.ylabel("cumulative percentage")
plt.grid(True)
plt.xscale("log")
plt.title("Cumulative Passing vs Size")
plt.show()
# plt.savefig(r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\Graphs" + "\rectangular.png")
os.chdir(gra)
plt.savefig("books_read.png")
# plt.show()
with open(path, "w", newline="") as f:
    filename = [["area of a(mm2)"]]
    writer = csv.writer(f)
    writer.writerows(filename)
    writer.writerows(map(lambda x: [x], batch))

# with open(r"C:\Users\singh\OneDrive\Desktop\MTP\NEw\data.csv", 'w',encoding='utf-8') as f:
#     writer = csv.writer(f, delimiter='HAHA')
#     for i in
#     # c1 = my_sheet.cell(row = 1, column = 1)
#     writer.writerows(zip(batch,seive))
