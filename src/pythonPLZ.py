from matplotlib import pyplot as plt
from matplotlib.image import imread
img = imread('./picture.jpeg')

plt.imshow(img)
plt.show()