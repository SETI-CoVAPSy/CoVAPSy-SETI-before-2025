import numpy as np
import cv2
from KMeans import KMeans

image = cv2.imread("./test.jpg")
image = cv2.resize(image, (640, 480),
                   interpolation=cv2.INTER_AREA)
get_origine_resize = 480 // 2, 640 // 2
image = image[get_origine_resize[0]:get_origine_resize[0] + (
        get_origine_resize[0] * 2) // 3, :, :]
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
cv2.imwrite("./sortie_brute.png", image)
pixels = image.reshape((-1, 3))
liste_essais = [KMeans(3) for _ in range(10)]
for km in liste_essais:
        km.fit(pixels)
liste_energie = [km.get_energy(pixels) for km in liste_essais]
kmeans = liste_essais[np.argmin(liste_energie)]
dominant_cluster = np.argmax(np.bincount(kmeans.predict(
pixels.reshape((-1, 3)))))
labels = kmeans.predict(pixels.reshape((-1, 3)))
cv2.imwrite("./sortie_kmeans.png", np.array((dominant_cluster == labels) * 255, dtype=np.uint8).reshape(image.shape[:2]))
