import math
import numpy as np
from sklearn.decomposition import TruncatedSVD
from KMeans import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import cv2
from abc import ABC, abstractmethod
import os
from joblib import Parallel, delayed


class ControleurVisionAbstract(ABC):

    @staticmethod
    @abstractmethod
    def get_height_camera_orig():
        pass

    @staticmethod
    @abstractmethod
    def get_width_camera_orig():
        pass

    @staticmethod
    @abstractmethod
    def get_fov_camera_orig():
        pass

    @staticmethod
    @abstractmethod
    def get_height_camera_resize():
        pass

    @staticmethod
    @abstractmethod
    def get_width_camera_resize():
        pass

    @staticmethod
    @abstractmethod
    def get_reglage_portee_transformation():
        pass

    @staticmethod
    @abstractmethod
    def get_reglage_angle_transformation():
        pass

    @staticmethod
    @abstractmethod
    def get_reglage_decalage_y():
        pass

    @staticmethod
    @abstractmethod
    def get_facteur_correction_angle():
        pass

    @staticmethod
    @abstractmethod
    def get_vitesse_recul():
        pass

    @staticmethod
    @abstractmethod
    def get_seuil_couleurs():
        pass

    @staticmethod
    @abstractmethod
    def get_seuil_recul():
        pass

    @staticmethod
    @abstractmethod
    def get_recul_nombre():
        pass

    @staticmethod
    @abstractmethod
    def get_throttle_recul_nombre():
        pass

    @staticmethod
    @abstractmethod
    def get_vitesse_max():
        pass

    @staticmethod
    @abstractmethod
    def get_angle_max():
        pass

    @abstractmethod
    def __init__(self):
        self.recule_jusqua = 0
        self.throttle_recule_jusqua = 0
        self.last_direction = 0
        self.image_prec = None
        self.kmeans = None
        self._init_pca()
        self._init_lda()
        self._init_remap()

    @abstractmethod
    def get_image(self):
        pass

    @classmethod
    def get_origine_resize(cls):
        return cls.get_height_camera_resize() // 2, cls.get_width_camera_resize() // 2

    def _init_pca(self):
        self.pca = TruncatedSVD(n_components=1, n_iter=1)

    def _init_lda(self):
        self.lda = LinearDiscriminantAnalysis(n_components=1)

    def _init_remap(self):
        focal_length = self.get_height_camera_orig() / (2 * math.tan(self.get_fov_camera_orig() / 2))

        def transformation(x_uncentered, y):
            new_x = ((x_uncentered - self.get_origine_resize()[1]) / (1 + y)) * self.get_origine_resize()[1] + \
                    self.get_origine_resize()[1]
            new_y = (1 / (1 + y)) * self.get_origine_resize()[
                0] * focal_length / self.get_reglage_portee_transformation()
            return new_x, new_y

        self.map_x = np.zeros((self.get_height_camera_resize(), self.get_width_camera_resize()), dtype=np.float32)
        self.map_y = np.zeros((self.get_height_camera_resize(), self.get_width_camera_resize()), dtype=np.float32)
        for y in range(self.get_height_camera_resize()):
            for x_uncentered in range(self.get_width_camera_resize()):
                new_x, new_y = transformation(x_uncentered, y)
                if 0 <= new_x < self.get_width_camera_resize() and 0 <= new_y < self.get_height_camera_resize() / 2:
                    self.map_x[y, x_uncentered] = new_x
                    self.map_y[y, x_uncentered] = new_y
                else:
                    self.map_x[y, x_uncentered] = 0
                    self.map_y[y, x_uncentered] = 0

    def _make_pca(self, X):
        if X.shape[0] > 10:
            self.pca.fit(X)
            var, axe = self.pca.explained_variance_[0], self.pca.components_[0]
        else:
            var, axe = 0, 0
        try:
            angle_degre = np.degrees(np.arctan2(axe[0], axe[1])) - 90
        except TypeError:
            angle_degre = 0
        return var, angle_degre

    def _rendre_plus_convexe(self, image):
        enveloppe_convexe_contour, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(enveloppe_convexe_contour) == 0:
            return image
        enveloppe_convexe_contour = cv2.convexHull(np.vstack(enveloppe_convexe_contour))
        partie_non_convexe = np.zeros_like(image)
        cv2.drawContours(partie_non_convexe, [enveloppe_convexe_contour], -1, 255, thickness=cv2.FILLED)
        partie_non_convexe[image > 0] = 0
        partie_non_convexe = np.argwhere(partie_non_convexe > 125).astype(np.float32)
        partie_image = np.argwhere(image > 125).astype(np.float32)
        if len(partie_non_convexe) < 10:
            return image
        self.lda.fit(np.vstack([partie_non_convexe, partie_image]),
                     np.hstack([np.ones(len(partie_non_convexe), np.int32),
                                np.zeros(len(partie_image))]))
        h, w = image.shape
        h, w = np.meshgrid(np.arange(w), np.arange(h))
        coords = np.vstack([h.ravel(), w.ravel()]).T.astype(np.float32)
        responses = self.lda.predict(coords).reshape(image.shape)
        image[responses > 0.5] = 0
        return image

    def _transforme_image(self, image):
        image = cv2.remap(image, self.map_x, self.map_y, interpolation=cv2.INTER_NEAREST)
        if image[1, 1] == 0:
            self.image_prec = image
        elif self.image_prec is not None:
            image = self.image_prec
            # Parfois le remap bug et c'est détectable car ça contamine l'arrière plan
        shape = image.shape
        image = cv2.resize(image, (
            int(shape[1] / 10 * self.get_reglage_angle_transformation()),
            shape[0] // 10
        ),
                           interpolation=cv2.INTER_AREA)
        image = cv2.erode(image,
                          kernel=np.ones((1, 2), dtype=np.uint8),
                          iterations=2)
        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        return image

    @abstractmethod
    def set_vitesse_m_s(self, vitesse_m_s):
        pass

    @abstractmethod
    def set_direction_degre(self, angle_degre):
        pass

    def recule(self):
        if self.throttle_recule_jusqua == 0:
            self.recule_jusqua = self.get_recul_nombre()
            self.throttle_recule_jusqua = self.get_throttle_recul_nombre()
            if self.last_direction > 0:
                self.set_direction_degre(-max(self.get_angle_max(), self.last_direction))
            else:
                self.set_direction_degre(-min(-self.get_angle_max(), self.last_direction))
            self.set_vitesse_m_s(-self.get_vitesse_recul())
            return True
        else:
            return False

    def _traitement_kmeans(self, image):
        pixels = image.reshape((-1, 3))
        N_KMEANS = 10
        if self.kmeans is None:
            self.kmeans = [KMeans(pixels.shape[0]) for _ in range(N_KMEANS)]
            for i in range(N_KMEANS):
                if os.path.exists(f"kmeans{i}.save"):
                    self.kmeans[i].load(f"kmeans{i}.save")
                else:
                    self.kmeans[i].fit(pixels)
                    self.kmeans[i].save(f"kmeans{i}.save")
            labels = [self.kmeans[i].predict(pixels.reshape((-1, 3))) for i in range(N_KMEANS)]
        else:
            labels = [self.kmeans[i].predict_and_adapt(pixels) for i in range(N_KMEANS)]
        preds = [(np.array((self.kmeans[i].get_dominant_cluster() == labels[i]) * 255, dtype=np.uint8)).reshape(image.shape[:2]) for i in range(N_KMEANS)]
        res =  np.mean(np.stack(preds, axis=0), axis=0)
        cv2.imwrite("/dev/shm/test_kmeans.png", res)
        return res

    def _traitement_floodfill(self, image):
        tolerance = (64, 64, 64)
        nb_masks = 10
        connectivity = 4
        anchor_bottom_margin = 0.1
        floodfill_flags = (connectivity | cv2.FLOODFILL_MASK_ONLY | (int(255/nb_masks) << 8) | cv2.FLOODFILL_FIXED_RANGE)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.multiply(hsv[..., 1], 3)  # Increase saturation
        image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        alpha = 2  # Contrast control
        beta = 0   # Brightness control
        image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
        h, w = image.shape[:2]
        mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        margin = int(h * anchor_bottom_margin)
        y = h - margin

        def tache_floodfill(i):
            x = int(w * (i + 0.5) / nb_masks)
            mask_copy = mask.copy()
            cv2.floodFill(image, mask=mask_copy, seedPoint=(x, y), newVal=0, loDiff=tolerance, upDiff=tolerance, flags=floodfill_flags)
            return mask_copy[1:-1, 1:-1]

        masks = [tache_floodfill(i) for i in range(nb_masks)]
        res = np.sum(masks, axis=0, dtype=np.uint8)
        cv2.imwrite("/dev/shm/test_floodfill.png", res)
        return res

    def _traitement_statique_gris(self, image):
        res = cv2.inRange(image,
                          np.array([0, 0, 50]),
                          np.array([200, 130, 160]))
        cv2.imwrite("/dev/shm/test_statique_gris.png", res)
        return res

    def _traitement_statique_ni_rouge_ni_vert(self, image):
        vert = cv2.inRange(image,
        		   np.array([20, 100, 50]),
        		   np.array([80, 255, 255]))
        rouge = cv2.inRange(image,
        		    np.array([0, 100, 50]),
        		    np.array([20, 255, 255]))
        rouge = cv2.bitwise_or(rouge,
                               cv2.inRange(image,
                               		   np.array([160, 100, 50]),
                               		   np.array([180, 255, 255])))
        res = cv2.bitwise_not(cv2.bitwise_or(vert, rouge))
        cv2.imwrite("/dev/shm/test_vert.png", vert)
        cv2.imwrite("/dev/shm/test_rouge.png", rouge)
        cv2.imwrite("/dev/shm/test_statique_ni_rouge_ni_vert.png", res)
        return res

    def _pretraitement_image(self, image):
        image = cv2.resize(image, (self.get_width_camera_resize(), self.get_height_camera_resize()),
                           interpolation=cv2.INTER_AREA)
        cv2.imwrite("/dev/shm/test_image.png", image)
        image = image[self.get_origine_resize()[0]:self.get_origine_resize()[0] + (
                self.get_origine_resize()[0] * 2) // 3, :, :]
        image = cv2.pyrMeanShiftFiltering(image, 10, 50)
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        propositions_masque_sol = Parallel(n_jobs=4, backend="threading")(
            delayed(func)(hsv_image) for func in [
                self._traitement_kmeans,
                #self._traitement_floodfill,
                self._traitement_statique_gris,
                #self._traitement_statique_ni_rouge_ni_vert
            ]
        )
        masque_sol = np.mean(np.stack(propositions_masque_sol, axis=0), axis=0)
        masque_sol = (masque_sol > 200).astype(np.uint8) * 255
        cv2.imwrite("/dev/shm/test_image_restreinte.png", image)
        cv2.imwrite("/dev/shm/test_fusion.png", masque_sol)
        masque_sol = cv2.morphologyEx(masque_sol, cv2.MORPH_OPEN, np.ones((10, 10), np.uint8))
        hsv_image_devant = hsv_image[20:, (self.get_origine_resize()[1] + 2):(self.get_origine_resize()[1] + 5)]
        compte_vert = np.count_nonzero(cv2.inRange(hsv_image_devant,
                                                   np.array([20, 100, 50]),
                                                   np.array([80, 255, 255])))
        compte_rouge = cv2.inRange(hsv_image_devant,
                                   np.array([0, 100, 50]),
                                   np.array([20, 255, 255]))
        compte_rouge = cv2.bitwise_or(compte_rouge,
                                      cv2.inRange(hsv_image_devant,
                                                  np.array([160, 100, 50]),
                                                  np.array([180, 255, 255])))
        compte_rouge = np.count_nonzero(compte_rouge)
        masque_sol = self._transforme_image(masque_sol)
        cv2.imwrite("/dev/shm/test_carte.png", masque_sol)
        return masque_sol, compte_rouge, compte_vert

    def _image_vers_points(self, masque_sol):
        points_sol = np.argwhere(masque_sol == 255)
        points_sol[:, 1] -= masque_sol.shape[1] // 2
        points_sol[:, 0] += self.get_reglage_decalage_y()
        return points_sol

    @abstractmethod
    def run(self):
        pass

    def run_iteration(self):
        image = self.get_image()
        masque_sol, compte_rouge, compte_vert = self._pretraitement_image(image)
        masque_sol = self._rendre_plus_convexe(masque_sol)
        points_carte = self._image_vers_points(masque_sol)

        if self.recule_jusqua == 0:
            if self.throttle_recule_jusqua > 0:
                self.throttle_recule_jusqua -= 1
            var, angle_degre = self._make_pca(points_carte)
            vitesse_m_s = np.sqrt(var / 70)

            if vitesse_m_s < self.get_seuil_recul():
                if self.recule():
                    return
            print(f"vitesse guidante {vitesse_m_s}")

            angle_degre *= self.get_facteur_correction_angle()

            if vitesse_m_s < self.get_seuil_couleurs():
                if compte_rouge > 10:
                    angle_degre = -self.get_angle_max()
                elif compte_vert > 10:
                    angle_degre = self.get_angle_max()
            vitesse_m_s = min(vitesse_m_s, 1) * self.get_vitesse_max()
            vitesse_m_s = max(vitesse_m_s, self.get_vitesse_max() / 2)

            self.set_direction_degre(angle_degre)
            self.set_vitesse_m_s(vitesse_m_s)
            self.last_direction = angle_degre
        else:
            self.recule_jusqua -= 1
