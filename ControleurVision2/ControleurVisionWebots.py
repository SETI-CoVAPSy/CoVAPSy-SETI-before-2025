from vehicle import Driver
from controller import Camera
from ControleurVisionAbstract import ControleurVisionAbstract

import numpy as np


class ControleurVisionWebots(ControleurVisionAbstract):
    @staticmethod
    def get_height_camera_orig():
        return 1232

    @staticmethod
    def get_width_camera_orig():
        return 1640

    @staticmethod
    def get_fov_camera_orig():
        return 1.085

    @staticmethod
    def get_height_camera_resize():
        return 480

    @staticmethod
    def get_width_camera_resize():
        return 640

    @staticmethod
    def get_reglage_portee_transformation():
        return 10

    @staticmethod
    def get_reglage_angle_transformation():
        return 0.6

    @staticmethod
    def get_reglage_decalage_y():
        return 40  # 20

    @staticmethod
    def get_facteur_correction_angle():
        return 1.2

    @staticmethod
    def get_vitesse_recul():
        return 0.5

    @staticmethod
    def get_seuil_couleurs():
        return 0.25

    @staticmethod
    def get_seuil_recul():
        return 0.01

    @staticmethod
    def get_recul_nombre():
        return 40

    @staticmethod
    def get_throttle_recul_nombre():
        return 60

    @staticmethod
    def get_vitesse_max():
        return 1.6

    @staticmethod
    def get_angle_max():
        return 20

    def set_vitesse_m_s(self, vitesse_m_s):
        self.driver.setCruisingSpeed(vitesse_m_s * 3.6)

    def set_direction_degre(self, angle_degre):
        self.driver.setSteeringAngle(-angle_degre * 3.14 / 180)

    def get_image(self):
        image = self.camera.getImage()
        image = np.frombuffer(image, dtype=np.uint8).reshape(
            (self.get_height_camera_orig(), self.get_width_camera_orig(), 4))
        return image[:, :, :3]

    def _init_driver(self):
        self.driver = Driver()
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)

    def _init_camera(self):
        self.camera = Camera("camera")
        self.camera.enable(30)

    def __init__(self):
        super().__init__()
        self._init_driver()
        self._init_camera()

    def run(self):
        while self.driver.step() != -1:
            self.run_iteration()


if __name__ == '__main__':
    controller = ControleurVisionWebots()
    controller.run()
