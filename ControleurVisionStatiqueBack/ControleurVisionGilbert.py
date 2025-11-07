from gilbertvehicle import Driver
from CameraGilbert import CameraGilbert
from ControleurVisionAbstract import ControleurVisionAbstract
import time, ffmpeg_video
from datetime import datetime


class ControleurVisionGilbert(ControleurVisionAbstract):
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
        return 40 #20

    @staticmethod
    def get_facteur_correction_angle():
        return 1.5

    @staticmethod
    def get_vitesse_recul():
        return 0.05

    @staticmethod
    def get_seuil_couleurs():
        return 0.3

    @staticmethod
    def get_seuil_recul():
        return 0.01

    @staticmethod
    def get_recul_nombre():
        return 30

    @staticmethod
    def get_throttle_recul_nombre():
        return 80

    @staticmethod
    def get_vitesse_max():
        return 2.0

    @staticmethod
    def get_angle_max():
        return 20

    def get_image(self):
        image = self.camera.getImage()
        if image is None:
            raise Exception("Pas d'acquisition d'image")
        if self.video:
            self.video.add_opencv_image(image)
        image = image.reshape(
            (self.get_height_camera_orig(), self.get_width_camera_orig(), 3))
        return image

    def _init_driver(self):
        self.driver = Driver()
        self.driver.setSteeringAngle(0)
        self.driver.setCruisingSpeed(0)

    def _init_camera(self):
        self.camera = CameraGilbert("camera")
        self.camera.enable(30)
        for i in range(0,200):
            self.get_image()
        input("Appuyer sur entrée pour commencer la course")

    def _init_constantes(self):
        self.direction_prop = 1  # -1 pour les variateurs inversés ou un petit rapport correspond à une marche avant
        self.pwm_stop_prop = 7.6
        self.point_mort_prop = 0.21
        self.delta_pwm_max_prop = 1.5  # pwm à laquelle on atteint la vitesse maximale

        self.vitesse_max_m_s_hard = 8  # vitesse que peut atteindre la voiture
        self.vitesse_max_m_s_soft = 0.5  # vitesse maximale que l'on souhaite atteindre

        # paramètres de départ, avec des butées très proche du centre
        self.direction = -1  # 1 pour angle_pwm_min a gauche, -1 pour angle_pwm_min à droite
        self.angle_pwm_min = 6.5  # min
        self.angle_pwm_max = 10  # max
        self.angle_pwm_centre = 8.8

        self.angle_degre_max = +18  # vers la gauche
        self.angle_degre = 0
        
    def set_direction_degre(self, angle):
        self.driver.set_direction_degre(angle)
        print(f"Angle: {angle}")
        
    def set_vitesse_m_s(self, vitesse):
        self.driver.set_vitesse_m_s(vitesse)
        print(f"Vitesse: {vitesse}")

    def __init__(self, video=False):
        super().__init__()
        
        self.driver = Driver()
        
        
        self._init_constantes()

        
        if video:
            self.video = ffmpeg_video.FfmpegVideo(f"{time.time():.0f}_{datetime.isoformat(datetime.now()).replace(':', '-').split('.')[0]}.mjpg")
        else:
            self.video = None
        self._init_camera()

    def run(self):
        while True:
            try:
                super().run_iteration()
            except KeyboardInterrupt:
                print("Stopping")
                self.set_direction_degre(0)
                self.set_vitesse_m_s(0)
                self.driver.stop()
                self.camera.stop()
                if self.video: self.video.stop()
                break

controller = ControleurVisionGilbert(video=False)
controller.run()
