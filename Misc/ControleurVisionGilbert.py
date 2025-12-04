from vehicle import Driver
from CameraGilbert import CameraGilbert
from ControleurVisionAbstract import ControleurVisionAbstract
from rpi_hardware_pwm import HardwarePWM
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
        return 1.2

    @staticmethod
    def get_vitesse_recul():
        return 0.05

    @staticmethod
    def get_seuil_couleurs():
        return 0.25

    @staticmethod
    def get_seuil_recul():
        return 0.025

    def set_vitesse_m_s(self, vitesse_m_s):
        if vitesse_m_s > self.vitesse_max_m_s_soft:
            vitesse_m_s = self.vitesse_max_m_s_soft
        elif vitesse_m_s < -self.vitesse_max_m_s_hard:
            vitesse_m_s = -self.vitesse_max_m_s_hard

        vitesse = vitesse_m_s * self.delta_pwm_max_prop / self.vitesse_max_m_s_hard
        if vitesse_m_s == 0:
            new_duty_cycle = self.pwm_stop_prop
        elif vitesse_m_s > 0:
            new_duty_cycle = self.pwm_stop_prop + self.direction_prop * (self.point_mort_prop + vitesse)
        else:
            new_duty_cycle = (self.pwm_stop_prop - self.direction_prop * (self.point_mort_prop * 2 - vitesse))

        self.pwm_prop.change_duty_cycle(new_duty_cycle)
        print("Speed:", new_duty_cycle, "Speed (m/s): ", vitesse_m_s)

    def set_direction_degre(self, angle_degre):
        #angle_degre *= -1
        angle_pwm = self.angle_pwm_centre + self.direction * (self.angle_pwm_max - self.angle_pwm_min) * angle_degre / (
                2 * self.angle_degre_max)
        if angle_pwm > self.angle_pwm_max:
            angle_pwm = self.angle_pwm_max
        if angle_pwm < self.angle_pwm_min:
            angle_pwm = self.angle_pwm_min
        self.pwm_dir.change_duty_cycle(angle_pwm)
        print("Angle:", angle_pwm, "Angle degrés: ", angle_degre)

    def get_image(self):
        image = self.camera.getImage()
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

    def __init__(self, video=False):
        super().__init__()
        self._init_constantes()
        self.pwm_dir = HardwarePWM(pwm_channel=1, hz=50)
        self.pwm_prop = HardwarePWM(pwm_channel=0, hz=50)

        self.current_speed = 0
        self.current_angle = 0
        self.pwm_dir.start(self.angle_pwm_centre)
        self.pwm_prop.start(self.pwm_stop_prop)
        self._init_camera()
        
        if video:
            self.video = ffmpeg_video.FfmpegVideo(f"{time.time():.0f}_{datetime.isoformat(datetime.now()).replace(':', '-').split('.')[0]}.mjpg")
        else:
            self.video = None

    def run(self):
        while True:
            try:
                super().run_iteration()
            except KeyboardInterrupt:
                print("Stopping")
                self.pwm_dir.stop()
                self.pwm_prop.stop()
                self.camera.stop()
                if self.video: self.video.stop()
                break

controller = ControleurVisionGilbert(video=False)
controller.run()
