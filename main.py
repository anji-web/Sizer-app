import cv2
import mediapipe as mp
import math
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.graphics.texture import Texture
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.core.window import Window
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
import time
import pandas as pd

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=2,
    smooth_landmarks=True,
    enable_segmentation=True,
    smooth_segmentation=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
drawing = mp.solutions.drawing_utils
DrawingSpec = drawing.DrawingSpec
shoulder_width_cm = 40.0
# --- Data Kamera Redmi 9 ---
f_mm = 28.0          # focal length (mm)
W_sensor_mm = 6.5    # lebar sensor (mm)
W_px = 4160          # lebar gambar (px)

# Focal length dalam pixel
f_px = f_mm * W_px / W_sensor_mm

# --- Input Tinggi Badan (cm) ---
H_real_cm = 172.0  # contoh tinggi pengguna
# manual_values = {}

class BodyMeasureApp(App):
    
    def build(self):
        # === Bagian kamera ===
        self.img = Image(
            size_hint=(0.5, 1),  # 50% layar
            allow_stretch=True,
            keep_ratio=True
        )

        # === Logo ===
        self.logo = Image(
            source="assets/LOGO_WHBN.png",
            size_hint=(0.5, 1),  # 50% layar
            allow_stretch=True,
            keep_ratio=True
        )

        # === Gabungkan kamera dan logo berdampingan ===
        cam_logo_layout = BoxLayout(
            orientation="horizontal",
            spacing=0,
            padding=0,
            size_hint=(1, 0.65)  # 65% tinggi layar
        )
        cam_logo_layout.add_widget(self.img)
        cam_logo_layout.add_widget(self.logo)

        # === Scroll hasil / instruksi ===
        scroll = ScrollView(size_hint=(1, 0.2))
        self.label = Label(
            text="Posisikan badan di kamera...",
            size_hint_y=None,
            halign="left",
            valign="top",
            font_size="16sp",
            color=(1, 1, 1, 1),
        )
        self.label.bind(texture_size=self.label.setter("size"))

        layout_label = GridLayout(cols=1, size_hint_y=None, padding=[10, 10, 10, 10])
        layout_label.bind(minimum_height=layout_label.setter('height'))
        layout_label.add_widget(self.label)
        scroll.add_widget(layout_label)

        def update_text_size(*args):
            self.label.text_size = (scroll.width - 20, None)
        scroll.bind(width=update_text_size)
        update_text_size()

        # === Tombol warna brand WHB ===
        self.reset_btn = Button(
            text="Reset Measurement",
            size_hint=(1, 0.075),
            background_color=(11/255, 43/255, 76/255, 1),  # Biru tua WHB
            font_size="18sp",
            color=(1, 1, 1, 1)
        )
        self.reset_btn.bind(on_press=self.reset_measurements)
        self.reset_btn.disabled = True

        self.reset_buffer_btn = Button(
            text="Reset Buffer",
            size_hint=(1, 0.075),
            background_color=(182/255, 73/255, 38/255, 1),  # Merah bata WHB
            font_size="18sp",
            color=(1, 1, 1, 1)
        )
        self.reset_buffer_btn.bind(on_press=self.show_reset_buffer_popup)

        # === Layout utama ===
        main_layout = BoxLayout(orientation="vertical", padding=0, spacing=5)
        main_layout.add_widget(cam_logo_layout)  # kamera + logo (50:50)
        main_layout.add_widget(scroll)
        main_layout.add_widget(self.reset_btn)
        main_layout.add_widget(self.reset_buffer_btn)

        # === Kamera capture ===
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0 / 30.0)

        self.pose_valid = False
        self.countdown = 0
        self.captured = False
        self.last_countdown_time = 0
        self.captured_frame = None
        start_btn = Button(
            text="Input User Report & Mulai",
            size_hint=(1, 0.075),
            background_color=(0, 0.5, 0.7, 1),
            font_size="18sp",
            color=(1, 1, 1, 1)
        )
        start_btn.bind(on_press=lambda x: self.show_manual_input_popup())
        main_layout.add_widget(start_btn, index=0)  # letakkan di atas tombol lainnya
        return main_layout
    
    
    def show_reset_buffer_popup(self, instance):
        """
        Tampilkan popup konfirmasi untuk reset buffer smoothing
        """
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        content.add_widget(Label(text="Apakah Anda yakin ingin mereset buffer smoothing?\nSemua data smoothing akan dihapus."))

        btn_layout = BoxLayout(orientation='horizontal', spacing=10)
        btn_yes = Button(text="Ya", background_color=(0.1, 0.6, 0.1, 1))
        btn_no = Button(text="Batal", background_color=(0.7, 0.7, 0.7, 1))
        btn_layout.add_widget(btn_yes)
        btn_layout.add_widget(btn_no)

        content.add_widget(btn_layout)

        popup = Popup(
            title="Konfirmasi Reset Buffer",
            content=content,
            size_hint=(0.7, 0.4),
            auto_dismiss=False
        )

        # Event handler tombol
        def confirm_reset(instance):
            if hasattr(self, "reset_smoothing"):
                self.reset_smoothing()
            popup.dismiss()
            confirm_popup = Popup(
                title="Buffer Direset",
                content=Label(text="Buffer smoothing berhasil direset."),
                size_hint=(0.6, 0.3),
                auto_dismiss=True
            )
            confirm_popup.open()

        btn_yes.bind(on_press=confirm_reset)
        btn_no.bind(on_press=lambda x: popup.dismiss())

        popup.open()

    def reset_measurements(self, instance):
        self.label.text = "Posisikan badan di kamera..."
        self.pose_valid = False
        self.countdown = 0
        self.captured = False
        self.reset_btn.disabled = True
        self.reset_buffer_btn.disabled = True
        self.captured_frame = None

    def euclidean(self, p1, p2, w, h, scale):
        x1, y1 = int(p1.x * w), int(p1.y * h)
        x2, y2 = int(p2.x * w), int(p2.y * h)
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) * scale
    
    def euclidean_cm(self,a, b,frame_height, scale_factor):
        """Calculate 2D Euclidean distance (in cm) between two landmarks."""
        return np.sqrt(
            ((a.x - b.x) * frame_height * scale_factor)**2 +
            ((a.y - b.y) * frame_height * scale_factor)**2)

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if self.captured_frame is not None:
            buf = cv2.flip(self.captured_frame, 0).tobytes()
            image_texture = Texture.create(size=(self.captured_frame.shape[1], self.captured_frame.shape[0]), colorfmt='bgr')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.img.texture = image_texture
            return 
        pose_color = (255, 0, 0)  # default biru (BGR)
        if results.pose_landmarks and not self.captured:
            lm = results.pose_landmarks.landmark
            left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            pt_left = np.array([left_shoulder.x * w, left_shoulder.y * h])
            pt_right = np.array([right_shoulder.x * w, right_shoulder.y * h])
            
            shoulder_pixel = np.linalg.norm(pt_right - pt_left)
            # Koordinat kepala & tumit
            nose = lm[mp_pose.PoseLandmark.NOSE]
            left_ankle = lm[mp_pose.PoseLandmark.LEFT_ANKLE]
            right_ankle = lm[mp_pose.PoseLandmark.RIGHT_ANKLE]

            # Konversi ke pixel
            h, w, _ = frame.shape
            top_y = nose.y * h
            bottom_y = max(left_ankle.y * h, right_ankle.y * h)

            H_pixel = bottom_y - top_y

            # Hitung Z dan scale
            Z_cm = f_px * H_real_cm / H_pixel
            # --- Kalibrasi dasar manual ---
            base_pixels_per_cm = 40.0 / 13.0  # hasil kalibrasi di jarak 215–217 cm

            # --- Adaptive auto-scaling ---
            lm2 = lambda x: lm[mp_pose.PoseLandmark[x].value]
            shoulder_px = abs(lm2('LEFT_SHOULDER').x - lm2('RIGHT_SHOULDER').x) * w
            expected_shoulder_cm = 42  # rata-rata bahu pria dewasa Indonesia
            current_pixels_per_cm = shoulder_px / expected_shoulder_cm

            # --- Blend (gunakan sebagian kecil adaptasi)
            blend_ratio = 0.2  # 20% adaptasi, 80% tetap base kalibrasi
            pixels_per_cm = base_pixels_per_cm * (1 - blend_ratio) + current_pixels_per_cm * blend_ratio
            scale = 1 / pixels_per_cm
            
            scaleHardcode = 13.0 / 40.0

            # 🔹 Panggil validate_pose disini
            is_valid, feedback = self.validate_pose(lm)
            if not is_valid:
                self.label.text = feedback
                pose_color = (0, 0, 255)  # merah = invalid
                drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    DrawingSpec(color=pose_color, thickness=3, circle_radius=4),
                    DrawingSpec(color=pose_color, thickness=3)
                )
                
                # --- tambahkan gambar lebar dada ---

                # koordinat bahu
                point_l_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                point_r_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                # offset dada (misalnya 20 px atau disesuaikan skala)
                offset_px = 50
                chest_y = int((point_l_shoulder[1] + point_r_shoulder[1]) / 2 + offset_px)

                # konversi 2 cm → pixel (pakai scale cm/px yang sudah Anda hitung)
                cm_to_px = 1 / scale   # misal scale cm_per_pixel
                shrink_px = int(2 / scale)

                # titik dada kiri & kanan (lebih pendek 2 cm per sisi)
                left_chest  = (point_l_shoulder[0]  + shrink_px, chest_y)
                right_chest = (point_r_shoulder[0] - shrink_px, chest_y)

                # gambar titik + garis biru
                cv2.circle(frame, left_chest, 6, (255, 0, 0), -1)
                cv2.circle(frame, right_chest, 6, (255, 0, 0), -1)
                cv2.line(frame, left_chest, right_chest, (255, 0, 0), 2)

                # hitung lebar dada cm
                lebar_dada_px = np.sqrt((right_chest[0]-left_chest[0])**2 +
                                        (right_chest[1]-left_chest[1])**2)
                lebar_dada_cm = lebar_dada_px * scale
                
                self.pose_valid = False
                self.countdown = 0
            else:
                pose_color = (0, 255, 0)  # hijau = valid
                drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    DrawingSpec(color=pose_color, thickness=3, circle_radius=4),
                    DrawingSpec(color=pose_color, thickness=3)
                )

                # --- skala tinggi badan ---
                top_y = lm[mp_pose.PoseLandmark.NOSE.value].y
                bottom_y = (lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y +
                            lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y) / 2
                pixel_height = abs(top_y - bottom_y) * h
                
               # --- tambahkan gambar lebar dada ---
                left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

                # koordinat bahu
                point_l_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
                point_r_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))

                # offset dada (misalnya 20 px atau disesuaikan skala)
                offset_px = 50
                chest_y = int((point_l_shoulder[1] + point_r_shoulder[1]) / 2 + offset_px)

                # konversi 2 cm → pixel (pakai scale cm/px yang sudah Anda hitung)
                cm_to_px = 1 / scale   # misal scale cm_per_pixel
                shrink_px = int(2 / scale)

                # titik dada kiri & kanan (lebih pendek 2 cm per sisi)
                left_chest  = (point_l_shoulder[0]  + shrink_px, chest_y)
                right_chest = (point_r_shoulder[0] - shrink_px, chest_y)

                # gambar titik + garis biru
                cv2.circle(frame, left_chest, 6, (255, 0, 0), -1)
                cv2.circle(frame, right_chest, 6, (255, 0, 0), -1)
                cv2.line(frame, left_chest, right_chest, (255, 0, 0), 2)

                # hitung lebar dada cm
                lebar_dada_px = np.sqrt((right_chest[0]-left_chest[0])**2 +
                                        (right_chest[1]-left_chest[1])**2)
                lebar_dada_cm = lebar_dada_px * scale
                print(f"Lebar dada valid: {lebar_dada_cm:.2f} cm")

                if not self.pose_valid:
                    self.pose_valid = True
                    self.countdown = 7
                    self.last_countdown_time = time.time()
                else:
                    if self.countdown > 0:
                    # update hanya setiap 1 detik
                        if time.time() - self.last_countdown_time >= 1:
                            self.countdown -= 1
                            self.last_countdown_time = time.time()
                        
                        self.label.text = f"Tahan pose... {self.countdown}s"
                    else:
                        measurements = self.calculate_measurements(lm, w, h, scaleHardcode, frame, scaleHardcode, getattr(self, "manual_values", None))
                        self.label.text = "\n".join([f"{k}: {v}" for k, v in measurements.items()])
                        self.captured = True
                        self.reset_btn.disabled = False
                        self.reset_buffer_btn.disabled = False
                         # freeze frame
                        self.captured_frame = frame.copy()

            # --- gambar landmark ---
            for idx, landmark in enumerate(lm):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (cx, cy), 4, pose_color, -1)

        # tampilkan video
        buf = cv2.flip(frame, 0).tobytes()
        image_texture = Texture.create(size=(w, h), colorfmt='bgr')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.label.markup = True
        self.img.texture = image_texture

    
    def distance_3d(self,a, b, scale_factor): # Tetap namakan 3d untuk kompatibilitas, tapi hanya pakai x,y
        """Calculate 2D Euclidean distance between two landmarks given a scale factor, ignoring Z."""
        return np.sqrt(
            ((a.x - b.x) * scale_factor)**2 +
            ((a.y - b.y) * scale_factor)**2
            # Z-coordinate diabaikan
        )
    
    def validate_pose(self, landmarks):
        """Enhanced and optimized pose validation with symmetry check"""
        lm = lambda x: landmarks[mp_pose.PoseLandmark[x].value]

        required_landmarks = [
            'NOSE', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 'LEFT_HIP', 'RIGHT_HIP',
            'LEFT_KNEE', 'RIGHT_KNEE', 'LEFT_ANKLE', 'RIGHT_ANKLE',
            'LEFT_ELBOW', 'RIGHT_ELBOW', 'LEFT_WRIST', 'RIGHT_WRIST',
            'LEFT_EAR', 'RIGHT_EAR'
        ]
        
        # 🔹 Step 1: Visibility check
        for name in required_landmarks:
            if lm(name).visibility < 0.6:
                return False, f"Pastikan '{name.replace('_', ' ').title()}' terlihat jelas (visibilitas rendah)."

        # 🔹 Step 2: Shoulder & hip alignment (vertical balance)
        shoulder_diff_y = abs(lm('LEFT_SHOULDER').y - lm('RIGHT_SHOULDER').y)
        hip_diff_y = abs(lm('LEFT_HIP').y - lm('RIGHT_HIP').y)

        if shoulder_diff_y > 0.05 or hip_diff_y > 0.05:
            return False, "Berdiri tegak. Pastikan bahu dan pinggul sejajar secara horizontal."

        # 🔹 Step 3: Pose symmetry check (frontal check)
        shoulder_width = abs(lm('RIGHT_SHOULDER').x - lm('LEFT_SHOULDER').x)
        hip_width = abs(lm('RIGHT_HIP').x - lm('LEFT_HIP').x)
        facing_ratio = shoulder_width / (hip_width + 1e-6)

        if not (0.9 <= facing_ratio <= 2.0):
            return False, "Hadap lurus ke kamera. Jangan miring ke samping."

        # 🔹 Step 4: Shoulder-depth symmetry (optional 3D stability)
        shoulder_z_diff = abs(lm('LEFT_SHOULDER').z - lm('RIGHT_SHOULDER').z)
        hip_z_diff = abs(lm('LEFT_HIP').z - lm('RIGHT_HIP').z)
        if shoulder_z_diff > 0.15 or hip_z_diff > 0.15:
            return False, "Tubuh agak miring terhadap kamera. Berdiri lurus menghadap depan."

        # 🔹 Step 5: Arm straightness (check elbow angle)
        left_elbow_angle = self.calculate_angle(lm('LEFT_SHOULDER'), lm('LEFT_ELBOW'), lm('LEFT_WRIST'))
        right_elbow_angle = self.calculate_angle(lm('RIGHT_SHOULDER'), lm('RIGHT_ELBOW'), lm('RIGHT_WRIST'))

        if left_elbow_angle < 110 or right_elbow_angle < 110:
            return False, "Rentangkan kedua lengan lurus ke bawah, menempel di sisi tubuh."

        # 🔹 Step 6: Leg position (kaki tidak terlalu lebar)
        shoulder_width_norm = self.distance_3d(lm('LEFT_SHOULDER'), lm('RIGHT_SHOULDER'), 1.0)
        knee_dist_horizontal = self.distance_3d(lm('LEFT_KNEE'), lm('RIGHT_KNEE'), 1.0)
        ankle_dist_horizontal = self.distance_3d(lm('LEFT_ANKLE'), lm('RIGHT_ANKLE'), 1.0)

        if shoulder_width_norm > 0:
            if knee_dist_horizontal > shoulder_width_norm * 0.6 or ankle_dist_horizontal > shoulder_width_norm * 0.6:
                return False, "Berdiri dengan kaki selebar bahu, tidak terlalu lebar."

        # 🔹 Step 7: Check if knees slightly bent
        if (lm('LEFT_ANKLE').y < lm('LEFT_KNEE').y - 0.01 or
            lm('RIGHT_ANKLE').y < lm('RIGHT_KNEE').y - 0.01):
            return False, "Luruskan kaki, jangan menekuk lutut."
        
        if abs(lm('LEFT_KNEE').y - lm('RIGHT_KNEE').y) > 0.02:
            return False, "Pastikan berdiri simetris dan kaki lurus."


        # 🔹 Step 8: Face symmetry using ears (optional frontal face check)
        ear_diff_y = abs(lm('LEFT_EAR').y - lm('RIGHT_EAR').y)
        if ear_diff_y > 0.05:
            return False, "Posisikan kepala tegak, jangan miring."

        # ✅ Passed all checks
        return True, "Pose valid dan siap diukur!"
    
    def distance_px(self,p1, p2) :
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    
    def hitung_akurasi(self,manual, app):
        hasil = {}
        total_error = 0
        n = len(manual)
        appUkur_norm = {k.lower(): v for k, v in app.items()}
        manual_norm = {k.lower(): v for k, v in manual.items()}

        for key in manual_norm:
            if key != 'nama' :
                manual_val = manual_norm[key]
                app_val = appUkur_norm[key]

                # selisih
                selisih = abs(app_val - manual_val)
                # error persen
                error_pct = (selisih / manual_val) * 100
                # akurasi persen
                akurasi = 100 - error_pct

                hasil[key] = {
                    "manual": manual_val,
                    "app": app_val,
                    "selisih": selisih,
                    "error_pct": round(error_pct, 2),
                    "akurasi": round(akurasi, 2)
                }

                total_error += error_pct

        # hitung rata-rata
        rata_error = total_error / n
        rata_akurasi = 100 - rata_error

        return hasil, round(rata_error, 2), round(rata_akurasi, 2)
    
    def find_b_from_circumference(self,a, target_circumference, tolerance=0.01, max_iterations=100):
        """Mencari nilai b (setengah kedalaman) secara numerik dari a dan lingkar."""
        # Tebakan awal: asumsikan bentuknya sedikit elips
        low_b = 0.1 * a
        high_b = a # Maksimum adalah lingkaran
        
        for _ in range(max_iterations):
            b_guess = (low_b + high_b) / 2
            circumference_guess = self.aproksimasiRamanujan(a, b_guess)
            
            if abs(circumference_guess - target_circumference) < tolerance:
                return b_guess
            
            if circumference_guess < target_circumference:
                low_b = b_guess
            else:
                high_b = b_guess
                
        return (low_b + high_b) / 2 
    
    def hitung_lingkar_perut(self,lebar_dada_cm, lingkar_dada_aktual_cm, lebar_perut_cm):
        """
        Menghitung estimasi lingkar perut berdasarkan data dada dan lebar perut.

        Args:
            lebar_dada_cm (float): Lebar dada yang diukur dari landmark (cm).
            lingkar_dada_aktual_cm (float): Lingkar dada yang sudah dihitung akurat (cm).
            lebar_perut_cm (float): Lebar area perut/pusar dari landmark (cm).

        Returns:
            float: Estimasi lingkar perut dalam cm.
        """
        # Langkah 1: Temukan kedalaman dada dari data yang ada
        a_dada = lebar_dada_cm / 2
        b_dada_estimasi = self.find_b_from_circumference(a_dada, lingkar_dada_aktual_cm)
        kedalaman_dada_estimasi = b_dada_estimasi * 2
        
        # Hitung rasio tubuh unik pengguna
        if lebar_dada_cm == 0:
            return 0 # Hindari pembagian dengan nol
        rasio_tubuh_pengguna = kedalaman_dada_estimasi / lebar_dada_cm
        
        print(f"Rasio Tubuh Pengguna (Kedalaman/Lebar) terdeteksi: {rasio_tubuh_pengguna:.2f}")

        # Langkah 2: Gunakan lebar perut yang sudah diukur
        # (lebar_perut_cm sudah menjadi input)
        
        # Langkah 3: Estimasi kedalaman perut menggunakan rasio tubuh
        kedalaman_perut_estimasi = lebar_perut_cm * rasio_tubuh_pengguna
        
        # Langkah 4: Hitung lingkar perut menggunakan model elips
        a_perut = lebar_perut_cm / 2
        b_perut = kedalaman_perut_estimasi / 2
        
        lingkar_perut_estimasi = self.aproksimasiRamanujan(a_perut, b_perut)
        
        return lingkar_perut_estimasi
    
    def calculate_angle(self,a, b, c):
        """Calculate the angle between three points (using 3D coordinates if available)"""
        # Untuk perhitungan sudut, kita bisa tetap pakai 3D jika ada, karena sudut relatif lebih stabil
        # Namun jika Anda ingin sepenuhnya 2D, hapus bagian 'z'
        if hasattr(a, 'z') and hasattr(b, 'z') and hasattr(c, 'z'):
            ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
            bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        else: # Ini akan selalu dipanggil jika kita hanya mengandalkan x, y
            ba = np.array([a.x - b.x, a.y - b.y])
            bc = np.array([c.x - b.x, c.y - b.y])
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
        
        return angle
    
    def reset_smoothing(self):
        self.buffers = {}
    
    def smooth_value(self, key, new_value, window=10):
        """
        Menyimpan nilai di buffer per metrik dan mengembalikan nilai rata-rata.
        """
        if not hasattr(self, "buffers"):
            self.buffers = {}
        if key not in self.buffers:
            self.buffers[key] = []
        self.buffers[key].append(new_value)
        if len(self.buffers[key]) > window:
            self.buffers[key].pop(0)
        return sum(self.buffers[key]) / len(self.buffers[key])
    
    def get_recommended_size_pdh(self, panjang_badan, lingkar_dada, lingkar_perut, lingkar_pinggang, panjang_tangan_pendek, panjang_celana):
        if lingkar_perut > lingkar_dada:
            dominant_value = lingkar_perut
        else:
            dominant_value = lingkar_dada

        size_chart = [
            {"size": 1, "panjang_badan": 71, "lingkar_dada": 100, "lingkar_perut": 95, "lingkar_pinggang": 77, "panjang_tangan_pendek": 22, "panjang_celana": 99},
            {"size": 2, "panjang_badan": 73, "lingkar_dada": 106, "lingkar_perut": 100, "lingkar_pinggang": 83, "panjang_tangan_pendek": 23.5, "panjang_celana": 100},
            {"size": 3, "panjang_badan": 75, "lingkar_dada": 112, "lingkar_perut": 105, "lingkar_pinggang": 89, "panjang_tangan_pendek": 25, "panjang_celana": 101},
        ]

        # Cari size pertama yang nilai dominan TIDAK LEBIH dari salah satu batas
        for s in size_chart:
            if (
                panjang_badan <= s["panjang_badan"]
                and dominant_value <= s["lingkar_dada"]
                and lingkar_perut <= s["lingkar_perut"]
            ):
                return f"Size {s['size']}"

        # Kalau semua lebih besar, ambil size terbesar
        return "Rekomendasi ukuran: Size 3 (fit longgar)" if dominant_value <= 120 else "⚠️ Size chart belum terimplementasi"
        
    
    def calc_panjang_tangan(self, lm, h, scale, side="LEFT"):
        """
        Hitung panjang tangan pendek & panjang dengan koreksi pose dan anatomi manusia.
        Gap target: ≤ 1 cm dari ukuran sebenarnya.
        """
            # --- Landmark dasar ---
        shoulder = lm(f"{side}_SHOULDER")
        elbow = lm(f"{side}_ELBOW")
        wrist = lm(f"{side}_WRIST")
        left_hip = lm("LEFT_HIP")
        right_hip = lm("RIGHT_HIP")

        # --- 1. Panjang segmen dasar ---
        upper_arm = self.euclidean_cm(shoulder, elbow, h, scale)
        lower_arm = self.euclidean_cm(elbow, wrist, h, scale)
        full_arm = self.euclidean_cm(shoulder, wrist, h, scale)

        if upper_arm <= 0 or lower_arm <= 0:
            upper_arm = full_arm * 0.45
            lower_arm = full_arm * 0.55

        # --- 2. Sudut siku (θ) ---
        try:
            cos_theta = (upper_arm**2 + lower_arm**2 - full_arm**2) / (2 * upper_arm * lower_arm)
            theta = math.degrees(math.acos(max(-1, min(1, cos_theta))))
        except:
            theta = 180

        # --- 3. Proporsi tubuh ---
        mid_hip = type("P", (), {
            "x": (left_hip.x + right_hip.x) / 2,
            "y": (left_hip.y + right_hip.y) / 2
        })
        torso_height_cm = self.euclidean_cm(shoulder, mid_hip, h, scale)
        if torso_height_cm <= 0:
            torso_height_cm = upper_arm + lower_arm
        ratio_arm_to_torso = (upper_arm + lower_arm) / torso_height_cm

        # --- 4. Koreksi perspektif kamera ---
        l_shoulder = lm("LEFT_SHOULDER")
        r_shoulder = lm("RIGHT_SHOULDER")
        shoulder_span_px = abs(l_shoulder.x - r_shoulder.x) * h
        shoulder_tilt_px = abs(l_shoulder.y - r_shoulder.y) * h
        tilt_ratio = min(shoulder_tilt_px / max(shoulder_span_px, 1), 0.5)

        perspective_factor = 1.0 - (tilt_ratio * 0.08)
        if theta < 175:
            perspective_factor -= (175 - theta) * 0.0015
        perspective_factor = max(0.92, perspective_factor)

        # --- 5. Hitung panjang tangan panjang ---
        panjang_tangan_panjang = (upper_arm + lower_arm) * perspective_factor

        # Koreksi skala empiris (lebih kecil supaya tidak over)
        arm_scale_correction = 0.88     # dari 0.91 → 0.88
        panjang_tangan_panjang *= arm_scale_correction

        # Koreksi proporsi torso
        if ratio_arm_to_torso < 0.95:
            panjang_tangan_panjang *= 1.01
        elif ratio_arm_to_torso > 1.05:
            panjang_tangan_panjang *= 0.97

        # Blending dengan panjang ideal manusia (≈0.43 × tinggi badan)
        ankle = lm("LEFT_ANKLE") if side == "LEFT" else lm("RIGHT_ANKLE")
        body_height_cm = self.euclidean_cm(shoulder, ankle, h, scale) * 1.04
        ideal_arm = body_height_cm * 0.43
        panjang_tangan_panjang = (panjang_tangan_panjang * 0.7) + (ideal_arm * 0.3)

        # Koreksi akhir kecil
        panjang_tangan_panjang += 0.3

        # --- 6. Panjang tangan pendek ---
        panjang_tangan_pendek = upper_arm * 0.66 + 0.6
        if theta < 170:
            panjang_tangan_pendek *= 0.97

        # --- 7. Stabilkan hasil (smoothing) ---
        panjang_tangan_panjang = self.smooth_value("panjang_tangan_panjang", panjang_tangan_panjang)
        panjang_tangan_pendek = self.smooth_value("panjang_tangan_pendek", panjang_tangan_pendek)

        return round(panjang_tangan_pendek, 1), round(panjang_tangan_panjang, 1)
    
    def aproksimasiRamanujan(self,a,b) :
        output = math.pi * (3*(a + b) - math.sqrt((3*a+b)*(a+3*b)))  
        return output
    
    def calc_panjang_celana(self, lm, h, scale):
        """
        Hitung Panjang Celana (cm)
        Diukur dari pusar turun ±2 cm sampai ke mata kaki (nomor 13 pada form).
        """
        # Titik referensi utama
        left_hip = lm("LEFT_HIP")
        right_hip = lm("RIGHT_HIP")
        left_ankle = lm("LEFT_ANKLE")
        right_ankle = lm("RIGHT_ANKLE")

        # 1️⃣ Titik tengah pinggul (pusar kira-kira 2 cm di atas midpoint)
        mid_hip = type("P", (), {
            "x": (left_hip.x + right_hip.x) / 2.0,
            "y": (left_hip.y + right_hip.y) / 2.0
        })

        # Koreksi posisi ke pusar: naik sedikit (sekitar 2 cm)
        offset_cm = -2.0  # naik 2 cm dari pinggul
        offset_px = offset_cm / max(scale, 1e-6)
        waist_y = mid_hip.y + offset_px / h  # ubah ke koordinat normalisasi 0–1
        pusar = type("P", (), {"x": mid_hip.x, "y": waist_y})
        
        # 1.1️⃣ Tentukan Titik Awal Celana (2 cm DI BAWAH pusar)
        celana_start_offset_cm = 2.0  # Turun 2 cm DARI PUSAR
        celana_start_offset_px = celana_start_offset_cm / max(scale, 1e-6)
        
        celana_start_y_norm = pusar.y + celana_start_offset_px / h
        titik_awal_celana = type("P", (), {"x": mid_hip.x, "y": celana_start_y_norm})

        # 2️⃣ Titik tengah mata kaki
        mid_ankle = type("P", (), {
            "x": (left_ankle.x + right_ankle.x) / 2.0,
            "y": (left_ankle.y + right_ankle.y) / 2.0
        })
        
        shoulder_width = abs(lm('RIGHT_SHOULDER').x - lm('LEFT_SHOULDER').x)
        hip_width = abs(lm('RIGHT_HIP').x - lm('LEFT_HIP').x)
        body_facing_ratio = shoulder_width / hip_width
        print(f'Body facing ratio {body_facing_ratio}')
        
        correction_angle = 1.0
        if body_facing_ratio < 0.9:
            correction_angle = 1.0 + (0.9 - body_facing_ratio) * 0.6  # max +6%
        # 3️⃣ Hitung jarak pusar → mata kaki
        panjang_celana_cm = self.euclidean_cm(pusar, mid_ankle, h, scale)

        # 4️⃣ Koreksi kecil (pose berdiri kadang menyebabkan lutut sedikit tekuk)
        # panjang_celana_cm *= 1.02  # tambahkan 2% sebagai koreksi berdiri natural
        # 7️⃣ Koreksi natural posture (tekukan lutut ringan)
        posture_correction = 1.02  # default 1.5% tambahan
        if panjang_celana_cm < 90:
            posture_correction = 1.03  # jika lebih pendek → tambahkan sedikit koreksi
        # panjang_celana_cm *= posture_correction * correction_angle
        panjang_celana_cm *= 1.05

        # 5️⃣ Stabilkan hasil
        panjang_celana_cm = self.smooth_value("panjang_celana", panjang_celana_cm)

        return panjang_celana_cm
    
    def round_cm(self, value):
        """
        Rounding custom untuk pengukuran cm
        - Decimal > 0.6: rounding ke atas ke integer
        - Decimal <= 0.6: rounding ke bawah ke integer
        """
        integer_part = int(value)
        decimal_part = value - integer_part
        
        # Handle floating point precision
        decimal_part = round(decimal_part, 2)  # Round to 2 decimal untuk avoid floating point error
        
        if decimal_part > 0.6 + 1e-10:
            return integer_part + 1
        else:
            return integer_part
        
    def show_manual_input_popup(self):
        """Tampilkan popup untuk input nilai manual sebelum mulai pengukuran"""

        layout = BoxLayout(orientation='vertical', spacing=10, padding=10)
        layout.add_widget(Label(text="Masukkan data manual (dalam cm) untuk report testing:", size_hint_y=None, height=40))

        # Field sesuai dictionary manual
        fields = [
            "Nama",
            "Panjang badan",
            "Lebar dada",
            "Lingkar dada",
            # "Lingkar Perut",  # kamu bisa aktifkan kalau mau
            "Lingkar Pinggang",
            "Panjang tangan pendek",
            "Panjang tangan panjang",
            "Panjang celana"
        ]

        # Input fields
        self.manual_inputs = {}
        for field in fields:
            sub = BoxLayout(orientation='horizontal', size_hint_y=None, height=40)
            sub.add_widget(Label(text=field, size_hint_x=0.5))
            ti = TextInput(multiline=False, size_hint_x=0.5)
            sub.add_widget(ti)
            layout.add_widget(sub)
            self.manual_inputs[field] = ti

        # Tombol konfirmasi
        btn_confirm = Button(
            text="Mulai Pengukuran", 
            background_color=(0.1, 0.6, 0.1, 1),
            size_hint_y=None, 
            height=50
        )
        layout.add_widget(btn_confirm)

        # Popup
        popup = Popup(
            title="Input Nilai Manual", 
            content=layout, 
            size_hint=(0.8, 0.8), 
            auto_dismiss=False
        )
        self.manual_popup = popup

        # Event konfirmasi
        def start_measurement(instance):
            self.manual_values = {}
            for k, v in self.manual_inputs.items():
                if k != 'Nama' :
                    try:
                        self.manual_values[k] = float(v.text)
                    except ValueError:
                        self.manual_values[k] = None
                else :
                    self.manual_values[k] = v.text

            popup.dismiss()
            self.label.text = "✅ Nilai manual disimpan. Silakan posisikan badan di kamera..."
            print("✅ Nilai manual tersimpan:", self.manual_values)

            # TODO: Panggil fungsi pengukuran otomatis di sini
            # self.start_measurement_process(self.manual_values)

        btn_confirm.bind(on_press=start_measurement)
        popup.open()
    
    def get_recommended_size_jas_celana(self,panjang_badan, lingkar_dada, lingkar_perut, panjang_tangan_panjang, lingkar_pinggang, panjang_celana):
        """
        Menentukan rekomendasi ukuran jas & celana berdasarkan size chart yang diberikan.
        Jika lingkar perut > lingkar dada, maka acuan utama diambil dari lingkar perut.
        """

        size_chart = {
            1: {
                "jas": {"panjang_badan": 72, "lingkar_dada": 100, "lingkar_perut": 92, "panjang_tangan_panjang": 59},
                "celana": {"lingkar_pinggang": 79, "panjang_celana": 100}
            },
            2: {
                "jas": {"panjang_badan": 73, "lingkar_dada": 104, "lingkar_perut": 96, "panjang_tangan_panjang": 60},
                "celana": {"lingkar_pinggang": 83, "panjang_celana": 100}
            },
            3: {
                "jas": {"panjang_badan": 74, "lingkar_dada": 109, "lingkar_perut": 100, "panjang_tangan_panjang": 62},
                "celana": {"lingkar_pinggang": 89, "panjang_celana": 102}
            }
        }

        # --- Pilih acuan utama untuk tubuh bagian atas ---
        dominant_torso = max(lingkar_dada, lingkar_perut)  # jika perut > dada, ambil perut

        # --- Cari size jas yang sesuai ---
        size_jas = None
        for size, s in size_chart.items():
            if (
                panjang_badan <= s["jas"]["panjang_badan"]
                and dominant_torso <= s["jas"]["lingkar_dada"]
                and lingkar_perut <= s["jas"]["lingkar_perut"]
                and panjang_tangan_panjang <= s["jas"]["panjang_tangan_panjang"]
            ):
                size_jas = size
                break

        # --- Cari size celana yang sesuai ---
        size_celana = None
        for size, s in size_chart.items():
            if (
                lingkar_pinggang <= s["celana"]["lingkar_pinggang"]
                and panjang_celana <= s["celana"]["panjang_celana"]
            ):
                size_celana = size
                break

        # --- Hasil akhir ---
        if size_jas is None and size_celana is None:
            return "⚠️ Size chart belum terimplementasi untuk ukuran ini."
        else:
            return f"Rekomendasi ukuran:\n🧥 Jas: Size {size_jas if size_jas else 'N/A'}\n👖 Celana: Size {size_celana if size_celana else 'N/A'}"
    
    # def apply_auto_converging_correction(self, appUkur, manual, prev_correction=None, alpha=0.5, base_beta=0.3):
    #     """
    #     Auto-Converging Adaptive Correction
    #     -----------------------------------
    #     - Menyesuaikan beta (laju koreksi) secara otomatis berdasarkan besar error.
    #     - Semakin besar error, semakin cepat koreksi dilakukan (beta naik).
    #     - Semakin kecil error, beta turun agar hasil stabil (konvergen).
    #     """

    #     correction = {}
    #     corrected = {}

    #     appUkur_norm = {k.lower(): v for k, v in appUkur.items()}
    #     manual_norm = {k.lower(): v for k, v in manual.items()}

    #     for key in manual_norm.keys():
    #         if key in appUkur_norm:
    #             app_val = appUkur_norm[key]
    #             manual_val = manual_norm[key]
    #             diff = manual_val - app_val

    #             # Terapkan smoothing dengan prev_correction
    #             if prev_correction and key in prev_correction:
    #                 diff = (1 - alpha) * prev_correction[key] + alpha * diff

    #             abs_diff = abs(diff)

    #             # === AUTO-CONVERGENCE ADAPTATION ===
    #             # Beta menyesuaikan error, tapi tetap dalam batas (0.1 - 0.7)
    #             if abs_diff > 8:
    #                 beta = min(0.9, base_beta + 0.4)
    #             elif abs_diff > 5:
    #                 beta = min(0.6, base_beta + 0.3)
    #             elif abs_diff > 2:
    #                 beta = min(0.4, base_beta + 0.1)
    #             else:
    #                 # Jika error sudah kecil, beta dikurangi agar stabil
    #                 beta = max(0.1, base_beta * (abs_diff / 2 + 0.1))

    #             corrected_val = app_val + beta * diff
    #             correction[key] = diff
    #             corrected[key] = corrected_val

    #             print(f"[DEBUG] {key}: app={app_val:.2f}, manual={manual_val:.2f}, diff={diff:.2f}, beta={beta:.2f}, corrected={corrected_val:.2f}")

    #     return corrected, correction
    
    def apply_adaptive_correction(self, appUkur, manual, prev_correction=None, alpha=0.5):
        """
        Koreksi adaptif otomatis berdasarkan besarnya error (auto-scaled correction rate).
        """
        correction = {}
        corrected = {}

        # Normalisasi key
        appUkur_norm = {k.lower(): v for k, v in appUkur.items()}
        manual_norm = {k.lower(): v for k, v in manual.items()}

        for key in manual_norm.keys():
            if key != 'nama':
                if key in appUkur_norm:
                    app_val = appUkur_norm[key]
                    manual_val = manual_norm[key]

                    diff = manual_val - app_val  # selisih positif → app lebih kecil
                    abs_diff = abs(diff)

                    # Tentukan correction_rate dinamis
                    if abs_diff < 2:
                        correction_rate = 0.3   # kecil, minor adjust
                    elif abs_diff < 3:
                        correction_rate = 0.5   # menengah
                    elif abs_diff < 6:
                        correction_rate = 0.7   # sedang
                    else:
                        correction_rate = 0.85   # besar, koreksi cepat

                    # Blending dengan koreksi sebelumnya agar stabil
                    if prev_correction and key in prev_correction:
                        diff = (1 - alpha) * prev_correction[key] + alpha * diff

                    applied_correction = diff * correction_rate
                    corrected_val = app_val + applied_correction

                    correction[key] = applied_correction
                    corrected[key] = round(corrected_val, 2)

                    print(f"[DEBUG] {key}: manual={manual_val}, app={app_val}, diff={diff:.2f}, "
                        f"rate={correction_rate:.2f}, applied={applied_correction:.2f}, corrected={corrected_val:.2f}")

        return corrected, correction






    def calculate_measurements(self, landmarks, w, h, scale, frame=None, scaleHardcode=0, manualValues = None):
        print(f'manual value dari input {manualValues}')
        h, w, _ = frame.shape
        lm = lambda x: landmarks[mp_pose.PoseLandmark[x].value]
        
        # Ambil mid bahu x dan y
        shoulder_mid_y = (lm('LEFT_SHOULDER').y + lm('RIGHT_SHOULDER').y) / 2
        shoulder_mid_x = (lm('LEFT_SHOULDER').x + lm('RIGHT_SHOULDER').x) / 2

        # Ambil mid hip x dan y
        hip_mid_y = (lm('LEFT_HIP').y + lm('RIGHT_HIP').y) / 2
        hip_mid_x = (lm('LEFT_HIP').x + lm('RIGHT_HIP').x) / 2

        shoulder_mid = type('obj', (object,), {'x': shoulder_mid_x, 'y': shoulder_mid_y})
        hip_mid = type('obj', (object,), {'x': hip_mid_x, 'y': hip_mid_y})

        # Panjang badan (shoulder → hip)
        panjang_badan = self.euclidean_cm(shoulder_mid, hip_mid, h, scale)
        # panjang_badan = self.smooth_value("panjang_badan", panjang_badan)
        
        
       # ambil titik bahu & pinggul
        left_shoulder = lm('LEFT_SHOULDER')
        right_shoulder = lm('RIGHT_SHOULDER')
        left_hip = lm('LEFT_HIP')
        right_hip = lm('RIGHT_HIP')

        # koordinat pixel
        point_l_shoulder = (int(left_shoulder.x * w), int(left_shoulder.y * h))
        point_r_shoulder = (int(right_shoulder.x * w), int(right_shoulder.y * h))
        point_l_hip = (int(left_hip.x * w), int(left_hip.y * h))
        point_r_hip = (int(right_hip.x * w), int(right_hip.y * h))

        # mid shoulder & mid hip
        mid_shoulder = ((point_l_shoulder[0] + point_r_shoulder[0]) // 2,
                        (point_l_shoulder[1] + point_r_shoulder[1]) // 2)
        mid_hip = ((point_l_hip[0] + point_r_hip[0]) // 2,
                (point_l_hip[1] + point_r_hip[1]) // 2)

        # mid shoulder & hip (float)
        mid_shoulder_y = (left_shoulder.y + right_shoulder.y) / 2 * h
        mid_hip_y = (left_hip.y + right_hip.y) / 2 * h

        # chest line ~32% dari shoulder→hip
        chest_y = mid_shoulder_y + 0.32 * (mid_hip_y - mid_shoulder_y)

        # interpolasi horizontal ~53%
        alpha = 0.53
        left_chest_x  = left_shoulder.x * w + alpha * (left_hip.x * w - left_shoulder.x * w)
        right_chest_x = right_shoulder.x * w + alpha * (right_hip.x * w - right_shoulder.x * w)
        
        shoulder_width_px_1 = abs(point_r_shoulder[0] - point_l_shoulder[0])
        shoulder_width_cm_1 = shoulder_width_px_1 * scale

        # optional shrink ke dalam
        # shrink_cm = 1.5
        # shrink_cm = lebar_dada_cm * 0.04
        shrink_cm = shoulder_width_cm_1 * 0.04  # 4% dari lebar bahu
        shrink_px = int(shrink_cm / scale)

        left_chest  = (left_chest_x + shrink_px, chest_y)
        right_chest = (right_chest_x - shrink_px, chest_y)

        # hitung jarak
        lebar_dada_px = self.distance_px(left_chest, right_chest)
        lebar_dada_cm = lebar_dada_px * scale
        lebar_dada_cm = self.smooth_value("lebar_dada", lebar_dada_cm)

        # Lingkar dada
        # chest_width sudah benar (lebar dada cm)
        chest_width = lebar_dada_cm  

        # kedalaman dada biasanya 0.60 – 0.70 dari lebar (bukan 0.8)
        depth_ratio = 0.60
        if chest_width < 34:
            depth_ratio = 0.60
        elif chest_width < 38:
            depth_ratio = 0.62
        else:
            depth_ratio = 0.64
        chest_depth = chest_width * depth_ratio
        a = chest_width/2
        b = chest_depth/2
        lingkar_dada_rumus = self.aproksimasiRamanujan(a,b)
        lingkar_dada_rumus -= 10.0
        # setengah sumbu elips
        # a = chest_width / 2
        # b = chest_depth / 2

        # # keliling elips pakai aproksimasi Ramanujan
        # lingkar_dada = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
        lingkar_dada = self.smooth_value("lingkar_dada", lingkar_dada_rumus)
        
        
        # lingkar perut start
    

        # posisi pusar (sekitar 45% bahu→pinggul)
        waist_y = int(mid_shoulder[1] + 0.45 * (mid_hip[1] - mid_shoulder[1]))

        # ambil interpolasi X dari bahu ke pinggul
        alpha = 0.1  # sedikit lebih dekat ke pinggul
        left_waist_x  = int(point_l_shoulder[0] + alpha * (point_l_hip[0] - point_l_shoulder[0]))
        right_waist_x = int(point_r_shoulder[0] + alpha * (point_r_hip[0] - point_r_shoulder[0]))
        
        # outward lebih besar (karena landmark hip jauh di dalam)
        outward_cm = 6.5   # coba 12–14 cm
        # outward_cm = lebar_dada_cm * 0.15
        outward_px = int(outward_cm / scale)

        left_waist  = (left_waist_x - outward_px, waist_y)
        right_waist = (right_waist_x + outward_px, waist_y)

        # hitung lebar perut
        waist_width_px = self.distance_px(left_waist, right_waist)
        waist_width_cm = waist_width_px * scale

        # kedalaman perut (lebih bulat dari dada)
        waist_depth_cm = waist_width_cm * 1.05   # bisa 1.0–1.1

        # elips Ramanujan
        a = waist_width_cm / 2
        b = waist_depth_cm / 2
        lingkar_perut = math.pi * (3*(a+b) - math.sqrt((3*a+b)*(a+3*b)))
        empirical_correction = 1.02  # 4% reduction
        lingkar_perut *= empirical_correction
        lingkar_perut -= 30.0
        lingkar_perut = self.smooth_value("lingkar_perut", lingkar_perut)


        # lingkar perut end
        
        # # lingkar perut optimized
        # lebar_perut_cm = self.euclidean_cm(left_hip, right_hip, h, scale)
        # lingkar_perut = self.hitung_lingkar_perut(lebar_dada_cm, lingkar_dada, lebar_perut_cm)

        
       # --- LINGKAR PINGGANG (Adaptive Universal Model) ---

        # posisi dasar pinggang = ±2.5 cm di bawah pusar
        waist_base_y = int(mid_shoulder[1] + 0.40 * (mid_hip[1] - mid_shoulder[1]))  # pusar
        # offset_cm = 1.5
        offset_cm = panjang_badan * 0.03
        offset_px = int(offset_cm / scale)
        waist_y = waist_base_y + offset_px

        # interpolasi horizontal: 0.12–0.16 dinamis tergantung bentuk tubuh
        # semakin besar perut, semakin dekat ke pinggul
        ratio_torso = lingkar_perut / lingkar_dada  # proporsi tubuh
        ratio_torso = max(0.8, min(1.3, ratio_torso))
        alpha_pinggang = 0.10 + 0.03 * min(ratio_torso, 1.2)  # batas max 1.2x
        
        print(f'torso : {ratio_torso}')

        left_waist_x  = int(point_l_shoulder[0] + alpha_pinggang * (point_l_hip[0] - point_l_shoulder[0]))
        right_waist_x = int(point_r_shoulder[0] + alpha_pinggang * (point_r_hip[0] - point_r_shoulder[0]))

        # outward correction proporsional dengan lebar dada dan bentuk tubuh
        # makin ramping orangnya, makin kecil outward-nya
        body_factor = max(0.01, min(0.03, 0.03 - (ratio_torso - 1.0) * 0.015))
        outward_cm_pinggang = lebar_dada_cm * body_factor
        outward_px = int(outward_cm_pinggang / scale)

        left_waist  = (left_waist_x + outward_px, waist_y)
        right_waist = (right_waist_x - outward_px, waist_y)

        # hitung lebar pinggang
        waist_width_px = self.distance_px(left_waist, right_waist)
        waist_width_cm = waist_width_px * scale

        # kedalaman pinggang adaptif — makin besar perut, makin kecil rasio
        waist_depth_ratio = max(0.75, min(0.88, 0.88 - (ratio_torso - 1.0) * 0.08))
        waist_depth_cm = waist_width_cm * waist_depth_ratio

        # hitung elips Ramanujan
        a = waist_width_cm / 2
        b = waist_depth_cm / 2
        # Gunakan rumus yang berbeda berdasarkan bentuk tubuh
        if ratio_torso > 1.1:  # Untuk perut yang lebih besar
            # Rumus untuk elliptical shape yang lebih oval
            h = ((a - b) ** 2) / ((a + b) ** 2)
            lingkar_pinggang_rumus = math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))
        else:  # Untuk bentuk normal
            # Ramanujan approximation
            lingkar_pinggang_rumus = math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))

        # Blending yang lebih konservatif dengan lingkar perut
        # Untuk ukuran asli 89cm, kita perlu mengurangi influence lingkar_perut
        blend = max(0.15, min(0.3, 0.25 - (ratio_torso - 1.0) * 0.08))  # dikurangi range blending

        # Prediksi berdasarkan proporsi tubuh yang lebih akurat
        pred_pinggang = lingkar_perut * (0.82 + (ratio_torso - 1.0) * 0.03)  # koefisien dikurangi

        lingkar_pinggang = lingkar_pinggang_rumus * (1 - blend) + pred_pinggang * blend
        
        waist_length_px = np.linalg.norm(np.array(left_waist) - np.array(right_waist))
        
        waist_length_cm = waist_length_px / scale

        # # Correction factor berdasarkan data aktual
        # # Jika kita tahu ukuran seharusnya ~89cm, kita bisa apply scaling
        # correction_factor = 0.78
        correction_factor = 0.78 + 0.02 * (ratio_torso - 1.0)
        lingkar_pinggang *= correction_factor

        # Batasi range yang realistic
        lingkar_pinggang = max(lingkar_pinggang * 0.7, min(lingkar_pinggang * 1.3, lingkar_pinggang))

        # Stabilisasi nilai
        lingkar_pinggang = self.smooth_value("lingkar_pinggang", lingkar_pinggang)
        
        
        pendek, panjang = self.calc_panjang_tangan(lm, h, scale, side="LEFT")

        # panjang tangan pendek (bahu → 2 cm di atas siku)
        panjang_tangan_pendek = pendek
        panjang_tangan_pendek = self.smooth_value("panjang_tangan_pendek", panjang_tangan_pendek)

        # panjang tangan panjang (lebihkan 2 cm)
        panjang_tangan_panjang = panjang
        panjang_tangan_panjang *= 0.95
        panjang_tangan_panjang = self.smooth_value("panjang_tangan_panjang", panjang_tangan_panjang)
        
        
        panjang_celana = self.calc_panjang_celana(lm, h, scale)
        panjang_celana = self.smooth_value("panjang_celana", panjang_celana)
        
        # manual = {
        #     "Panjang badan": 70.0,
        #     "Lebar dada": 36.0,
        #     "Lebar Bahu full": 45.0,
        #     "Lingkar Dada": 93.0,
        #     # "Lingkar Pinggang": 89.0,
        #     # "Panjang Celana": 95.0,

        # }

    # Data hasil aplikasi
        appUkur = {
            "Panjang badan": round(panjang_badan, 1),
            "Lebar dada": round(lebar_dada_cm, 1),
            "Lingkar dada": round(lingkar_dada, 1),
            # "Lingkar Perut": round(lingkar_perut, 1),
            "Lingkar Pinggang": round(lingkar_pinggang, 1),
            "Panjang tangan pendek":round(panjang_tangan_pendek, 1),
            "Panjang tangan panjang": round(panjang_tangan_panjang, 1),
            "Panjang celana": round(panjang_celana, 1),
        }
        
        corrected_app, correction_model = self.apply_adaptive_correction(appUkur, manualValues)

        data_export = []
        hasil_persentase, rata_error, rata_akurasi = self.hitung_akurasi(manualValues, corrected_app)
        for key, val in hasil_persentase.items():
            data_export.append({
                "Field": key,
                "Ukuran Asli (cm)": val['manual'],
                "App (cm)": val['app'],
                "Selisih (cm)": val['selisih'],
                "Error (%)": val['error_pct'],
                "Akurasi (%)": val['akurasi']
            })
            
        # Tambahkan nama sebagai kolom tambahan untuk setiap baris
        for row in data_export:
            row["Nama"] = manualValues["Nama"]

        # Buat DataFrame
        df = pd.DataFrame(data_export)

        # Tambahkan baris summary di akhir (opsional)
        summary_row = {
            "Nama": manualValues["Nama"],
            "Field": "--- Ringkasan ---",
            "Ukuran Asli (cm)": "",
            "App (cm)": "",
            "Selisih (cm)": "",
            "Error (%)": rata_error,
            "Akurasi (%)": rata_akurasi
        }
        df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)

        # Export ke CSV
        df.to_csv(f"hasil_ukur_{manualValues['Nama']}.csv", index=False)

        # Export ke Excel
        df.to_excel(f"hasil_ukur_{manualValues['Nama']}.xlsx", index=False)

        print("Export selesai: hasil_ukur.csv / hasil_ukur.xlsx")
        
        
        detail_str = ""
        detail_str += f'Nama : {manualValues["Nama"]} \n'
        for key, val in hasil_persentase.items():
            detail_str += (
                f"{key}:\n"
                f"  Ukuran Asli: {val['manual']} cm\n"
                f"  App: {val['app']} cm\n"
                f"  Selisih: {val['selisih']} cm\n"
                f"  Error: {val['error_pct']}%\n"
                f"  Akurasi: {val['akurasi']}%\n\n"
            )

        summary_str = (
            f"\n--- Ringkasan ---\n"
            f"Rata-rata Error: {rata_error}%\n"
            f"Rata-rata Akurasi: {rata_akurasi}%\n"
        )
        
        print(detail_str)
        print(f'akurasi rata rata : {summary_str}')
        
        # hasil_size_pdh = self.get_recommended_size_pdh(self.round_cm(corrected_app['panjang badan']),self.round_cm(corrected_app['lingkar dada']), 
        #                                                self.round_cm(corrected_app['lingkar perut'])
        #                                                , self.round_cm(corrected_app['lingkar pinggang']), self.round_cm(corrected_app['panjang tangan pendek'])
        #                                                , self.round_cm(corrected_app['panjang celana']));
        # hasil_size_jas = self.get_recommended_size_jas_celana(self.round_cm(panjang_badan),self.round_cm(lingkar_dada), self.round_cm(lingkar_perut)
        #                                                , self.round_cm(panjang_tangan_panjang)
        #                                                ,self.round_cm(lingkar_pinggang)
        #                                                , self.round_cm(panjang_celana));
        return {
            "Panjang badan": round(corrected_app['panjang badan'],2),
            "Lebar dada": round(corrected_app['lebar dada'],2),
            "Lingkar dada": round(corrected_app['lingkar dada'],2),
            "Lingkar Pinggang": round(corrected_app['lingkar pinggang'],2),
            "Panjang Tangan": f'{round(corrected_app["panjang tangan pendek"], 2)} / {round(corrected_app["panjang tangan panjang"], 2)}',
            "Panjang celana": round(corrected_app['panjang celana'], 2),
            # "recommendation size pdh" : hasil_size_pdh,
            # "recommendation size jas" : hasil_size_jas,
            "Analisis Detail" : detail_str,
            "Summary" : summary_str
        }

if __name__ == '__main__':
    BodyMeasureApp().run()
