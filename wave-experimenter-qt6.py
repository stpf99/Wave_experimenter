import sys
import numpy as np
from scipy import signal
from scipy.io import wavfile
import pygame
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QSlider, QLabel, QPushButton, QRadioButton, QGroupBox, QComboBox, 
                             QGraphicsView, QGraphicsScene)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPen, QPainterPath, QColor

class InstrumentPresets:
    """Klasa zawierająca presety dla różnych typów instrumentów"""
    @staticmethod
    def get_preset_categories():
        return {
            'Percussion': ['Kick', 'Snare', 'HiHat', 'Maracas'],
            'Wind': ['Flute', 'Trumpet'],
            'Vocal': ['Choir', 'Vowel']
        }

    @staticmethod
    def get_percussion_preset(type_name):
        presets = {
            'Kick': {
                'wave_shape1': 'sine', 'wave_shape2': 'square', 'wave_mix': 0.3, 'frequency_multiplier': 1.0,
                'attack_time': 0.01, 'decay_time': 0.15, 'sustain_level': 0.0, 'release_time': 0.1,
                'harm1_weight': 1.0, 'harm2_weight': 0.4, 'harm3_weight': 0.2, 'harm4_weight': 0.1, 'harm5_weight': 0.05,
                'frequency': 55, 'fm_frequency': 40, 'fm_depth': 0.5, 'distortion': 0.2, 'noise_level': 0.1,
                'bit_crush': 0.0, 'filter_cutoff': 200, 'filter_resonance': 0.3, 'chorus_depth': 0.0, 'chorus_rate': 0.0
            },
            'Snare': {
                'wave_shape1': 'noise', 'wave_shape2': 'triangle', 'wave_mix': 0.7, 'frequency_multiplier': 1.0,
                'attack_time': 0.001, 'decay_time': 0.2, 'sustain_level': 0.1, 'release_time': 0.25,
                'harm1_weight': 0.8, 'harm2_weight': 0.6, 'harm3_weight': 0.4, 'harm4_weight': 0.2, 'harm5_weight': 0.1,
                'frequency': 180, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.3, 'noise_level': 0.7,
                'bit_crush': 0.2, 'filter_cutoff': 5000, 'filter_resonance': 0.5, 'chorus_depth': 0.0, 'chorus_rate': 0.0
            },
            'HiHat': {
                'wave_shape1': 'noise', 'wave_shape2': 'sine', 'wave_mix': 0.9, 'frequency_multiplier': 1.0,
                'attack_time': 0.001, 'decay_time': 0.05, 'sustain_level': 0.0, 'release_time': 0.1,
                'harm1_weight': 0.5, 'harm2_weight': 0.3, 'harm3_weight': 0.2, 'harm4_weight': 0.1, 'harm5_weight': 0.05,
                'frequency': 8000, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.1, 'noise_level': 0.8,
                'bit_crush': 0.4, 'filter_cutoff': 10000, 'filter_resonance': 0.2, 'chorus_depth': 0.0, 'chorus_rate': 0.0
            },
            'Maracas': {
                'wave_shape1': 'noise', 'wave_shape2': 'sine', 'wave_mix': 0.95, 'frequency_multiplier': 1.0,
                'attack_time': 0.001, 'decay_time': 0.03, 'sustain_level': 0.0, 'release_time': 0.05,
                'harm1_weight': 0.4, 'harm2_weight': 0.2, 'harm3_weight': 0.1, 'harm4_weight': 0.05, 'harm5_weight': 0.02,
                'frequency': 6000, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.0, 'noise_level': 0.9,
                'bit_crush': 0.5, 'filter_cutoff': 8000, 'filter_resonance': 0.1, 'chorus_depth': 0.0, 'chorus_rate': 0.0
            }
        }
        return presets.get(type_name, {})

    @staticmethod
    def get_wind_preset(type_name):
        presets = {
            'Flute': {
                'wave_shape1': 'sine', 'wave_shape2': 'triangle', 'wave_mix': 0.2, 'frequency_multiplier': 1.0,
                'attack_time': 0.1, 'decay_time': 0.1, 'sustain_level': 0.8, 'release_time': 0.2,
                'harm1_weight': 1.0, 'harm2_weight': 0.4, 'harm3_weight': 0.2, 'harm4_weight': 0.1, 'harm5_weight': 0.05,
                'frequency': 261.63, 'fm_frequency': 5.0, 'fm_depth': 0.1, 'distortion': 0.0, 'noise_level': 0.05,
                'bit_crush': 0.0, 'filter_cutoff': 2000, 'filter_resonance': 0.1, 'vibrato_rate': 5.0, 'vibrato_depth': 0.15,
                'chorus_depth': 0.1, 'chorus_rate': 0.5
            },
            'Trumpet': {
                'wave_shape1': 'square', 'wave_shape2': 'sawtooth', 'wave_mix': 0.4, 'frequency_multiplier': 1.0,
                'attack_time': 0.05, 'decay_time': 0.1, 'sustain_level': 0.7, 'release_time': 0.15,
                'harm1_weight': 1.0, 'harm2_weight': 0.8, 'harm3_weight': 0.6, 'harm4_weight': 0.4, 'harm5_weight': 0.2,
                'frequency': 349.23, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.3, 'noise_level': 0.1,
                'bit_crush': 0.0, 'filter_cutoff': 3000, 'filter_resonance': 0.4, 'vibrato_rate': 6.0, 'vibrato_depth': 0.2,
                'chorus_depth': 0.0, 'chorus_rate': 0.0
            }
        }
        return presets.get(type_name, {})

    @staticmethod
    def get_vocal_preset(type_name):
        presets = {
            'Choir': {
                'wave_shape1': 'sine', 'wave_shape2': 'triangle', 'wave_mix': 0.3, 'frequency_multiplier': 1.0,
                'attack_time': 0.2, 'decay_time': 0.1, 'sustain_level': 0.8, 'release_time': 0.3,
                'harm1_weight': 1.0, 'harm2_weight': 0.6, 'harm3_weight': 0.4, 'harm4_weight': 0.2, 'harm5_weight': 0.1,
                'frequency': 220.0, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.0, 'noise_level': 0.05,
                'bit_crush': 0.0, 'filter_cutoff': 2500, 'filter_resonance': 0.2, 'vibrato_rate': 5.0, 'vibrato_depth': 0.1,
                'chorus_depth': 0.2, 'chorus_rate': 0.3, 'formant1_freq': 500, 'formant1_amp': 1.0,
                'formant2_freq': 1500, 'formant2_amp': 0.5, 'formant3_freq': 2500, 'formant3_amp': 0.25
            },
            'Vowel': {
                'wave_shape1': 'sine', 'wave_shape2': 'triangle', 'wave_mix': 0.2, 'frequency_multiplier': 1.0,
                'attack_time': 0.15, 'decay_time': 0.1, 'sustain_level': 0.7, 'release_time': 0.25,
                'harm1_weight': 1.0, 'harm2_weight': 0.5, 'harm3_weight': 0.3, 'harm4_weight': 0.15, 'harm5_weight': 0.07,
                'frequency': 200.0, 'fm_frequency': 0.0, 'fm_depth': 0.0, 'distortion': 0.0, 'noise_level': 0.05,
                'bit_crush': 0.0, 'filter_cutoff': 3000, 'filter_resonance': 0.1, 'vibrato_rate': 4.0, 'vibrato_depth': 0.1,
                'chorus_depth': 0.15, 'chorus_rate': 0.2, 'formant1_freq': 800, 'formant1_amp': 1.0,
                'formant2_freq': 1150, 'formant2_amp': 0.6, 'formant3_freq': 2800, 'formant3_amp': 0.3
            }
        }
        return presets.get(type_name, {})

class WaveExperimenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Wave Experimenter")
        self.setGeometry(100, 100, 1280, 360)

        self.sample_rate = 44100
        self.duration = 2.0
        self.frequency = 440.0
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))

        self.instrument_presets = InstrumentPresets()
        self.wave_shapes = {
            'sine': lambda t, freq: np.sin(2 * np.pi * freq * t),
            'square': lambda t, freq: signal.square(2 * np.pi * freq * t),
            'sawtooth': lambda t, freq: signal.sawtooth(2 * np.pi * freq * t),
            'triangle': lambda t, freq: signal.sawtooth(2 * np.pi * freq * t, 0.5),
            'noise': self.noise_generator,
            'formant': self.formant_generator
        }

        self.params = {
            'wave_shape1': 'sine', 'wave_shape2': 'sine', 'wave_mix': 0.0, 'frequency_multiplier': 1.0,
            'harm1_weight': 1.0, 'harm2_weight': 0.5, 'harm3_weight': 0.25, 'harm4_weight': 0.125, 'harm5_weight': 0.0625,
            'attack_time': 0.1, 'decay_time': 0.2, 'sustain_level': 0.7, 'release_time': 0.3,
            'vibrato_rate': 5.0, 'vibrato_depth': 0.0, 'fm_frequency': 0.0, 'fm_depth': 0.0,
            'distortion': 0.0, 'noise_level': 0.0, 'bit_crush': 0.0, 'filter_cutoff': 20000, 'filter_resonance': 0.0,
            'formant1_freq': 500, 'formant1_amp': 1.0, 'formant2_freq': 1500, 'formant2_amp': 0.5,
            'formant3_freq': 2500, 'formant3_amp': 0.25, 'chorus_depth': 0.0, 'chorus_rate': 0.5
        }

        pygame.mixer.init(frequency=self.sample_rate)
        self.sound = None
        self.is_looping = False
        self.play_position = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_play_position)
        self.setup_gui()

    def noise_generator(self, t, freq):
        noise = np.random.normal(0, 1, len(t))
        if self.params['filter_cutoff'] < 20000:
            b, a = signal.butter(2, self.params['filter_cutoff'] / (self.sample_rate / 2), btype='low')
            noise = signal.filtfilt(b, a, noise)
        return noise

    def formant_generator(self, t, freq):
        base_wave = np.sin(2 * np.pi * freq * t)
        formants = np.zeros_like(t)
        for i in range(1, 4):
            f_freq = self.params[f'formant{i}_freq']
            f_amp = self.params[f'formant{i}_amp']
            formants += f_amp * np.sin(2 * np.pi * f_freq * t)
        return base_wave * (1 + formants) / 2

    def apply_preset(self, category, preset_name):
        preset_getters = {
            'Percussion': self.instrument_presets.get_percussion_preset,
            'Wind': self.instrument_presets.get_wind_preset,
            'Vocal': self.instrument_presets.get_vocal_preset,
        }
        if category in preset_getters:
            preset = preset_getters[category](preset_name)
            for param, value in preset.items():
                if param in self.params:
                    self.params[param] = value
                    if param in self.sliders:
                        scale = 100 if any(x in param for x in ['weight', 'level', 'depth', 'amp']) else \
                                100 if 'time' in param else 1
                        self.sliders[param].setValue(int(value * scale))
            self.frequency = preset.get('frequency', self.frequency)
        self.update_all()

    def setup_gui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        # Podgląd fali
        self.wave_view = QGraphicsView()
        self.wave_scene = QGraphicsScene()
        self.wave_view.setScene(self.wave_scene)
        self.wave_view.setFixedHeight(150)
        main_layout.addWidget(self.wave_view)

        control_layout = QHBoxLayout()

        # Lewa część: suwaki
        left_layout = QVBoxLayout()
        self.sliders = {}

        param_groups = {
            'Wave': ['wave_mix', 'frequency_multiplier'],
            'Harmonics': [f'harm{i}_weight' for i in range(1, 6)],
            'ADSR': ['attack_time', 'decay_time', 'sustain_level', 'release_time'],
            'Effects': ['vibrato_rate', 'vibrato_depth', 'fm_frequency', 'fm_depth', 
                        'distortion', 'noise_level', 'bit_crush', 'chorus_depth', 'chorus_rate'],
            'Filter': ['filter_cutoff', 'filter_resonance'],
            'Formants': ['formant1_freq', 'formant1_amp', 'formant2_freq', 'formant2_amp', 
                        'formant3_freq', 'formant3_amp']
        }

        for group_name, params in param_groups.items():
            group = QGroupBox(group_name)
            group_layout = QVBoxLayout()
            for param in params:
                slider_layout = QHBoxLayout()
                label = QLabel(param)
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(0, 100 if any(x in param for x in ['weight', 'level', 'depth', 'amp']) else 
                                200 if 'time' in param else 2000 if 'freq' in param else 100)
                slider.setValue(int(self.params[param] * (100 if any(x in param for x in ['weight', 'level', 'depth', 'amp']) else 
                                                        100 if 'time' in param else 1)))
                slider.valueChanged.connect(lambda val, p=param: self.update_param(p, val))
                slider_layout.addWidget(label)
                slider_layout.addWidget(slider)
                group_layout.addLayout(slider_layout)
                self.sliders[param] = slider
            group.setLayout(group_layout)
            left_layout.addWidget(group)
        control_layout.addLayout(left_layout)

        # Prawa część: wybór presetów i fal
        right_layout = QVBoxLayout()

        wave_group = QGroupBox("Wave Shapes")
        wave_layout = QVBoxLayout()
        self.wave1_radios = {}
        for shape in self.wave_shapes.keys():
            radio1 = QRadioButton(shape)
            radio1.toggled.connect(lambda checked, s=shape: self.update_wave_shape1(s) if checked else None)
            self.wave1_radios[shape] = radio1
            wave_layout.addWidget(radio1)
            if shape == 'sine':
                radio1.setChecked(True)
        wave_group.setLayout(wave_layout)
        right_layout.addWidget(wave_group)

        preset_group = QGroupBox("Presets")
        preset_layout = QVBoxLayout()
        self.category_combo = QComboBox()
        self.preset_combo = QComboBox()
        categories = self.instrument_presets.get_preset_categories()
        self.category_combo.addItems(categories.keys())
        self.category_combo.currentTextChanged.connect(self.update_presets)
        self.preset_combo.currentTextChanged.connect(self.apply_selected_preset)
        self.update_presets(list(categories.keys())[0])
        preset_layout.addWidget(self.category_combo)
        preset_layout.addWidget(self.preset_combo)
        preset_group.setLayout(preset_layout)
        right_layout.addWidget(preset_group)

        # Przyciski
        play_btn = QPushButton("Play")
        play_btn.clicked.connect(self.play_sound)
        loop_btn = QPushButton("Loop Play")
        loop_btn.clicked.connect(self.toggle_loop)
        save_btn = QPushButton("Save WAV")
        save_btn.clicked.connect(self.save_wave)
        right_layout.addWidget(play_btn)
        right_layout.addWidget(loop_btn)
        right_layout.addWidget(save_btn)
        control_layout.addLayout(right_layout)

        main_layout.addLayout(control_layout)
        self.update_all()

    def update_param(self, param, value):
        scale = 1.0 if 'freq' in param else 0.01 if any(x in param for x in ['time', 'depth', 'level', 'amp']) else 1.0
        self.params[param] = value * scale
        self.update_all()

    def update_wave_shape1(self, shape):
        self.params['wave_shape1'] = shape
        self.update_all()

    def update_presets(self, category):
        self.preset_combo.clear()
        self.preset_combo.addItems(self.instrument_presets.get_preset_categories()[category])

    def apply_selected_preset(self, preset_name):
        category = self.category_combo.currentText()
        self.apply_preset(category, preset_name)

    def generate_wave(self):
        current_frequency = self.frequency * self.params['frequency_multiplier']
        wave1 = self.wave_shapes[self.params['wave_shape1']](self.t, current_frequency)
        wave2 = self.wave_shapes[self.params['wave_shape2']](self.t, current_frequency)
        wave = wave1 * (1 - self.params['wave_mix']) + wave2 * self.params['wave_mix']

        harmonic_wave = np.zeros_like(self.t)
        for i in range(1, 6):
            weight = self.params[f'harm{i}_weight']
            harmonic_wave += weight * np.sin(2 * np.pi * current_frequency * i * self.t)
        wave = wave * 0.5 + harmonic_wave * 0.5

        if self.params['fm_depth'] > 0:
            fm_mod = self.params['fm_depth'] * np.sin(2 * np.pi * self.params['fm_frequency'] * self.t)
            wave = np.sin(2 * np.pi * current_frequency * self.t + fm_mod)

        if self.params['vibrato_depth'] > 0:
            vibrato = self.params['vibrato_depth'] * np.sin(2 * np.pi * self.params['vibrato_rate'] * self.t)
            wave = np.sin(2 * np.pi * current_frequency * self.t + vibrato)

        if self.params['noise_level'] > 0:
            wave += np.random.normal(0, self.params['noise_level'], len(wave))

        if self.params['distortion'] > 0:
            wave = np.tanh(wave * (1 + self.params['distortion'] * 10))

        if self.params['bit_crush'] > 0:
            levels = 2 ** (16 - int(self.params['bit_crush'] * 14))
            wave = np.round(wave * levels) / levels

        if self.params['chorus_depth'] > 0:
            delay = 0.03 * self.params['chorus_depth']
            delayed_wave = np.interp(self.t - delay, self.t, wave, left=0, right=0)
            wave += delayed_wave * self.params['chorus_rate']

        wave *= self.apply_adsr()
        return wave / np.max(np.abs(wave))

    def apply_adsr(self):
        length = len(self.t)
        attack_samples = int(self.params['attack_time'] * length)
        decay_samples = int(self.params['decay_time'] * length)
        release_samples = int(self.params['release_time'] * length)
        sustain_samples = length - attack_samples - decay_samples - release_samples
        attack = np.linspace(0, 1, attack_samples)
        decay = np.linspace(1, self.params['sustain_level'], decay_samples)
        sustain = np.ones(sustain_samples) * self.params['sustain_level']
        release = np.linspace(self.params['sustain_level'], 0, release_samples)
        return np.concatenate([attack, decay, sustain, release])

    def update_all(self):
        self.update_waveform()
        if self.is_looping:
            self.stop_sound()
            self.start_loop()

    def update_waveform(self):
        wave = self.generate_wave()
        self.wave_scene.clear()
        width = 1280
        height = 150
        samples = wave[:min(2000, len(wave))]
        step = width / len(samples)

        # Rysowanie osi
        self.wave_scene.addLine(0, height / 2, width, height / 2, QPen(Qt.GlobalColor.gray))  # Oś X (czas)
        self.wave_scene.addLine(0, 0, 0, height, QPen(Qt.GlobalColor.gray))  # Oś Y (amplituda)

        # Rysowanie fali
        path = QPainterPath()
        path.moveTo(0, height / 2)
        for i, sample in enumerate(samples):
            x = i * step
            y = height / 2 - (sample * height / 2)
            path.lineTo(x, y)
        self.wave_scene.addPath(path, QPen(Qt.GlobalColor.blue, 1))

        # Wskaźnik odtwarzania
        self.play_line = self.wave_scene.addLine(0, 0, 0, height, QPen(Qt.GlobalColor.red, 2))
        self.wave_view.setSceneRect(0, 0, width, height)

    def update_play_position(self):
        if self.is_looping:
            self.play_position += 10 / (self.duration * 1000)  # Przesuwaj co 10 ms
            if self.play_position >= 1:
                self.play_position = 0
                self.start_loop()  # Ponowne odtworzenie po zakończeniu
            x = self.play_position * 1280
            self.play_line.setLine(x, 0, x, 150)

    def play_sound(self):
        wave = self.generate_wave()
        wave_stereo = np.vstack((wave, wave)).T.astype(np.float32)
        wave_int16 = np.int16(wave_stereo * 32767).copy(order='C')
        self.sound = pygame.sndarray.make_sound(wave_int16)
        self.sound.play()

    def toggle_loop(self):
        if not self.is_looping:
            self.start_loop()
            self.is_looping = True
            self.timer.start(10)  # Aktualizacja co 10 ms
        else:
            self.stop_sound()
            self.is_looping = False
            self.timer.stop()
            self.play_position = 0
            self.update_waveform()

    def start_loop(self):
        wave = self.generate_wave()
        wave_stereo = np.vstack((wave, wave)).T.astype(np.float32)
        wave_int16 = np.int16(wave_stereo * 32767).copy(order='C')
        self.sound = pygame.sndarray.make_sound(wave_int16)
        self.sound.play(-1)  # -1 oznacza nieskończoną pętlę

    def stop_sound(self):
        if self.sound:
            self.sound.stop()

    def save_wave(self):
        wave = self.generate_wave()
        wave_stereo = np.vstack((wave, wave)).T.astype(np.float32)
        wave_int16 = np.int16(wave_stereo * 32767).copy(order='C')
        filename = f"custom_wave_{self.frequency}Hz.wav"
        wavfile.write(filename, self.sample_rate, wave_int16)
        print(f"Saved to {filename}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WaveExperimenter()
    window.show()
    sys.exit(app.exec())
