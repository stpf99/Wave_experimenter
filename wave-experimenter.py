import numpy as np
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
import pygame
import os

class InstrumentPresets:
    """Klasa zawierająca presety dla różnych typów instrumentów"""
    
    @staticmethod
    def get_preset_categories():
        return {
            'Percussion': ['Kick', 'Snare', 'HiHat', 'Tom', 'Cymbal', 'Maracas'],
            'Wind': ['Flute', 'Trumpet', 'Saxophone', 'Clarinet'],
            'Plucked': ['Guitar', 'Bass', 'Harp'],
            'Vocal': ['Choir', 'Voice', 'Vowel'],
            'Synth': ['Lead', 'Pad', 'Bass', 'Pluck'],
            'World': ['Sitar', 'Gamelan', 'Didgeridoo']
        }

    @staticmethod
    def get_percussion_preset(type_name):
        presets = {
            'Kick': {
                'wave_shape1': 'sine',
                'wave_shape2': 'square',
                'wave_mix': 0.2,
                'frequency_multiplier': 1.0,
                'attack_time': 0.01,
                'decay_time': 0.1,
                'sustain_level': 0.0,
                'release_time': 0.1,
                'harm1_weight': 1.0,
                'harm2_weight': 0.5,
                'harm3_weight': 0.1,
                'frequency': 60,
                'fm_frequency': 50,
                'fm_depth': 0.3,
            },
            'Snare': {
                'wave_shape1': 'noise',
                'wave_shape2': 'sine',
                'wave_mix': 0.7,
                'frequency_multiplier': 1.0,
                'attack_time': 0.001,
                'decay_time': 0.1,
                'sustain_level': 0.1,
                'release_time': 0.2,
                'noise_level': 0.6,
                'frequency': 200,
            },
            'HiHat': {
                'wave_shape1': 'noise',
                'wave_mix': 1.0,
                'attack_time': 0.001,
                'decay_time': 0.05,
                'sustain_level': 0.0,
                'release_time': 0.1,
                'noise_level': 0.8,
                'bit_crush': 0.3,
                'frequency': 8000,
            },
            'Maracas': {
                'wave_shape1': 'noise',
                'wave_mix': 1.0,
                'attack_time': 0.001,
                'decay_time': 0.03,
                'sustain_level': 0.0,
                'release_time': 0.05,
                'noise_level': 0.9,
                'bit_crush': 0.5,
                'frequency': 6000,
                'filter_cutoff': 8000,
                'filter_resonance': 0.1,
            }
        }
        return presets.get(type_name, {})

    @staticmethod
    def get_wind_preset(type_name):
        presets = {
            'Flute': {
                'wave_shape1': 'sine',
                'wave_shape2': 'triangle',
                'wave_mix': 0.2,
                'attack_time': 0.1,
                'decay_time': 0.1,
                'sustain_level': 0.8,
                'release_time': 0.2,
                'harm1_weight': 1.0,
                'harm2_weight': 0.3,
                'harm3_weight': 0.1,
                'vibrato_rate': 5.0,
                'vibrato_depth': 0.1,
            },
            'Trumpet': {
                'wave_shape1': 'square',
                'wave_shape2': 'sawtooth',
                'wave_mix': 0.3,
                'attack_time': 0.05,
                'decay_time': 0.1,
                'sustain_level': 0.7,
                'release_time': 0.15,
                'harm1_weight': 1.0,
                'harm2_weight': 0.7,
                'harm3_weight': 0.5,
                'harm4_weight': 0.3,
                'distortion': 0.2,
            }
        }
        return presets.get(type_name, {})

    @staticmethod
    def get_vocal_preset(type_name):
        presets = {
            'Choir': {
                'wave_shape1': 'sine',
                'wave_shape2': 'triangle',
                'wave_mix': 0.3,
                'attack_time': 0.2,
                'decay_time': 0.1,
                'sustain_level': 0.8,
                'release_time': 0.3,
                'harm1_weight': 1.0,
                'harm2_weight': 0.5,
                'harm3_weight': 0.3,
                'harm4_weight': 0.2,
                'harm5_weight': 0.1,
                'vibrato_rate': 5.0,
                'vibrato_depth': 0.1,
                'formant_frequencies': [500, 1500, 2500],
                'formant_amplitudes': [1.0, 0.5, 0.25],
            },
            'Vowel': {
                'wave_shape1': 'sine',
                'wave_shape2': 'triangle',
                'wave_mix': 0.2,
                'attack_time': 0.1,
                'decay_time': 0.1,
                'sustain_level': 0.7,
                'release_time': 0.2,
                'formant_presets': {
                    'a': [800, 1150, 2800, 3500, 4950],
                    'e': [350, 1900, 2500, 3500, 4950],
                    'i': [270, 2300, 3000, 3500, 4950],
                    'o': [450, 800, 2830, 3500, 4950],
                    'u': [325, 700, 2530, 3500, 4950]
                }
            }
        }
        return presets.get(type_name, {})
        
class WaveExperimenter:
    def __init__(self):
        self.sample_rate = 44100
        self.duration = 2.0
        self.frequency = 440.0  # A4 default
        
        
        # Dodanie presetów instrumentów
        self.instrument_presets = InstrumentPresets()
        self.current_preset = None
        
        # Dodanie presetów częstotliwości
        self.frequency_ranges = {
            'Sub Bass': 20,      # 20-60 Hz
            'Bass': 60,          # 60-250 Hz
            'Low Mids': 250,     # 250-500 Hz
            'Mids': 500,         # 500-2000 Hz
            'High Mids': 2000,   # 2000-4000 Hz
            'Presence': 4000,    # 4000-6000 Hz
            'Brilliance': 6000,  # 6000-20000 Hz
        }
        
        self.t = np.linspace(0, self.duration, int(self.sample_rate * self.duration))
        
        # Podstawowe kształty fal
        self.wave_shapes = {
            'sine': lambda t, freq: np.sin(2 * np.pi * freq * t),
            'square': lambda t, freq: signal.square(2 * np.pi * freq * t),
            'sawtooth': lambda t, freq: signal.sawtooth(2 * np.pi * freq * t),
            'triangle': lambda t, freq: signal.sawtooth(2 * np.pi * freq * t, 0.5),
            'pulse': lambda t, freq: signal.square(2 * np.pi * freq * t, duty=self.params['pulse_width']),
            'custom': self.custom_wave
        }
        
        # Rozszerzenie wave_shapes o nowe typy
        self.wave_shapes.update({
            'noise': self.noise_generator,
            'formant': self.formant_generator,
        })
        
        # Parametry do eksperymentowania
        self.params = {
            # Podstawowe parametry fali
            'wave_shape1': 'sine',
            'wave_shape2': 'sine',
            'wave_mix': 0.0,
            'pulse_width': 0.5,
            'frequency_multiplier': 1.0,  # Nowy parametr do dostrajania częstotliwości
            
            # Harmoniczne i ich fazy
            'harm1_weight': 1.0,
            'harm2_weight': 0.5,
            'harm3_weight': 0.25,
            'harm4_weight': 0.125,
            'harm5_weight': 0.0625,
            'phase_shift2': 0.0,
            'phase_shift3': 0.0,
            'phase_shift4': 0.0,
            'phase_shift5': 0.0,
            
            # ADSR
            'attack_time': 0.1,
            'decay_time': 0.2,
            'sustain_level': 0.7,
            'release_time': 0.3,
            
            # Modulacje
            'vibrato_rate': 5.0,
            'vibrato_depth': 0.0,
            'tremolo_rate': 6.0,
            'tremolo_depth': 0.0,
            'fm_frequency': 0.0,
            'fm_depth': 0.0,
            'am_frequency': 0.0,
            'am_depth': 0.0,
            
            # Zniekształcenia i efekty
            'distortion': 0.0,
            'noise_level': 0.0,
            'bit_crush': 0.0,
            'fold_amount': 0.0,
            
            # Parametry niestandardowej fali
            'custom_param1': 0.5,
            'custom_param2': 0.5,
            'custom_param3': 0.5,
        }

        # Dodanie nowych parametrów dla filtrów i formantów
        self.params.update({
            'filter_cutoff': 20000,
            'filter_resonance': 0.0,
            'formant1_freq': 500,
            'formant1_amp': 1.0,
            'formant2_freq': 1500,
            'formant2_amp': 0.5,
            'formant3_freq': 2500,
            'formant3_amp': 0.25,
        })

        pygame.mixer.init(frequency=self.sample_rate)
        self.setup_gui()

    def noise_generator(self, t, freq):
        """Generator szumu z możliwością filtrowania"""
        noise = np.random.normal(0, 1, len(t))
        if self.params['filter_cutoff'] < 20000:
            # Implementacja prostego filtru dolnoprzepustowego
            b, a = signal.butter(2, self.params['filter_cutoff'] / (self.sample_rate/2))
            noise = signal.filtfilt(b, a, noise)
        return noise

    def formant_generator(self, t, freq):
        """Generator dźwięku z formantami"""
        base_wave = np.sin(2 * np.pi * freq * t)
        formants = np.zeros_like(t)
        
        for i in range(1, 4):
            formant_freq = self.params[f'formant{i}_freq']
            formant_amp = self.params[f'formant{i}_amp']
            formants += formant_amp * np.sin(2 * np.pi * formant_freq * t)
        
        return base_wave * (1 + formants) / 2

    def apply_preset(self, category, preset_name):
        """Aplikuje preset dla wybranego instrumentu"""
        preset_getters = {
            'Percussion': self.instrument_presets.get_percussion_preset,
            'Wind': self.instrument_presets.get_wind_preset,
            'Vocal': self.instrument_presets.get_vocal_preset,
        }
        
        if category in preset_getters:
            preset = preset_getters[category](preset_name)
            if preset:
                for param, value in preset.items():
                    if param in self.params:
                        self.params[param] = value
                        if param in self.sliders:
                            self.sliders[param].set_val(value)
                self.current_preset = (category, preset_name)
                self.update(None)

    def setup_gui(self):
        """Tworzy rozszerzony interfejs do eksperymentowania"""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Wykresy
        self.ax_wave = plt.subplot2grid((3, 4), (0, 0), colspan=3)
        self.ax_spectrum = plt.subplot2grid((3, 4), (1, 0), colspan=3)
        
        # Wybór kształtu fali
        self.ax_wave_shape1 = plt.subplot2grid((3, 4), (0, 3))
        self.ax_wave_shape2 = plt.subplot2grid((3, 4), (1, 3))
        
        # Dodanie wyboru zakresu częstotliwości
        self.ax_freq_range = plt.subplot2grid((3, 4), (2, 3))
        self.radio_freq_range = RadioButtons(self.ax_freq_range, 
                                           list(self.frequency_ranges.keys()), 
                                           active=3)  # Domyślnie 'Mids'
        self.radio_freq_range.on_clicked(self.update_frequency_range)
        
        wave_shapes = list(self.wave_shapes.keys())
        self.radio_shape1 = RadioButtons(self.ax_wave_shape1, wave_shapes, active=0)
        self.radio_shape2 = RadioButtons(self.ax_wave_shape2, wave_shapes, active=0)
        
        self.radio_shape1.on_clicked(self.update_wave_shape1)
        self.radio_shape2.on_clicked(self.update_wave_shape2)
        
        # Dodanie wyboru presetów
        categories = self.instrument_presets.get_preset_categories()
        ax_preset_cat = plt.axes([0.85, 0.7, 0.1, 0.2])
        self.preset_category = RadioButtons(ax_preset_cat, list(categories.keys()))
        
        ax_preset = plt.axes([0.85, 0.4, 0.1, 0.2])
        self.preset_selection = RadioButtons(ax_preset, categories['Percussion'])
        
        def update_presets(category):
            self.preset_selection.labels = categories[category]
            self.preset_selection.active = 0
            plt.draw()
        
        self.preset_category.on_clicked(update_presets)
        self.preset_selection.on_clicked(
            lambda name: self.apply_preset(self.preset_category.value_selected, name)
        )        

        # Suwaki parametrów
        plt.subplots_adjust(left=0.1, bottom=0.3)
        slider_height = 0.015
        slider_spacing = 0.02
        current_y = 0.05
        self.sliders = {}
        
        # Dodanie suwaka do dostrajania częstotliwości
        ax_freq_mult = plt.axes([0.1, 0.25, 0.15, slider_height])
        self.sliders['frequency_multiplier'] = Slider(ax_freq_mult, 'Freq Fine Tune', 0.5, 2.0, valinit=1.0)
        self.sliders['frequency_multiplier'].on_changed(self.update)
        
        # Grupowanie parametrów
        param_groups = {
            'Podstawowe': ['wave_mix', 'pulse_width'],
            'Harmoniczne': [f'harm{i}_weight' for i in range(1, 6)] + [f'phase_shift{i}' for i in range(2, 6)],
            'ADSR': ['attack_time', 'decay_time', 'sustain_level', 'release_time'],
            'Modulacje': ['vibrato_rate', 'vibrato_depth', 'tremolo_rate', 'tremolo_depth',
                         'fm_frequency', 'fm_depth', 'am_frequency', 'am_depth'],
            'Efekty': ['distortion', 'noise_level', 'bit_crush', 'fold_amount'],
            'Custom': ['custom_param1', 'custom_param2', 'custom_param3']
        }
        
        current_x = 0.1
        for group_name, params in param_groups.items():
            for param in params:
                ax = plt.axes([current_x, current_y, 0.15, slider_height])
                slider = Slider(ax, param, 0.0, 
                              1.0 if any(word in param for word in ['weight', 'level', 'depth', 'mix', 'amount', 'param']) 
                              else 2.0 if 'time' in param 
                              else 20.0 if 'rate' in param or 'frequency' in param 
                              else 2*np.pi if 'phase' in param 
                              else 1.0,
                              valinit=self.params[param])
                slider.on_changed(self.update)
                self.sliders[param] = slider
                current_y += slider_spacing
            
            if current_y > 0.25:
                current_x += 0.2
                current_y = 0.05
        
        # Przyciski
        ax_play = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.btn_play = Button(ax_play, 'Play')
        self.btn_play.on_clicked(self.play_sound)
        
        ax_save = plt.axes([0.8, 0.1, 0.1, 0.04])
        self.btn_save = Button(ax_save, 'Save WAV')
        self.btn_save.on_clicked(self.save_wave)
        
        self.update(None)
        plt.show()

    def update_frequency_range(self, label):
        """Aktualizuje częstotliwość bazową na podstawie wybranego zakresu"""
        self.frequency = self.frequency_ranges[label]
        self.update(None)

    def generate_wave(self):
        """Generuje falę z aktualnymi parametrami"""
        current_frequency = self.frequency * self.params['frequency_multiplier']
        
        # Generowanie podstawowych kształtów fal
        wave1 = self.wave_shapes[self.params['wave_shape1']](self.t, current_frequency)
        wave2 = self.wave_shapes[self.params['wave_shape2']](self.t, current_frequency)
        
        # Mieszanie fal
        wave = wave1 * (1 - self.params['wave_mix']) + wave2 * self.params['wave_mix']
        
        # Dodawanie harmonicznych
        harmonic_wave = np.zeros_like(self.t)
        for i in range(1, 6):
            phase_shift = self.params.get(f'phase_shift{i}', 0.0) if i > 1 else 0.0
            weight = self.params[f'harm{i}_weight']
            harmonic_wave += weight * np.sin(2 * np.pi * self.frequency * i * self.t + phase_shift)
        
        wave = wave * 0.5 + harmonic_wave * 0.5
        wave = wave / np.max(np.abs(wave))
        
        # Modulacja FM
        if self.params['fm_depth'] > 0:
            fm_mod = self.params['fm_depth'] * np.sin(2 * np.pi * self.params['fm_frequency'] * self.t)
            wave = np.sin(2 * np.pi * self.frequency * self.t + fm_mod)
        
        # Modulacja AM
        if self.params['am_depth'] > 0:
            am_mod = 1 + self.params['am_depth'] * np.sin(2 * np.pi * self.params['am_frequency'] * self.t)
            wave *= am_mod
        
        # Wave folding
        if self.params['fold_amount'] > 0:
            wave = np.sin(wave * np.pi * self.params['fold_amount'])
        
        # Bit crushing
        if self.params['bit_crush'] > 0:
            levels = 2 ** (16 - int(self.params['bit_crush'] * 14))
            wave = np.round(wave * levels) / levels
        
        # Pozostałe efekty (vibrato, tremolo, szum, zniekształcenia)
        if self.params['vibrato_depth'] > 0:
            vibrato = self.params['vibrato_depth'] * np.sin(2 * np.pi * self.params['vibrato_rate'] * self.t)
            phase_mod = 2 * np.pi * self.t * vibrato
            wave = np.sin(2 * np.pi * self.frequency * self.t + phase_mod)
        
        if self.params['tremolo_depth'] > 0:
            tremolo = 1 + self.params['tremolo_depth'] * np.sin(2 * np.pi * self.params['tremolo_rate'] * self.t)
            wave *= tremolo
        
        if self.params['noise_level'] > 0:
            noise = np.random.normal(0, self.params['noise_level'], len(wave))
            wave += noise
        
        if self.params['distortion'] > 0:
            wave = np.tanh(wave * (1 + self.params['distortion'] * 10)) / (1 + self.params['distortion'])
        
        # Aplikacja ADSR
        wave *= self.apply_adsr()
        
        return wave / np.max(np.abs(wave))

    def custom_wave(self, t, freq):
        """
        Generuje niestandardową falę bazową na podstawie parametrów
        Możesz dostosować tę funkcję do własnych potrzeb
        """
        p1 = self.params['custom_param1']
        p2 = self.params['custom_param2']
        p3 = self.params['custom_param3']
        
        # Przykład złożonej fali
        wave = (np.sin(2 * np.pi * freq * t) * p1 +
                np.sin(2 * np.pi * freq * t * 2) * p2 * (1 - p1) +
                signal.sawtooth(2 * np.pi * freq * t) * p3)
        return wave / np.max(np.abs(wave))
        
    def update_wave_shape1(self, label):
        self.params['wave_shape1'] = label
        self.update(None)
    
    def update_wave_shape2(self, label):
        self.params['wave_shape2'] = label
        self.update(None)
    
    def update(self, _):
        """Aktualizuje wyświetlanie po zmianie parametrów"""
        for param, slider in self.sliders.items():
            self.params[param] = slider.val
            
        wave = self.generate_wave()
        
        # Aktualizacja wykresu fali
        self.ax_wave.clear()
        self.ax_wave.plot(self.t[:2000], wave[:2000])
        self.ax_wave.set_title('Kształt fali')
        self.ax_wave.set_xlabel('Czas [s]')
        self.ax_wave.set_ylabel('Amplituda')
        
        # Aktualizacja widma częstotliwości
        self.ax_spectrum.clear()
        spectrum = np.abs(np.fft.fft(wave))
        freqs = np.fft.fftfreq(len(wave), 1/self.sample_rate)
        self.ax_spectrum.plot(freqs[:len(freqs)//2], spectrum[:len(spectrum)//2])
        self.ax_spectrum.set_title('Widmo częstotliwości')
        self.ax_spectrum.set_xlabel('Częstotliwość [Hz]')
        self.ax_spectrum.set_ylabel('Amplituda')
        self.ax_spectrum.set_xlim(0, 5000)  # Pokazuje częstotliwości do 5kHz
        
        plt.draw()

    def play_sound(self, _):
        """Odtwarza wygenerowany dźwięk w stereo"""
        wave = self.generate_wave()
        # Konwersja do stereo przez zduplikowanie kanału mono
        stereo_shape = (len(wave), 2)
        wave_stereo = np.zeros(stereo_shape, dtype=np.float32)
        wave_stereo[:, 0] = wave  # lewy kanał
        wave_stereo[:, 1] = wave  # prawy kanał
        
        # Konwersja do int16 i upewnienie się, że tablica jest C-contiguous
        wave_int16 = np.int16(wave_stereo * 32767).copy(order='C')
        sound = pygame.sndarray.make_sound(wave_int16)
        sound.play()
    
    def save_wave(self, _):
        """Zapisuje wygenerowany dźwięk do pliku WAV w stereo"""
        wave = self.generate_wave()
        # Konwersja do stereo
        stereo_shape = (len(wave), 2)
        wave_stereo = np.zeros(stereo_shape, dtype=np.float32)
        wave_stereo[:, 0] = wave  # lewy kanał
        wave_stereo[:, 1] = wave  # prawy kanał
        
        wave_int16 = np.int16(wave_stereo * 32767)
        filename = f"custom_wave_{self.frequency}Hz.wav"
        wavfile.write(filename, self.sample_rate, wave_int16)
        print(f"Zapisano do {filename}")

    def apply_adsr(self):
        """Aplikuje obwiednię ADSR"""
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

if __name__ == "__main__":
    experimenter = WaveExperimenter()
