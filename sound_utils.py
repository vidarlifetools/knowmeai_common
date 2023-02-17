import librosa
import librosa.display
import numpy as np
import pickle
from utilities.pyrapt import pitch
from scipy.ndimage.interpolation import shift
from feature_constants import\
    sound_mean_norm,\
    sound_windowing,\
    sound_pre_emphasis,\
    sound_remove_silent,\
    sound_noise_reduce,\
    sound_n_mfcc,\
    sound_sample_rate,\
    sound_window_length,\
    sound_window_step,\
    sound_feature_length,\
    sound_feature_step,\
    sound_ampl_normalization,\
    sound_use_pitch


class SoundFeature:
    def __init__(self):
        self.win_size = sound_window_length
        self.win_step = sound_window_step
        self.feat_size = sound_feature_length
        self.feat_step = sound_feature_step
        self.sr = sound_sample_rate
        self.n_mfcc = sound_n_mfcc
        self.window = "hann" if sound_windowing else None
        self.pre_emphasis = sound_pre_emphasis
        self.mean_normalization = sound_mean_norm
        self.noise_reduce = sound_noise_reduce
        self.ampl_normalization = sound_ampl_normalization
        self.use_pitch = sound_use_pitch
        # frame_step_size in raptparams is set to 0.016
        self.pitch_buffer = np.zeros((
            int(self.feat_size/0.016),), dtype=float)

    def get_mfcc(self, samples):
        #def sound_feature(samples, sr, win_size, stp_size, feature_size, n_mfcc, windowing=True, mean_norm=False):
        if len(samples) < self.feat_size * self.sr or self.n_mfcc > 13:
            print("Sound features: ", len(samples), self.feat_size, self.sr)
            return np.zeros(((self.n_mfcc) * int(self.feat_size / self.win_step)))

        # Use only mfcc coefficients 1 to 12, ignore 0, and ignore the last set of coefficients
        mfcc = librosa.feature.mfcc(y=samples,
                                    sr=self.sr,
                                    hop_length=int(self.win_step * self.sr),
                                    n_fft=int(self.win_size * self.sr),
                                    window=self.window,
                                    n_mfcc=13)[13 - self.n_mfcc:13, 0:int(len(samples) / (self.sr * self.win_step))]

        if self.mean_normalization:
            mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

        mfcc = mfcc.T
        # The faeture is a 1 dim array: [mfcc[0,0], mfcc[0,1], . . . . .mfcc[31,11]
        mfcc = np.resize(mfcc, (mfcc.shape[1] * mfcc.shape[0]))
        return mfcc
    def get_pitch(self, samples):
        # Use the last feat_step samples to calculate pitch values and add it to the pitch buffer. The pitch
        # buffer will contain pitch values for the last feat_size samples
        pitch_feats = pitch(samples[3*int(self.feat_step*self.sr):4*int(self.feat_step*self.sr)], self.sr)
        # Make it 32 elements to match the feature size when the whole audio sequence is analyzed
        pitch_feats = np.append(pitch_feats, 0.0)
        shift(self.pitch_buffer, -8, cval=0.0)
        self.pitch_buffer[3*int(self.feat_step/0.016):4*int(self.feat_step/0.016)] = pitch_feats
        return self.pitch_buffer

    def get_feature(self, samples):
        feature = self.get_mfcc(samples)
        if self.use_pitch:
            pitch_feats = self.get_pitch(samples)
            feature = np.concatenate((feature, pitch_feats))
        return feature


class SoundPrediction:
    def __init__(self, model_filename_source, model_filename_client, model_filename_sound, histogram_depth, histogram_limit = 0.0):
        with open(model_filename_source, "rb") as file:
            self.source_model = pickle.load(file)
        with open(model_filename_client, "rb") as file:
            self.client_model = pickle.load(file)
        with open(model_filename_sound, "rb") as file:
            self.sound_model = pickle.load(file)
        self.histogram_limit = histogram_limit
        self.histogram_depth = histogram_depth
        self.histogram = np.zeros((histogram_depth, self.client_model.n_outputs_), dtype=float)
        self.hist_idx = 0
        print(f"Histogram limit = {histogram_limit},histogram depth = {histogram_depth}")

    def get_class(self, ft):
        if len(ft) != 416:
            b = np.zeros((416 - len(ft),), dtype=float)
            ft = np.append(ft, b)
        pred = self.source_model.predict_proba((ft,))
        cl_cl = self.client_model.predict((ft,))
        self.histogram[self.hist_idx] *= 0.0
        self.histogram[self.hist_idx, cl_cl] = pred[0, 0]
        self.hist_idx = (self.hist_idx + 1) % self.histogram_depth
        vector = np.sum(self.histogram, 0)
        valid = True
        if np.sum(vector) < self.histogram_limit:
            valid = False
            vector *= 0.0
        sound_class = self.sound_model.predict((vector,))
        return valid, sound_class

    def clear_histogram(self):
        self.histogram *= 0.0
        self.hist_idx = 0

