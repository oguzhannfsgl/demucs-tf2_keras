# This is a pretty naive implementation of a phase vocoder.
# When the augmentations are not strong(tempo scale is between 0.88 and 1.12 - pitch shift is between +-2 semitones) and
# fft_length and hop size is high enough, artifacts are not noticable.
# Normally, to reduce the smearing effect, real frequencies are found via phase unwrapping and frequencies scaled from there
# But, i didn't want to deal with these complex stuff.

from cfg import CFG
import tensorflow as tf
import tensorflow_io as tfio

def increase_tempo_on_stft(x, tempo):
    # Skip some time steps in time-frequency domain
    num_frames = x.shape[2]
    skip_per = 1 / (tempo-1)
    mask = tf.math.floormod(tf.range(num_frames, dtype=tf.float32), skip_per) >= 1
    x = tf.boolean_mask(x, mask, axis=2)
    x = tf.signal.inverse_stft(x, frame_length=CFG.FRAME_LENGTH, 
                                frame_step=CFG.FRAME_STEP) # shape: [channels, sources, samples]
    return x


def decrease_tempo_on_stft(x, tempo):
    # Repeat some time steps in time-frequency domain
    num_frames = x.shape[2]
    repeat_per = tempo / (1-tempo)
    mask = tf.math.floormod(tf.range(num_frames, dtype=tf.float32), repeat_per) >= 1
    repeat = tf.where(mask==True, 1, 2)
    x = tf.repeat(x, repeats=repeat, axis=2)
    x = tf.signal.inverse_stft(x, frame_length=CFG.FRAME_LENGTH, 
                                   frame_step=CFG.FRAME_STEP)
    return x
    
    
    
def tempo_shift(x, tempo):
    """
    Changes the tempo of the sound by resampling it in the time-frequency domain.
    You may think that we could resample the audio in time domain directly, but this also
    changes the pitch of the sound. To not affect the pitch we are working on overlapped frames of the sound.
        
    For further information, see https://en.wikipedia.org/wiki/Audio_time_stretching_and_pitch_scaling
    
    args:
        x(tf.Tensor): audio array with shape: [samples, channels, sources]
        tempo(float): fraction to change the tempo of the audio. Expected in range [0.88, 1.12]
    returns:
        x(tf.Tensor): audio array with changed tempo.
    """
    x = tf.transpose(x, [1,2,0]) # shape: [channels, sources, samples]
    x_stft = tf.signal.stft(x, frame_length=CFG.FRAME_LENGTH,
                            frame_step=CFG.FRAME_STEP, pad_end=True) # shape: [channels, sources, time, fft_bins]
    num_frames = x_stft.shape[2]
    x = tf.cond(tempo>1, lambda:increase_tempo_on_stft(x_stft, tempo), lambda:x)
    x = tf.cond(tempo<1, lambda:decrease_tempo_on_stft(x_stft, tempo), lambda:x)
    x = tf.transpose(x, [2,0,1])
    return x


def pitch_shift(x, shift=-2):
    """
    To apply pitch shifting while keeping the samples same, we are first changing the tempo(duration)
        of the sound and then resampling in time domain.
        
    args:
        x(tf.Tensor): audio array with shape: [samples, channels, sources]
        shift(int): how much semitones to raise or lower the pitch. expected in range [-2, +2]
    returns:
        x(tf.Tensor): pitch shifted audio array with shape: [samples, channels, sources]
    """
    num_samples = x.shape[0]
    tempo = 1. / 1.06 ** tf.cast(shift, tf.float32)
    x = tempo_shift(x, tempo)
    x = tf.transpose(x, (2, 0, 1))# sources, samples, channels
    x = tfio.audio.resample(x, tf.cast(1000, tf.int64), tf.cast(tempo*1000, tf.int64))
    x = tf.transpose(x, (1, 2, 0))
    return x

                


def pitch_tempo_shift_batch(x):
    """
    Applying the augmention to the batch with p=0.2
    """
    apply_pitch = tf.random.uniform([CFG.BATCH_SIZE], minval=0, maxval=1)
    apply_tempo = tf.random.uniform([CFG.BATCH_SIZE], minval=0, maxval=1)
    
    tempo = tf.random.uniform([CFG.BATCH_SIZE], minval=0.88, maxval=1.12)
    pitch = tf.random.uniform([CFG.BATCH_SIZE], minval=-2, maxval=3, dtype=tf.int32)
    
    x_batch = []
    for i in range(CFG.BATCH_SIZE):
        xi = x[i]
        xi = tf.cond(apply_pitch[i]<0.2, lambda:pitch_shift(xi,pitch[i]), lambda:xi)
        xi = tf.image.resize_with_crop_or_pad(xi, target_height=CFG.NUM_SAMPLES, target_width=CFG.NUM_CHANNELS)
        xi = tf.reshape(xi, (CFG.NUM_SAMPLES, CFG.NUM_CHANNELS, CFG.NUM_SOURCES))
        xi = tf.cond(apply_tempo[i]<0.2, lambda:tempo_shift(xi,tempo[i]), lambda:xi)
        xi = tf.image.resize_with_crop_or_pad(xi, target_height=CFG.NUM_SAMPLES_MODEL, target_width=CFG.NUM_CHANNELS)
        xi = tf.reshape(xi, (CFG.NUM_SAMPLES_MODEL, CFG.NUM_CHANNELS, CFG.NUM_SOURCES))
        x_batch.append(xi)
    x = tf.stack(x_batch, axis=0)
    return x