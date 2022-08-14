import os
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio

from cfg import CFG

def decode_audio(filename, begin_idx):
    """
    Takes filename/foldername as input which have wav files of each sources and decodes them.
    """
    clip_length = CFG.CLIP_LENGTH
    all_source_audio = []
    for source_name in ["vocals", "drums", "bass", "other"]:
        source_filename = filename + "/" + source_name + ".wav"
        x = tfio.audio.AudioIOTensor(source_filename, dtype=tf.int16)
        x = x[begin_idx:begin_idx + clip_length * CFG.SAMPLE_RATE]
        all_source_audio.append(x)
    all_source_audio  = tf.reshape(tf.stack(all_source_audio, axis=-1), (CFG.NUM_SAMPLES, CFG.NUM_CHANNELS, CFG.NUM_SOURCES))
    return all_source_audio


def to_float(x):
    """
    Convert dtype of audio-which is decoded as int16- into float32
    """
    x = tf.cast(x, tf.float32) / 2 ** 15
    return x


def create_labels(x):
    """
    For training and validation, labels are created inside the model,
    this is just for convention
    """
    x_mix = tf.reduce_sum(x, axis=-1)
    x_mix = tf.clip_by_value(x_mix, -1, 1)
    return x, x_mix


def swap_channels(x):
    """
    Randomly swap channels
    """
    # x: unbatched audio array of shape: [samples, channels, sources]
    x = tf.transpose(tf.random.shuffle(tf.transpose(x, [1, 0, 2])), [1,0,2])
    return x

def random_scale(x):
    """
    Scale the audio with a random scalar in the range [CFG.SCALE_MIN, CFG.SCALE_MAX]
    """
    # x: batched audio array of shape: [batch_size, samples, channels, sources]
    scale = tf.random.uniform([CFG.BATCH_SIZE, 1, 1, CFG.NUM_SOURCES], minval=CFG.SCALE_MIN, maxval=CFG.SCALE_MAX)
    x = x * scale
    x = tf.clip_by_value(x, -1, 1)
    return x


def swap_batch_sources(x_batch):
    """
    Create new mixes by swapping sources in the batch
    """
    new_batch = []
    for i in range(CFG.NUM_SOURCES):
        new_source_batch = tf.random.shuffle(x_batch[:,:,:,i])
        new_batch.append(new_source_batch)
    return tf.stack(new_batch, axis=-1)


def random_multiply(x):
    """
    Randomly multiply each source by +- 1
    """
    random_values = tf.random.uniform([CFG.BATCH_SIZE, 1, 1, CFG.NUM_SOURCES], minval=0, maxval=2, dtype=tf.int32)
    random_values = 2 * tf.cast(random_values, x.dtype) - 1
    x = x * random_values
    return x


def process_audio(x, augment):
    x = to_float(x)
    x = random_scale(x) if CFG.RANDOM_SCALE and augment else x
    x = swap_batch_sources(x) if CFG.SWAP_BATCH_SOURCES and augment else x
    x = random_multiply(x) if CFG.RANDOM_MULTIPLY and augment else x
    x, y = create_labels(x)
    return x, y


class MusdbDataset(tf.keras.utils.Sequence):
    def __init__(self, filenames, batch_size, augment=True):
        self.filenames = filenames
        self.batch_size = batch_size
        self.augment = augment
        
        self.clips_per_audio = self.get_clips_per_audio(filenames)
        self.total_clips = sum(self.clips_per_audio)
        self.reset_idxs()
        
        
    def __len__(self):
        return int(self.total_clips // self.batch_size) - 1
    
    def __getitem__(self, idx):
        clip_idx_begin = idx * self.batch_size
        batch_audio = []
        batch_idxs = np.random.choice(self.idxs, self.batch_size, replace=False)
        self.idxs = [i for i in self.idxs if not i in batch_idxs]
        for i in batch_idxs:
            filename, beginning_sample = self.get_clip_info(i)
            audio_tensor = decode_audio(filename, beginning_sample)
            batch_audio.append(audio_tensor)
        batch_audio = tf.stack(batch_audio, axis=0)
        batch_x, batch_y = process_audio(batch_audio, self.augment)
        return batch_x, batch_y
    
    
    def on_epoch_end(self):
        self.reset_idxs()
    
    def reset_idxs(self):
        self.idxs = np.arange(self.total_clips)
        
    
    def get_clip_info(self, i):
        curr_sum, audio_idx = 0, -1
        while (curr_sum <= i and audio_idx + 1 < len(self.clips_per_audio)):
            curr_sum += self.clips_per_audio[audio_idx+1]
            audio_idx += 1
        audio_clip_idx = i - (curr_sum - self.clips_per_audio[audio_idx])
        return self.filenames[audio_idx], audio_clip_idx*CFG.NUM_SAMPLES
        
            
    def get_clips_per_audio(self, filenames):
        clips = []
        for filename in filenames:
            filename = os.path.join(filename, "mixture.wav")
            io_tensor = tfio.audio.AudioIOTensor(filename, dtype=tf.int16)
            num_samples = int(io_tensor.shape[0])
            clips.append(num_samples // CFG.NUM_SAMPLES)
        return clips