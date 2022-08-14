import tensorflow as tf
import tensorflow_io as tfio
from cfg import CFG
from phase_vocoder import pitch_tempo_shift_batch


def center_trim(tensor, reference_length):
    """
    https://github.com/facebookresearch/demucs/blob/c0eddd25dd910e06a649b8d56be91589602545c4/demucs/utils.py#L36
    
    Center trim `tensor` with respect to `reference`, along the last dimension.
    `reference` can also be a number, representing the length to trim to.
    If the size difference != 0 mod 2, the extra sample is removed on the right side.
    """
    delta = tf.cast(tensor.shape[1] - reference_length, tf.int32)
    tensor = tf.cond(delta>0, lambda:tensor[:, delta // 2:-(delta - delta // 2), :], lambda:tensor)
    return tensor

class GLU(tf.keras.layers.Layer):
    """
    Gated linear unit 
    """
    def __init__(self, hidden_units):
        super(GLU, self).__init__()
        self.hidden_units = hidden_units
        self.half_channels = tf.cast(self.hidden_units/2, tf.int32)
        
        self.linear_1 = tf.keras.layers.Dense(units=self.half_channels)
        self.linear_2 = tf.keras.layers.Dense(units=self.half_channels)
        self.sigmoid = tf.keras.layers.Activation("sigmoid")
        
    def call(self, x):
        x1, x2 = x[:,:,:self.half_channels], x[:,:,self.half_channels:]
        return self.sigmoid(self.linear_1(x1)) * self.linear_2(x2)
    
    def get_config(self):
        config = super().get_config()
        config.update({"hidden_units": self.hidden_units})
        
        
class RescaledHe(tf.keras.initializers.HeNormal):
    """
    In the paper, it is said to scaling the weights improves the result.
    In my case, without batch normalization, none of the initialiation methods works.
    Any idea ?
    """
    def __init__(self, reference_scale=0.1, **kwargs):
        super().__init__()
        self.reference_scale = reference_scale
        
    def __call__(self, shape, dtype=None):
        weight = super().__call__(shape, dtype)
        weight_std = tf.math.reduce_std(weight)
        scale = tf.math.sqrt(weight_std / self.reference_scale)
        weight = weight / scale
        return weight
    
    def get_config(self):
        config = super().get_config()
        config.update({"reference_scale":self.reference_scale})
        return config
    
    


    
class PitchTempoShifter(tf.keras.layers.Layer):
    """
    Wrapping pitch tempo shift function with frozen Layer class to benefit from GPU.
    With CPU(inside MusdbDataset class), it is too slow.
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        
    def call(self, x):
        return pitch_tempo_shift_batch(x)
    

class Mixer(tf.keras.layers.Layer):
    """
    Mixes the sources to be fed to the model and extracts label sources
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        
    def call(self, x):
        x_mix = tf.reduce_sum(x, axis=-1)
        x_mix = tf.clip_by_value(x_mix, -1, 1)
        return x_mix, tf.squeeze(tf.gather(x, CFG.OUTPUT_SOURCES, axis=-1))
    
    
class Resampler(tf.keras.layers.Layer):
    """
    Up or down samples the audio
    """
    def __init__(self):
        super().__init__()
        self.trainable = False
        
    def call(self, x, up_down="up"):
        if tf.rank(x)==3:
            if up_down == "up":
                x = tfio.audio.resample(x, 1, 2)
            else:
                x = tfio.audio.resample(x, 2, 1)
        else:
            x_resampled = []
            for i in range(len(CFG.OUTPUT_SOURCES)):
                xi = x[:,:,:,i] # shape: (bs, samples, channels)
                if up_down == "up":
                    xi = tfio.audio.resample(xi, 1, 2)
                else:
                    xi = tfio.audio.resample(xi, 2, 1)
                x_resampled.append(xi)
            x = tf.stack(x_resampled, axis=-1)
        return x
        
    
    
class LossLayer(tf.keras.layers.Layer):
    """
    Loss is computed inside the model(since sources are mixed inside the model),
    because pitch tempo shift aug. is applied inside the model
    """
    def __init__(self):
        super().__init__()
        
    def call(self, y_true, y_pred):
        return tf.math.reduce_mean(tf.math.abs(y_true - y_pred))


# Here, GLU implementation is different from pytroch implementation.
# So, i didn't expanded channels 2x(as they did in the paper) before GLU

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, channel_in, channel_out, use_bn=True, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        self.kernel_initializer_conv = RescaledHe() if kernel_initializer=="rescaled_he" else kernel_initializer
        self.kernel_initializer_conv_expand = RescaledHe() if kernel_initializer=="rescaled_he" else kernel_initializer
        
        self.conv = tf.keras.layers.Conv1D(filters=channel_out, kernel_size=8, strides=4, padding="valid", kernel_initializer=self.kernel_initializer_conv)
        self.conv_expand = tf.keras.layers.Conv1D(filters=channel_out*2, kernel_size=1, strides=1, padding="valid", kernel_initializer=self.kernel_initializer_conv_expand)
        self.relu = tf.keras.layers.Activation("relu")
        self.glu = GLU(channel_out*2)
        
        if self.use_bn:
            self.normalization1 = tf.keras.layers.BatchNormalization()
            self.normalization2 = tf.keras.layers.BatchNormalization()
        
        
    def call(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.normalization1(x)
        x = self.relu(x)
        x = self.conv_expand(x)
        if self.use_bn:
            x = self.normalization2(x)
        x = self.glu(x)
        return x
    
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, channel_in, channel_out, use_bn=True, kernel_initializer="glorot_uniform", is_last_layer=False, **kwargs):
        super().__init__(**kwargs)
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        self.is_last_layer = is_last_layer
        self.kernel_initializer_conv_transpose = RescaledHe() if kernel_initializer=="rescaled_he" else kernel_initializer
        self.kernel_initializer_conv_expand = RescaledHe() if kernel_initializer=="rescaled_he" else kernel_initializer
        
        self.conv_transpose = tf.keras.layers.Conv1DTranspose(filters=channel_out, kernel_size=8, strides=4, padding="valid", kernel_initializer=self.kernel_initializer_conv_transpose)
        self.conv_expand = tf.keras.layers.Conv1D(filters=channel_in, kernel_size=1, strides=1, padding="valid", kernel_initializer=self.kernel_initializer_conv_expand)
        self.glu = GLU(channel_in)
        self.relu = tf.keras.layers.Activation("relu")
        
        if self.use_bn:
            self.normalization1 = tf.keras.layers.BatchNormalization()
            self.normalization2 = tf.keras.layers.BatchNormalization()
        
    def call(self, x, encoder_output):
        encoder_output = center_trim(encoder_output, x.shape[1])
        x = x + encoder_output
        x = self.conv_expand(x)
        if self.use_bn:
            x = self.normalization1(x)
        x = self.glu(x)
        x = self.conv_transpose(x)
        if not self.is_last_layer:
            if self.use_bn:
                x = self.normalization2(x)
            x = self.relu(x)
        return x
    
    
class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, repeat=6, growth=2, input_channels=2, channels_first=64, use_bn=True, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.growth = growth
        self.input_channels = input_channels
        self.channels_first = channels_first
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        
        self.encoder_layers = [EncoderLayer(self.input_channels, self.channels_first, use_bn=self.use_bn, kernel_initializer=kernel_initializer) if layer_num==0 else EncoderLayer(self.channels_first*self.growth**(layer_num-1), self.channels_first*self.growth**layer_num, use_bn=self.use_bn, kernel_initializer=kernel_initializer) for layer_num in range(repeat)]
        
    def call(self, x):
        encoder_outputs = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encoder_outputs.insert(0,x)
        return x, encoder_outputs
    
    
    def get_output_channels(self):
        return self.channels_first * self.growth ** (self.repeat-1)
    
    
class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, repeat=6, growth=0.5, input_channels=2048, output_channels=8, use_bn=True, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.repeat = repeat
        self.growth = growth
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.use_bn = use_bn
        self.kernel_initializer = kernel_initializer
        
        self.decoder_layers = []
        for layer_num in range(repeat):
            if layer_num == repeat - 1:
                layer = DecoderLayer(int(self.input_channels*self.growth**(layer_num)),
                                     self.output_channels,
                                     use_bn=self.use_bn, kernel_initializer=kernel_initializer,
                                     is_last_layer=True)
            else:
                layer = DecoderLayer(int(self.input_channels*self.growth**(layer_num)),
                                     int(self.input_channels*self.growth**(layer_num+1)),
                                     use_bn=self.use_bn, kernel_initializer=kernel_initializer)
            self.decoder_layers.append(layer)
            
    def call(self, x, encoder_outputs):
        for decoder_layer, encoder_output in zip(self.decoder_layers, encoder_outputs):
            x = decoder_layer(x, encoder_output)
        return x
    
    
    
class BiLSTM(tf.keras.layers.Layer):
    def __init__(self, output_units, **kwargs):
        super().__init__()
        self.output_units = output_units
        self.bi_lstm_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(output_units, return_sequences=True))
        self.linear_1 = tf.keras.layers.Dense(output_units)
        self.bi_lstm_2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(output_units, return_sequences=True))
        self.linear_2 = tf.keras.layers.Dense(output_units)
        
    def call(self, x):
        x = self.linear_1(self.bi_lstm_1(x))
        x = self.linear_2(self.bi_lstm_2(x))
        return x
    
    
class Demucs(tf.keras.Model):
    def __init__(self, num_samples, output_sources=4, audio_channels=2,
                 channel_first=64, repeat=6, growth=2, resample=False,
                 use_bn=True, kernel_initializer="glorot_uniform", **kwargs):
        super().__init__(**kwargs)
        self.num_samples = num_samples
        self.output_sources = output_sources
        self.audio_channels = audio_channels
        self.output_channels = output_sources * audio_channels
        self.channel_first = channel_first
        self.repeat = repeat
        self.growth = growth
        self.use_bn = use_bn
        self.resample = resample
        self.kernel_initializer_final = RescaledHe() if kernel_initializer=="rescaled_he" else kernel_initializer
        
        self.new_num_samples = self.num_samples if not self.resample else self.num_samples * 2
        self.padding = self.valid_length(self.new_num_samples) - self.new_num_samples
        
        self.encoder_block = EncoderBlock(self.repeat, self.growth, self.audio_channels, self.channel_first,
                                          use_bn=self.use_bn, kernel_initializer=kernel_initializer)
        self.encoder_output_channels = self.encoder_block.get_output_channels()
        self.bi_lstm = BiLSTM(self.encoder_output_channels)
        self.decoder_block = DecoderBlock(self.repeat, 1/self.growth, self.encoder_output_channels, self.output_channels,
                                          use_bn=self.use_bn, kernel_initializer=kernel_initializer)
        self.pt_shifter = PitchTempoShifter()
        self.resampler = Resampler()
        self.mixer = Mixer()
        self.loss_layer = LossLayer()
        
        
    def call(self, x, training=False, predict=False):
        if not predict:
            x = tf.reshape(x, (-1, CFG.NUM_SAMPLES, CFG.NUM_CHANNELS, CFG.NUM_SOURCES))
            
        if training and CFG.AUGMENT and CFG.PITCH_TEMPO_SHIFT:
            x = self.pt_shifter(x)
        
        x = x[:,:self.num_samples]
        
        if CFG.RESAMPLE:
            x = self.resampler(x, "up")
        
        # Get input(mix) and label(sources) from unmixed audio tensor 
        if not predict:
            x, y = self.mixer(x)
        
        if CFG.NORMALIZE:
            mono  = tf.math.reduce_mean(x, axis=-1, keepdims=True) # mean in channel dim
            mean = tf.math.reduce_mean(mono, axis=1, keepdims=True) # mean in the sample dim
            std = tf.math.reduce_mean(mono, axis=1, keepdims=True) # std in sample dim
            x = (x - mean) / (std + 1e-5)
        
        x = tf.pad(x, [[0,0], [self.padding//2,self.padding-self.padding//2],[0,0]])
        x = tf.reshape(x, [-1, self.valid_length(self.new_num_samples), CFG.NUM_CHANNELS])
        x, encoder_outputs = self.encoder_block(x)
        x = self.bi_lstm(x)
        x = self.decoder_block(x, encoder_outputs)
        x = center_trim(x, self.new_num_samples)
        
        x = tf.reshape(x, (-1, self.new_num_samples, self.audio_channels, self.output_sources))
        x = tf.squeeze(x)
        if CFG.NORMALIZE:
            if tf.rank(x) > tf.rank(mean):
                mean = tf.expand_dims(mean, axis=1)
                std = tf.expand_dims(std, axis=1)
            x = x * std + mean
            
        if not predict:
            x = self.loss_layer(y, x)
        if predict and CFG.RESAMPLE:
            x = self.resampler(x, "down")

        return x
    
    
    def valid_length(self, length):
        """
        https://github.com/facebookresearch/demucs/blob/7b832a30ebd8e226b355a40d0d5b4939943801ba/demucs/demucs.py#L388
        
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolution, e.g. for all
        layers, size of the input - kernel_size % stride = 0.
        Note that input are automatically padded if necessary to ensure that the output
        has the same length as the input.
        """
        for _ in range(self.repeat):
            length = tf.math.ceil((length - 8) / 4) + 1
            length = tf.math.maximum(1., length)

        for idx in range(self.repeat):
            length = (length - 1) * 4 + 8

        return int(length)