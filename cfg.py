class CFG:

    SAMPLE_RATE = 44100
    NUM_CHANNELS = 2
    NUM_SOURCES = 4
    ALL_SOURCES = ["vocals", "drums", "bass", "other"]
    OUTPUT_SOURCES = [0, 1, 2] # Index of the sources from 'ALL_SOURCES' that will be predicted
    
    CLIP_LENGTH = 12 # Randomly crop CLIP_LENGTH seconds from sound file
    CLIP_LENGTH_MODEL = 6 # Length of the audio that will be fed into the model.

    NUM_SAMPLES = CLIP_LENGTH * SAMPLE_RATE
    NUM_SAMPLES_MODEL = CLIP_LENGTH_MODEL * SAMPLE_RATE
    
    BLOCKS = 3
    
    RESAMPLE = False # Upsample by factor of 2 (then downscale at the end)
    NORMALIZE = False
    USE_BATCH_NORMALIZATION = True
    INITIALIZER = "rescaled_he"#"glorot_uniform"
    
    
    ## Augmentations
    SCALE_MIN = 0.25
    SCALE_MAX = 1.25
    
    AUGMENT = False
    PITCH_TEMPO_SHIFT = True
    RANDOM_MULTIPLY = True
    RANDOM_SCALE = True
    SWAP_BATCH_SOURCES = True
    SWAP_CHANNELS = True

    # FFT PARAMS FOR PITCH TEMPO SHIFT AUGMENTATION
    FRAME_LENGTH = 2048*2
    FRAME_STEP = 512*2

    BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    EPOCHS = 150
    LR = 1e-3
    LR_REDUCER_PATIENCE = 6
    EARLY_STOPPER_PATIENCE = 9
    
    TRAIN_FOLDER = "/content/data/train"
    TEST_FOLDER = "/content/data/test"