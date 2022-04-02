import pyaudio

FORMAT = pyaudio.paInt16
CHANNELS = 1

# samples per second
RATE = 44100

# length in seconds
PRE_SAMPLE_LENGTH = 0.05
# size of sample = sample rates * sample_length
CHUNK1 = int(RATE * PRE_SAMPLE_LENGTH)

# length in seconds
SAMPLE_LENGTH = 0.25
# size of sample = sample rates * sample_length
CHUNK2 = int(RATE * (SAMPLE_LENGTH-PRE_SAMPLE_LENGTH))

# full sample...
CHUNK = CHUNK1 + CHUNK2

LOW_FREQUENCY = 0
HIGH_FREQUENCY = 10000

MIN_PEAK_DIST = 250

# tolerance for sum-of square comparison
MATCH_LIMIT = 1

SOUND_OFF_LEVEL = 160
SOUND_ON_LEVEL = 400

# included wave plot
PLOT_COUNT = 4

SOUND_DATA_FILE = 'peaks.csv'

