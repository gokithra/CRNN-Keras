CHAR_VECTOR = 'அ ஆ ஆ இ ஈ ஈ உ ஊ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ'.split()

letters = [letter for letter in CHAR_VECTOR]

num_classes = len(letters) + 1

img_w, img_h = 28, 28

# Network parameters
batch_size = 128
val_batch_size = 16

downsample_factor = 4
max_text_len = 9
