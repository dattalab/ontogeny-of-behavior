#!/usr/bin/env python
'''
Loads and runs Rockwell's U-NET on a specified moseq depth video extraction.
Takes in one parameter - the path to the extraction file.
'''
import sys
import tensorflow as tf
import h5py


def main(path):
    autoencoder_pth = '/n/groups/datta/win/longtogeny/rockwell-unet/unet_c57b6_generic_model_2.h5'
    autoencoder = tf.keras.models.load_model(autoencoder_pth)

    with h5py.File(path, 'r+') as h5f:
        frames = h5f['frames'][()]
        ae_frames = autoencoder.predict(frames[..., None], batch_size=512).squeeze()
        ae_frames[ae_frames > 120] = 0
        ae_frames[ae_frames < 5] = 0
        ae_frames = ae_frames.astype('uint8')

        if 'size_normalized_frames' in h5f:
            del h5f['size_normalized_frames']
        h5f.create_dataset('size_normalized_frames', data=ae_frames, dtype='uint8', compression='gzip')


if __name__ == "__main__":
    assert len(sys.argv) == 2, "Needs path to h5 file as parameter"
    main(sys.argv[1])