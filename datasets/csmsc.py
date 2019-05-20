from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
import os
from util import audio
import re


def build_from_path(in_dir, out_dir, num_workers=1, tqdm=lambda x: x):
    '''Preprocesses the CSMSC Speech dataset from a given input path into a given output directory.

      Args:
        in_dir: The directory where you have downloaded the CSMSC Speech dataset
        out_dir: The directory to write the output into
        num_workers: Optional number of worker processes to parallelize across
        tqdm: You can optionally pass tqdm to get a nice progress bar

      Returns:
        A list of tuples describing the training examples. This should be written to train.txt
    '''

    # We use ProcessPoolExecutor to parallelize across processes. This is just an optimization and you
    # can omit it and just call _process_utterance on each input if you want.
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    index = 1

    # TODO: Txt Name
    with open(os.path.join(in_dir, 'ProsodyLabeling', '000001-010000.txt'), encoding='utf-8') as f:
        phone = 0
        word_sep = []
        for line in f:
            if index > 10: #
                break
            if phone == 0:
                parts = line.strip().split()
                wav_path = os.path.join(in_dir, 'Wave', '%s.wav' % parts[0])
                word_sep = re.split(r'#[1|2|3|4|5][，|。]*', parts[1])
                phone = 1
            else:
                text = line.strip().split()
                out = ''
                k = 0
                cnt = 0
                for i in range(len(text)):
                    if i != 0:
                        out += ' '
                    out += text[i]
                    if i == k + len(word_sep[cnt]) - 1:
                        out += ' 0'
                        k += len(word_sep[cnt])
                        cnt += 1
                futures.append(executor.submit(
                    partial(_process_utterance, out_dir, index, wav_path, out)))
                index += 1
                phone = 0
    return [future.result() for future in tqdm(futures)]


def _process_utterance(out_dir, index, wav_path, text):
    '''Preprocesses a single utterance audio/text pair.

    This writes the mel and linear scale spectrograms to disk and returns a tuple to write
    to the train.txt file.

    Args:
      out_dir: The directory to write the spectrograms into
      index: The numeric index to use in the spectrogram filenames.
      wav_path: Path to the audio file containing the speech input
      text: The text spoken in the input audio file

    Returns:
      A (spectrogram_filename, mel_filename, n_frames, text) tuple to write to train.txt
    '''

    # Load the audio to a numpy array:
    wav = audio.load_wav(wav_path)

    # Compute the linear-scale spectrogram from the wav:
    spectrogram = audio.spectrogram(wav).astype(np.float32)
    n_frames = spectrogram.shape[1]

    # Compute a mel-scale spectrogram from the wav:
    mel_spectrogram = audio.melspectrogram(wav).astype(np.float32)

    # Write the spectrograms to disk:
    spectrogram_filename = 'csmsc-spec-%06d.npy' % index  # TODO: %06d?
    mel_filename = 'csmsc-mel-%06d.npy' % index
    np.save(os.path.join(out_dir, spectrogram_filename),
            spectrogram.T, allow_pickle=False)
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    # Return a tuple describing this training example:
    return (spectrogram_filename, mel_filename, n_frames, text)
