import argparse
import os

from meeg_tools.preprocessing import prepare_epochs_for_ica, run_ica, \
    run_autoreject, run_ransac
from meeg_tools.utils.epochs import create_epochs
from meeg_tools.utils.raw import read_raw_measurement, filter_raw
from meeg_tools.utils.log import update_log
from meeg_tools.utils.config import settings

settings['bandpass_filter']['low_freq'] = 1
settings['bandpass_filter']['high_freq'] = 30
settings['ica']['n_components'] = 20


def run_pipeline(source: str):
    target_path = os.path.join(source, 'preprocessed')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    files = [file for file in os.listdir(source) if
             file.endswith(('.edf', '.vhdr'))]

    log_file_path = os.path.join(target_path, 'log.csv')

    for file in files:
        raw_file_path = os.path.join(source, file)
        # Set montage to raw
        montage_file_path = 'Starstim32.locs'
        raw = read_raw_measurement(raw_file_path=raw_file_path,
                                   locs_file_path=montage_file_path)
        print(raw.info)

        # band-pass filering
        raw_bandpass = filter_raw(raw)

        # create epochs from filtered continuous data
        epochs = create_epochs(raw=raw_bandpass)

        # Change the order of channels
        ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz',
                    'F4', 'F8', 'T7', 'C3', 'Cz',
                    'C4', 'T8', 'P7', 'P3', 'Pz',
                    'P4', 'P8', 'O1', 'Oz', 'O2']
        epochs = epochs.copy().load_data().pick_channels(ch_names,
                                                         ordered=True)

        # initial rejection of bad epochs
        epochs_faster = prepare_epochs_for_ica(epochs=epochs)

        ica = run_ica(epochs=epochs_faster)
        ica.apply(epochs_faster)
        epochs_faster.info['description'] = f'n_components: {len(ica.exclude)}'

        reject_log = run_autoreject(epochs_faster, n_jobs=11, subset=False)
        epochs_autoreject = epochs_faster.copy().drop(reject_log.report,
                                                      reason='AUTOREJECT')

        ransac = run_ransac(epochs_autoreject)

        epochs_ransac = epochs_autoreject.copy()
        epochs_ransac.info['bads'] = ransac.bad_chs_
        bads_str = ', '.join(ransac.bad_chs_)
        epochs_ransac.info.update(description=epochs_autoreject.info[
                                                  'description'] + ', interpolated: ' + bads_str)

        epochs_ransac.set_eeg_reference('average')

        # save clean epochs
        fid = epochs.info['fid']
        epochs_autoreject.info.update(fid=f'{fid}_ICA_autoreject_ransac')
        postfix = '-epo.fif.gz'
        epochs_ransac.save(
            os.path.join(target_path,
                         f'{epochs_autoreject.info["fid"]}{postfix}'),
            overwrite=True)

        # Create a preprocessing log file
        update_log(log_file_path, epochs_ransac, "auto run")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str,
                        help="The directory where raw EEG files are.")
    args = parser.parse_args()

    if os.path.exists(args.source):
        run_pipeline(source=args.source)
