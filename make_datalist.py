"""
Regenerate a datalist file by scanning a dataset root for a marker file.

The official starter code ships datalist/train.txt and datalist/test.txt.
This helper reproduces them from the data directories alone, so the pipeline
can be run even when only the raw dataset is available.

A datalist is one relative directory per line -- exactly the string that
bridge/data_bridge.py joins onto --data_root:

    shapenet/<synset>/<model_id>

Usage:
    # training list (marker = the mesh)
    python make_datalist.py --root ./dataset_train \
        --marker models/model_normalized.obj --out ./datalist/train.txt

    # test list (marker = the noisy cloud)
    python make_datalist.py --root ./dataset_test_noisy \
        --marker noisy.npy --out ./datalist/test.txt

    # local evaluation set built by tools/make_eval_set.py
    python make_datalist.py --root ./eval_noisy \
        --marker noisy.npy --out ./datalist/eval.txt
"""
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, required=True,
                        help='dataset root to scan')
    parser.add_argument('--marker', type=str, default='noisy.npy',
                        help='file that marks a sample directory, relative to it '
                             '(e.g. "noisy.npy" or "models/model_normalized.obj")')
    parser.add_argument('--out', type=str, required=True,
                        help='datalist file to write')
    args = parser.parse_args()

    marker_parts = args.marker.replace('\\', '/').split('/')
    marker_name = marker_parts[-1]
    marker_depth = len(marker_parts)

    entries = []
    for dirpath, _, filenames in os.walk(args.root):
        if marker_name not in filenames:
            continue
        # walk back up past the marker's own subdirectories to the sample dir
        sample_dir = os.path.join(dirpath, marker_name)
        for _ in range(marker_depth):
            sample_dir = os.path.dirname(sample_dir)
        # only accept it if the marker really sits at the expected relative path
        if not os.path.exists(os.path.join(sample_dir, *marker_parts)):
            continue
        rel = os.path.relpath(sample_dir, args.root).replace(os.sep, '/')
        entries.append(rel)

    entries = sorted(set(entries))

    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, 'w') as f:
        for rel in entries:
            f.write(rel + '\n')

    print(f'Found {len(entries)} samples under {args.root} (marker: {args.marker})')
    print(f'Wrote {args.out}')
    for rel in entries[:5]:
        print(f'  {rel}')
    if len(entries) > 5:
        print(f'  ... and {len(entries) - 5} more')


if __name__ == '__main__':
    main()
