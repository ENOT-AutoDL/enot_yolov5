import argparse
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import numpy as np


def read_image_files(
        image_list_file: Union[str, Path],
        dataset_name: Optional[str] = None,
) -> np.ndarray:

    with open(image_list_file) as file_handler:

        image_files = file_handler.read().splitlines()
        image_files = np.array(sorted([x for x in image_files if '.' in x]))

        if dataset_name is not None:
            print(f'Found {len(image_files)} {dataset_name} images.')

        return image_files


def write_split_file(split_file: Union[str, Path], image_list: List[str]) -> None:
    with open(split_file, 'w') as file_handler:
        file_handler.write('\n'.join(image_list))
        file_handler.write('\n')


def generate_enot_split(
        dataset_root_folder: Union[str, Path],
        train_path: Union[str, Path],
        val_path: Union[str, Path],
        n_search: int = 5000,
        seed: int = 0,
) -> None:

    dataset_root_folder = Path(dataset_root_folder)

    print('Reading datasets...')
    train_images = read_image_files(dataset_root_folder / train_path, 'train')
    val_images = read_image_files(dataset_root_folder / val_path, 'val')
    print('Done!')

    print('Generating splits...')

    np.random.RandomState(seed).shuffle(train_images)

    n_pretrain = len(train_images) - n_search
    pretrain_images = sorted(train_images[:n_pretrain].tolist())
    search_images = sorted(train_images[n_pretrain:].tolist())

    print(
        f'Split: ['
        f'train/tune={len(train_images)}, '
        f'pretrain={len(pretrain_images)}, '
        f'search={len(search_images)}, '
        f'val={len(val_images)}'
        f']'
    )

    print('Done!')

    print('Writing split files...')
    write_split_file(dataset_root_folder / 'pretrain.txt', pretrain_images)
    write_split_file(dataset_root_folder / 'search.txt', search_images)
    print('Done!')

    print('Completed.')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-root-folder',
        type=str,
        default='../../datasets/coco/',
        help='dataset root folder (with txt image lists)',
    )
    parser.add_argument(
        '--train-path',
        type=str,
        default='train2017.txt',
        help='train file path (relative to dataset-root-folder)',
    )
    parser.add_argument(
        '--val-path',
        type=str,
        default='val2017.txt',
        help='val dataset path (relative to dataset-root-folder)',
    )
    parser.add_argument('--n-search', type=int, default=5000, help='number of search images')
    parser.add_argument('--seed', type=int, default=0, help='numpy random split seed')
    args = parser.parse_args()
    generate_enot_split(
        args.dataset_rood_folder,
        args.train_path,
        args.val_path,
        n_search=args.n_search,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
