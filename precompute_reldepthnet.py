import argparse
import os
import torch
from tqdm import tqdm

from relscalenet.models.relscale_cache import RelScaleNetCached
from relscalenet.dataset_reader import EvaluationDataset

DEFAULT_WEIGHTS_PATH = "weights/model_final.pth"
DEFAULT_DATA_DIR = "data"


def parse_args():
    parser = argparse.ArgumentParser(description="Run RelScaleNet on a dataset.")
    parser.add_argument("--weights", type=str, default=DEFAULT_WEIGHTS_PATH, help="Path to the model weights.")
    parser.add_argument("--keypoints", type=str, default="spsg", choices=["spsg", "spsg_old"], help="Keypoints to use.")
    parser.add_argument("--data-dir", type=str, dest="data_dir", default=DEFAULT_DATA_DIR, help="Directory containing the dataset.")
    parser.add_argument("-f", "--force", action="store_true", help="Force overwrite the cache file.")
    return parser.parse_args()


def main():
    args = parse_args()

    data_path = f"{args.data_dir}/scannet1500_{args.keypoints}.h5"
    images_path = f"{args.data_dir}/scannet1500-images/images"
    cache_path = f"{args.keypoints}_relscale_cache.h5"

    if args.force:
        print(f"Removing cache file: {cache_path}")
        os.remove(cache_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    relscalenet = RelScaleNetCached(DEFAULT_WEIGHTS_PATH, device)
    relscalenet.load_from_h5(cache_path)

    dataset = EvaluationDataset(data_path, images_path)

    for pair in tqdm(dataset, desc="Running RelScaleNet inference"):
        im1_path, im2_path = pair.image_paths()
        x1, x2 = pair.matches()

        # Predict relative scale
        relscalenet.predict_image_pair(im1_path, im2_path, x1, x2)

    relscalenet.cache.close()


if __name__ == "__main__":
    main()
