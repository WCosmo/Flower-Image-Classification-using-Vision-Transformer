import argparse
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from patchify import patchify
import tensorflow as tf
from train2 import load_data, tf_dataset
from vit import ViT

""" Hyperparameters """
hp = {}
hp["image_size"] = 200
hp["num_channels"] = 3
hp["patch_size"] = 25
hp["num_patches"] = (hp["image_size"]**2) // (hp["patch_size"]**2)
hp["flat_patches_shape"] = (hp["num_patches"], hp["patch_size"]*hp["patch_size"]*hp["num_channels"])

hp["batch_size"] = 16
hp["lr"] = 1e-4
hp["num_epochs"] = 500
hp["num_classes"] = 2
hp["class_names"] = ["cats", "dogs"]

hp["num_layers"] = 12
hp["hidden_dim"] = 768
hp["mlp_dim"] = 3072
hp["num_heads"] = 12
hp["dropout_rate"] = 0.1

def update_hp(epochs, heads, layers):
    global hp
    hp["num_epochs"] = epochs
    hp["num_layers"] = layers
    hp["num_heads"] = heads
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the test script given a dataset path.')
    parser.add_argument('--datapath', type=str, default='./cats_vs_dogs')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--heads', type=int, default=12)
    parser.add_argument('--layers', type=int, default=12)

    args = parser.parse_args()    

    update_hp(args.epochs, args.heads, args.layers)

    print('\nTest\n')
    print('Input dataset: ', args.datapath)
    print('Training epochs: ', hp["num_epochs"])
    print('Model layers: ', hp["num_layers"])
    print('Model heads: ', hp["num_heads"], '\n\n')

    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Paths """
    dataset_path = args.datapath
    model_path = os.path.join("files", "model.keras")

    """ Dataset """
    train_x, valid_x, test_x = load_data(dataset_path)
    print(f"Train: {len(train_x)} - Valid: {len(valid_x)} - Test: {len(test_x)}")

    test_ds = tf_dataset(test_x, batch=hp["batch_size"])

    """ Model """
    model = ViT(hp)
    model.load_weights(model_path)
    model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.Adam(hp["lr"]),
        metrics=["acc"]
    )

    result = model.evaluate(test_ds)    
