"""
Credits: This code is borrowed from https://github.com/Microsoft/FERPlus/tree/master/src
"""

import os
import numpy as np
import argparse

from ferplus_data_generator import FERPlusParameters, FERPlusReader, display_summary

emotion_table = {'neutral': 0,
                 'happiness': 1,
                 'surprise': 2,
                 'sadness': 3,
                 'anger': 4,
                 'disgust': 5,
                 'fear': 6,
                 'contempt': 7}

# List of folders for training, validation and test.
train_folders = ['FER2013Train']
valid_folders = ['FER2013Valid']
test_folders = ['FER2013Test']


def main(base_folder):
    training_mode = 'majority'
    model_name='VGG13'
    # create needed folders.
    output_model_path = os.path.join(base_folder, R'models')
    output_model_folder = os.path.join(output_model_path, model_name + '_' + training_mode)
    if not os.path.exists(output_model_folder):
        os.makedirs(output_model_folder)

    # create the model
    num_classes = len(emotion_table)

    input_height = 64
    input_width = 64

    # read FER+ dataset.
    print("Loading data...")
    train_params = FERPlusParameters(num_classes, input_height, input_width, training_mode, False)
    test_and_val_params = FERPlusParameters(num_classes, input_height, input_width, "majority", True)

    train_data_reader = FERPlusReader.create(base_folder, train_folders, "label.csv", train_params)
    val_data_reader = FERPlusReader.create(base_folder, valid_folders, "label.csv", test_and_val_params)
    test_data_reader = FERPlusReader.create(base_folder, test_folders, "label.csv", test_and_val_params)

    # print summary of the data.
    display_summary(train_data_reader, val_data_reader, test_data_reader)

    print("Processing train data...")
    train_processed_images, train_processed_labels = train_data_reader.process_images()
    print(train_processed_images.shape)
    print(train_processed_labels.shape)

    print("Processing test data...")
    test_processed_images, test_processed_labels = test_data_reader.process_images()
    print(test_processed_images.shape)
    print(test_processed_labels.shape)

    print("Processing test data...")
    val_processed_images, val_processed_labels = val_data_reader.process_images()
    print(val_processed_images.shape)
    print(val_processed_labels.shape)

    print("Saving processed images to...")
    print(base_folder + '/fer_train_processed_images.npy')
    print(base_folder + '/fer_train_processed_labels.npy')
    print(base_folder + '/fer_test_processed_images.npy')
    print(base_folder + '/fer_test_processed_labels.npy')
    print(base_folder + '/fer_val_processed_images.npy')
    print(base_folder + '/fer_val_processed_labels.npy')

    np.save(base_folder + '/fer_train_processed_images.npy', train_processed_images)
    np.save(base_folder + '/fer_train_processed_labels.npy', train_processed_labels)

    np.save(base_folder + '/fer_test_processed_images.npy', test_processed_images)
    np.save(base_folder + '/fer_test_processed_labels.npy', test_processed_labels)

    np.save(base_folder + '/fer_val_processed_images.npy', val_processed_images)
    np.save(base_folder + '/fer_val_processed_labels.npy', val_processed_labels)

    print('All processing done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d",
                        "--base_folder",
                        type=str,
                        help="Base folder containing the training, validation and testing folder.",
                        required=True)

    args = parser.parse_args()
    main(args.base_folder)
