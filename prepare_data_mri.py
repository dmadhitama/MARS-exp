"""
Script to prepare MRI radar data for training and testing.
This script loads MRI data and labels, processes them, and splits them into training and testing sets.
"""

# Step 1: Import required libraries
import numpy as np
import os
import pickle


# Step 2: Define helper function to verify matching filenames
def check_if_filename_same(data_filename, labels_filename):
    """
    Verify that data and label filenames correspond to the same sample.
    Compares the first part of the filenames (before the underscore).
    """
    assert data_filename.split("_")[0] == labels_filename.split("_")[0]
    print(f"Filenames {data_filename} and {labels_filename} match.")

# Step 3: Define main function for data preparation
def prepare_mri_data_labels(data_dir, labels_dir, output_dir, test_size=0.2):
    """
    Prepare MRI data and labels for training and testing.
    
    Args:
        data_dir: Directory containing MRI data files (.npy)
        labels_dir: Directory containing label files (.cpl)
        output_dir: Directory where processed data will be saved
        test_size: Proportion of data to use for testing (default: 0.2 or 20%)
    """
    # Step 4: List and sort all data and label files
    data_list = sorted(
        [d for d in os.listdir(data_dir) if d.endswith(".npy")],
    )
    labels_list = sorted(
       [d for d in os.listdir(labels_dir) if d.endswith(".cpl")]
    )
    assert len(data_list) == len(labels_list), "Number of data and labels is not the same."

    # Step 5: Initialize empty lists for training and testing data
    data_tr = []; labels_tr = []
    data_tt = []; labels_tt = []

    # Step 6: Process each pair of data and label files
    for i in range(len(data_list)):
        # Step 6.1: Verify that filenames match
        check_if_filename_same(data_list[i], labels_list[i])
        
        # Step 6.2: Load data file
        X = np.load(os.path.join(data_dir, data_list[i]))
        
        # Step 6.3: Load and process label file
        cpl = pickle.load(open(os.path.join(labels_dir, labels_list[i]), 'rb'))
        radar_avail_frames = cpl['radar_avail_frames']
        Y = cpl['refined_gt_kps'][radar_avail_frames[0]:radar_avail_frames[1]+1]
        Y = Y.reshape(-1, Y.shape[1]*Y.shape[2])
        
        # Step 6.4: Verify that data and labels have the same number of frames
        assert X.shape[0] == Y.shape[0], "Number of frames in data and labels does not match."
        
        # Step 7: Split data into training and testing sets
        if i < (len(data_list)*(1-test_size)):
            # Step 7.1: Add to training set
            data_tr.append(X)
            labels_tr.append(Y)
        else:
            # Step 7.2: Add to testing set
            data_tt.append(X)
            labels_tt.append(Y)
        
    # Step 8: Concatenate all data and labels
    data_tr = np.concatenate(data_tr)
    labels_tr = np.concatenate(labels_tr)
    data_tt = np.concatenate(data_tt)
    labels_tt = np.concatenate(labels_tt)
    print(f"Training data and labels: {data_tr.shape}, {labels_tr.shape}")
    print(f"Testing data and labels: {data_tt.shape}, {labels_tt.shape}")

    # Step 9: Save processed data to files
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "data_tr.npy"), data_tr)
    np.save(os.path.join(output_dir, "labels_tr.npy"), labels_tr)
    np.save(os.path.join(output_dir, "data_tt.npy"), data_tt)
    np.save(os.path.join(output_dir, "labels_tt.npy"), labels_tt)


# Step 10: Main execution block
if __name__ == "__main__":
    # Step 10.1: Define input and output directories
    mri_data_dir = "/home/ubuntu/gdrive/workspace/dataset_release/features/radar/"
    mri_labels_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    out_dir = "/home/ubuntu/gdrive/workspace/dataset_release/mri_radar_rede/"

    # Step 10.2: Run the data preparation function
    prepare_mri_data_labels(mri_data_dir, mri_labels_dir, out_dir)