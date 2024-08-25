import numpy as np
import os
import pickle


def check_if_filename_same(data_filename, labels_filename):
    assert data_filename.split("_")[0] == labels_filename.split("_")[0]
    print(f"Filenames {data_filename} and {labels_filename} match.")

def prepare_mri_data_labels(data_dir, labels_dir, output_dir, test_size=0.2):
    data_list = sorted(
        [d for d in os.listdir(data_dir) if d.endswith(".npy")],
    )
    labels_list = sorted(
       [d for d in os.listdir(labels_dir) if d.endswith(".cpl")]
    )
    assert len(data_list) == len(labels_list), "Number of data and labels is not the same."

    data_tr = []; labels_tr = []
    data_tt = []; labels_tt = []

    for i in range(len(data_list)):
        check_if_filename_same(data_list[i], labels_list[i])
        X = np.load(os.path.join(data_dir, data_list[i]))
        cpl = pickle.load(open(os.path.join(labels_dir, labels_list[i]), 'rb'))
        radar_avail_frames = cpl['radar_avail_frames']
        Y = cpl['refined_gt_kps'][radar_avail_frames[0]:radar_avail_frames[1]+1]
        Y = Y.reshape(-1, Y.shape[1]*Y.shape[2])
        assert X.shape[0] == Y.shape[0], "Number of frames in data and labels does not match."
        
        # save training data
        if i < (len(data_list)*(1-test_size)):
            data_tr.append(X)
            labels_tr.append(Y)
        else:
            data_tt.append(X)
            labels_tt.append(Y)
        
    # concatenate all data and labels
    data_tr = np.concatenate(data_tr)
    labels_tr = np.concatenate(labels_tr)
    data_tt = np.concatenate(data_tt)
    labels_tt = np.concatenate(labels_tt)
    print(f"Training data and labels: {data_tr.shape}, {labels_tr.shape}")
    print(f"Testing data and labels: {data_tt.shape}, {labels_tt.shape}")

    # save to file
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(os.path.join(output_dir, "data_tr.npy"), data_tr)
    np.save(os.path.join(output_dir, "labels_tr.npy"), labels_tr)
    np.save(os.path.join(output_dir, "data_tt.npy"), data_tt)
    np.save(os.path.join(output_dir, "labels_tt.npy"), labels_tt)


if __name__ == "__main__":
    mri_data_dir = "/home/ubuntu/gdrive/workspace/dataset_release/features/radar/"
    mri_labels_dir = "/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/"
    out_dir = "/home/ubuntu/gdrive/workspace/dataset_release/mri_radar_rede/"

    prepare_mri_data_labels(mri_data_dir, mri_labels_dir, out_dir)