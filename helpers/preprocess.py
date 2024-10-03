import os
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler
import turbojpeg
from tqdm import tqdm
from functools import partial
import multiprocessing
import cv2
import pickle
from loguru import logger

def read_frame(frame_path):
    return cv2.imread(frame_path)

# collect data and label path for each subject into a dictionary
def process_frame_dir(args):
    frame_dir, video_dirs, label_dirs, radar_dirs = args
    logger.info(f"Processing {frame_dir}")
    frame_dir_path = os.path.join(video_dirs, frame_dir)
    subject_name = frame_dir.split('_')[0]
    result = {subject_name: {"frames_dirs": {}}}

    for label in os.listdir(label_dirs):
        if subject_name == label.split('_')[0]:
            result[subject_name]["labels_path"] = os.path.join(label_dirs, label)
            break

    frames_subdirs = [
        os.path.join(frame_dir_path, f) 
        for f in os.listdir(frame_dir_path) 
        if os.path.isdir(os.path.join(frame_dir_path, f))
    ]

    for frames_subdir in frames_subdirs:
        result[subject_name]["frames_dirs"][frames_subdir] = [
            os.path.join(frames_subdir, f) 
            for f in os.listdir(frames_subdir) 
            if (f.endswith(".jpg") or f.endswith(".png")) and "subject" in f
        ]

    # collect radar data
    radar_dir_path = os.path.join(radar_dirs, f"{subject_name}_featuremap.npy")
    if os.path.exists(radar_dir_path):
        result[subject_name]["radar_dirs"] = radar_dir_path
    else:
        raise ValueError(f"Radar data not found for {subject_name}. Please check the radar data directory. Exiting...")

    return result

def get_pair_path(label_dirs, video_dirs, radar_dirs):
    frame_dirs = os.listdir(video_dirs)

    with multiprocessing.Pool() as pool:
        results = list(tqdm(
            pool.imap(process_frame_dir, [(fd, video_dirs, label_dirs, radar_dirs) for fd in frame_dirs]),
            total=len(frame_dirs),
            desc="Processing frame directories"
        ))

    data_label_dict = {}
    for result in results:
        data_label_dict.update(result)

    return data_label_dict

# load labels
# for subject_name, data in data_label_dict.items():
def parse_data_label(frames_dirs, label_path, radar_path, num_workers=4):
    with open(label_path, "rb") as f:
        cpl = pickle.load(f)
    labels = cpl["refined_gt_kps"]

    radar_data = np.load(radar_path)
    
    try:
        rgb0_paths, rgb1_paths = list(frames_dirs.keys())
    except Exception as ex:
        logger.error(f"Error: {ex}")
        logger.error(list(frames_dirs.keys()))
        return None, None

    rgb0_frames = frames_dirs[rgb0_paths]
    rgb1_frames = frames_dirs[rgb1_paths]
    # adjust the length of exceeded frames to match the length of labels (for radar data)
    assert len(rgb0_frames) == len(rgb1_frames) == labels.shape[0], \
        f"Length of frames and labels do not match: {len(rgb0_frames)} != {len(rgb1_frames)} != {labels.shape[0]}"
    data_path = []
    radar = []
    label = []

    # use the shortest length between frames and radar data
    min_length = min(len(rgb0_frames), len(radar_data))

    for i in tqdm(range(min_length)):
        data_path.append((rgb0_frames[i], rgb1_frames[i]))
        label.append(labels[i])
        radar.append(radar_data[i])
    return data_path, radar, label

class MARSDatasetChunked(Dataset):
    def __init__(
            self, 
            label_dirs, 
            video_dirs, 
            radar_dirs,
            num_workers=8, 
            cache_dir=None, 
            chunk_size=1000,
            target_size=(128, 128),
            radar_size=(14, 14, 5),
        ):
        self.data_label_dict = get_pair_path(label_dirs, video_dirs, radar_dirs)
        self.num_workers = num_workers
        self.cache_dir = cache_dir
        self.chunk_size = chunk_size
        self.target_size = target_size
        self.radar_size = radar_size
        self.frames = []
        self.radar = []
        self.labels = []
        self._load_data() # load data into memory

        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
            self.preprocessed_radar_file = os.path.join(self.cache_dir, 'preprocessed_radar.npy')
            self.preprocessed_file = os.path.join(self.cache_dir, 'preprocessed_frames_gray.npy')
            if not os.path.exists(self.preprocessed_radar_file):
                self._preprocess_and_cache_radar()
            if not os.path.exists(self.preprocessed_file):
                self._preprocess_and_cache()

            # load memmap for radar
            try:
                self.radar = np.memmap(
                    self.preprocessed_radar_file, 
                    dtype='float32', 
                    mode='r', 
                    shape=(len(self.radar), self.radar_size[0], self.radar_size[1], self.radar_size[2])
                )
            except Exception as e:
                print(f"Error loading memmap: {e}")
                print("Falling back to in-memory processing")
                self._preprocess_and_cache_radar()

            # load memmap for frames
            try:
                self.frames = np.memmap(
                    self.preprocessed_file, 
                    dtype='float32', 
                    mode='r', 
                    shape=(len(self.frames), 2, self.target_size[0], self.target_size[1])
                )
            except Exception as e:
                print(f"Error loading memmap: {e}")
                print("Falling back to in-memory processing")
                self._preprocess_and_cache()

        else:
            self._preprocess_and_cache_radar()
            self._preprocess_and_cache()

        print(f"Total frames: {len(self.frames)}")
        print(f"Total labels: {len(self.labels)}")
        print(f"Total radar: {len(self.radar)}")

    def _load_data(self):
        for subject_name, data in self.data_label_dict.items():
            print(f"Processing {subject_name}")
            frames_dirs = data["frames_dirs"]
            label_path = data["labels_path"]
            radar_path = data["radar_dirs"]
            X_image, X_radar, y = parse_data_label(frames_dirs, label_path, radar_path, num_workers=self.num_workers)
            if X_image is None or X_radar is None or y is None:
                continue
            self.frames.extend(X_image)
            self.radar.extend(X_radar)
            self.labels.extend(y)

        print(f"Loaded {len(self.frames)} frames and {len(self.labels)} labels")

    def _preprocess_and_cache_radar(self):
        print("Starting radar data preprocessing and caching")
        total_frames_radar = len(self.radar)
        shape = (total_frames_radar, self.radar_size[0], self.radar_size[1], self.radar_size[2])

        if self.cache_dir:
            try:
                print(f"Creating memmap file: {self.preprocessed_radar_file}")
                radar_mmap = np.memmap(self.preprocessed_radar_file, dtype='float32', mode='w+', shape=shape)
            except Exception as e:
                print(f"Error creating memmap: {e}")
                print("Falling back to in-memory processing")
                radar_mmap = np.zeros(shape, dtype='float32')
        else:
            radar_mmap = np.zeros(shape, dtype='float32')

        # Calculate global min and max values
        global_min = float('inf')
        global_max = float('-inf')

        for chunk_start in range(0, total_frames_radar, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_frames_radar)
            chunk = self.radar[chunk_start:chunk_end]
            chunk = np.array(chunk).reshape(-1, self.radar_size[0], self.radar_size[1], self.radar_size[2])
            global_min = min(global_min, np.min(chunk))
            global_max = max(global_max, np.max(chunk))

        print(f"Global min: {global_min}, Global max: {global_max}")

        # Normalize and store the data
        for chunk_start in range(0, total_frames_radar, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, total_frames_radar)
            print(f"Processing radar chunk {chunk_start}-{chunk_end}")
            chunk = self.radar[chunk_start:chunk_end]
            chunk = np.array(chunk).reshape(-1, self.radar_size[0], self.radar_size[1], self.radar_size[2]).astype(np.float32)
            
            # Normalize to 0-1 range
            chunk_normalized = (chunk - global_min) / (global_max - global_min)
            
            radar_mmap[chunk_start:chunk_end] = chunk_normalized

        if self.cache_dir:
            try:
                radar_mmap.flush()
            except Exception as e:
                print(f"Error flushing memmap: {e}")
        
        self.radar = radar_mmap
        print("Radar data preprocessing and caching completed")

    def _preprocess_and_cache(self):
        total_frames = len(self.frames)
        shape = (total_frames, 2, self.target_size[0], self.target_size[1])
        
        if self.cache_dir:
            try:
                frames_mmap = np.memmap(self.preprocessed_file, dtype='float32', mode='w+', shape=shape)
            except Exception as e:
                print(f"Error creating memmap: {e}")
                print("Falling back to in-memory processing")
                frames_mmap = np.zeros(shape, dtype='float32')
        else:
            frames_mmap = np.zeros(shape, dtype='float32')

        with multiprocessing.Pool(processes=self.num_workers) as pool:
            for chunk_start in range(0, total_frames, self.chunk_size):
                chunk_end = min(chunk_start + self.chunk_size, total_frames)
                chunk = self.frames[chunk_start:chunk_end]
                preprocessed_chunk = list(tqdm(
                    pool.imap(partial(self._preprocess_frames, target_size=self.target_size), chunk),
                    total=len(chunk),
                    desc=f"Preprocessing frames {chunk_start}-{chunk_end}"
                ))
                frames_mmap[chunk_start:chunk_end] = preprocessed_chunk

        if self.cache_dir:
            try:
                frames_mmap.flush()
            except Exception as e:
                print(f"Error flushing memmap: {e}")
        
        self.frames = frames_mmap

    @staticmethod
    def _preprocess_frames(frame_pair, target_size):
        jpeg = turbojpeg.TurboJPEG()
        try:
            frame0 = MARSDatasetChunked._read_and_transform(frame_pair[0], jpeg, target_size)
            frame1 = MARSDatasetChunked._read_and_transform(frame_pair[1], jpeg, target_size)
            return np.concatenate((frame0, frame1), axis=0)
        except Exception as e:
            print(f"Error preprocessing frames: {frame_pair}")
            print(f"Error details: {str(e)}")
            raise

    @staticmethod
    def _read_and_transform(frame_path, jpeg, target_size):
        try:
            with open(frame_path, 'rb') as in_file:
                frame = jpeg.decode(in_file.read(), pixel_format=turbojpeg.TJPF_GRAY)
            frame = cv2.resize(frame, target_size)
            return frame.reshape(1, *target_size).astype(np.float32) / 255.0
        except OSError as e:
            print(f"Error reading file: {frame_path}")
            print(f"File exists: {os.path.exists(frame_path)}")
            print(f"File size: {os.path.getsize(frame_path) if os.path.exists(frame_path) else 'N/A'}")
            print(f"Error details: {str(e)}")
            raise

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        if idx >= len(self.frames):
            raise IndexError(f"Index {idx} is out of bounds for dataset with {len(self.frames)} items")
        
         # Make a copy of the frame data
        frame = np.array(self.frames[idx])
        frame = torch.from_numpy(frame).float()
        
        # Make a copy of the radar data
        radar = np.array(self.radar[idx])
        radar = torch.from_numpy(radar).float()
        
        label = torch.from_numpy(self.labels[idx]).float().view(51)

        return frame, radar, label

class ChunkSampler(Sampler):
    def __init__(self, data_source, chunk_size):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self.num_samples = len(data_source)

    def __iter__(self):
        indices = list(range(self.num_samples))
        np.random.shuffle(indices)
        for i in range(0, self.num_samples, self.chunk_size):
            yield from indices[i:min(i + self.chunk_size, self.num_samples)]

    def __len__(self):
        return self.num_samples