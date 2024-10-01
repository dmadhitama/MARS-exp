import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run multipurpose training script")
    parser.add_argument("--label_dirs", type=str, default="/home/ubuntu/gdrive/workspace/dataset_release/aligned_data/pose_labels/")
    parser.add_argument("--video_dirs", type=str, default="/home/ubuntu/gdrive/workspace/blurred_videos/")
    parser.add_argument("--radar_dirs", type=str, default="/home/ubuntu/gdrive/workspace/dataset_release/features/radar/")
    parser.add_argument("--cache_dir", type=str, default='/home/ubuntu/MARS-RGB-Radar-cache-npy')
    parser.add_argument("--output_direct", type=str, default='model_mri_mod-RGB-Radar-e200/')
    parser.add_argument("--input_type", type=str, default="both", choices=["video", "radar", "both"])
    parser.add_argument("--kan", action="store_true")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--accumulation_steps", type=int, default=8)
    parser.add_argument("--target_size", type=str, default="160,160")
    parser.add_argument("--radar_size", type=str, default="14,14,5")
    parser.add_argument("--chunk_size", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--iters", type=int, default=1)

    return parser.parse_args()

def run_multipurpose():
    args = parse_args()
    return (
        args.label_dirs,
        args.video_dirs,
        args.radar_dirs,
        args.cache_dir,
        args.output_direct,
        args.input_type,
        args.kan,
        args.batch_size,
        args.accumulation_steps,
        tuple(map(int, args.target_size.split(','))),
        tuple(map(int, args.radar_size.split(','))),
        args.chunk_size,
        args.num_workers,
        args.prefetch_factor,
        args.epochs,
        args.iters
    )