#! /bin/bash

output_dir="/home/ubuntu/MARS-exp/mri_rgb_rede/"
user="ubuntu"

# Check permissions
ls -l $output_dir

# Change ownership if needed (replace 'your_username' with your actual username)
sudo chown -R $user:$user $output_dir
    
# Change permissions if needed
chmod -R 755 $output_dir