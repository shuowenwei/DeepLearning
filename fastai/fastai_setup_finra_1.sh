# Set proxies
export https_proxy=http://proxy.dev.aws.finra.org:3128
echo "export https_proxy=http://proxy.dev.aws.finra.org:3128" >> ~/.bashrc
# Update package cache, install kernel development support
sudo yum update -y
# This reboot is required to load the latest kernel version
sudo reboot