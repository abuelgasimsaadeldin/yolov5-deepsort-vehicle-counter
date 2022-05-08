## Vehicle Localization and Counting

This system is able to detect and classify 4 different classes of vehicles including; *car*, *motorcycle*, *bus* and *truck* and count the number of vehicles in each classes as they pass through a virtual polygon area.

![image](https://user-images.githubusercontent.com/68045710/167283114-53de6332-b88f-4446-b2ec-9c43f3182deb.png)

## Steps to Setup on Nvidia Jetson Nano
### 1) SWAP Jetson Nano to Free Space
```bash
$ sudo systemctl disable nvzramconfig  
$ sudo fallocate -l 4G /mnt/4GB.swap  
$ sudo chmod 600 /mnt/4GB.swap  
$ sudo mkswap /mnt/4GB.swap  
$ sudo su  
$ echo "/mnt/4GB.swap swap swap defaults 0 0" >> /etc/fstab  
$ exit  

REBOOT!   
```

### 2) Install torch and torchvision libraries
```bash
$ ./install_torch.sh
```

### 3) Install OpenCV
```bash
$ wget https://github.com/Qengineering/Install-OpenCV-Jetson-Nano/raw/main/OpenCV-4-5-5.sh
$ sudo chmod 755 ./OpenCV-4-5-5.sh
$ ./OpenCV-4-5-5.sh
```

### 4) Install requirements (only this step required for running on PC)
```bash
$ pip install -r requirements.txt
```

## Run Inference

*count.py*: Detects and counts the number of vehicles detected in each frame and displays it on the image window. (suitable for images)   

```bash
$ python count.py --source data/vehicle_test_images/* 
```

*track.py*: Detects and tracks each vehicle present in the video frame providing it with a unique id number for tracking. (suitable for videos)
 
 ```bash
$ python track.py --source data/vehicle_test_videos/live.mp4
```

## Live Setup

![image](https://user-images.githubusercontent.com/68045710/167283855-07b7c517-8e57-4c3f-adb1-8476ae114a20.png)

