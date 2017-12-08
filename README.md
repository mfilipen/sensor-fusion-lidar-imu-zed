# Sensor fusion
This repository was designed for analyzing data from LIDAR, ZED camera, IMU and also has modules for sensor fusion with EKF and PF

This research is dataset based. It requires dataset but suddenly it is too large for sharing via the internet (15 Gb). If you need dataset contact to me. 

The main packages:
- racecar_description - it is for rendering the dimensional model of the car in RViz
- sensor_fusion - it is for fusing data from different sensors.
    - /sensor_fusion/scripts/ExtendedKalmanFilter/ - sensor fusion with Extended Kalman Filter
    - /sensor_fusion/scripts/ParticalFilter/ - sensor fusion with Particle Filter
    - other modules - data processing, error calculation, plotting data
