# Kinect-Vicon Comparison  
**Evaluating Kinect Sensor Accuracy for Kinematic Analysis Using the KERAAL Dataset (Course project for Medical Robotics)**

## 🧩 Overview  
This project investigates the **accuracy of the Microsoft Kinect sensor** in measuring human motion compared to the **Vicon motion capture system (gold standard)**.  
The analysis focuses on **rehabilitation movements for low back pain**, using data from the **open-source KERAAL dataset**.


## 🎯 Objectives  
- Compare **Kinect** and **Vicon** motion capture data.  
- Compute kinematic measures in **position** and **velocity** domains.  
- Align Kinect and Vicon coordinate systems using the **Least-Squares Fitting of Two 3D Point Sets** algorithm.  
- Evaluate Kinect’s potential for **rehabilitation and clinical assessment**.

## ⚙️ Methods  
1. Data preprocessing and synchronization from the **KERAAL dataset**.  
2. Coordinate alignment between Kinect and Vicon data.  
3. Kinematic feature computation and error analysis.  
4. Evaluation of Kinect’s accuracy across different movement planes.
The image bellow shows the vector products used for definition of trunk rotation.
<img width="601" height="491" alt="image" src="https://github.com/user-attachments/assets/b696bd18-e20e-4c25-b7bf-6d9c5ef5d449" />
**Figure 1.** Defined vectors for calculation of trunk rotation.  

## 📊 Results (Summary)  
- Kinect provided **reliable position and velocity estimates** for trunk rotation movements.  
- Accuracy varied depending on **movement type** and **plane of motion**.


  <img width="908" height="617" alt="image" src="https://github.com/user-attachments/assets/e9ba2580-53b7-49a9-806a-d9d391db29f3" />
**Figure 2.** Kinect misidentification of the head and left shoulder positions in frame 110 of the *ELK* movement and results obtained for rotation, velocity and angular acceleration in RTK motion.  






