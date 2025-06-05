## Get started with Epipolar Inconsistency for deepfake videos. 

1. Download the dust3r folder.
https://github.com/naver/dust3r/tree/c9e9336a6ba7c1f1873f9295852cea6dffaf770d

2. Put the files epipolar_inconsistency.ipynb and epipolar_script.py in the dust3r-main folder.

3. Download all the videos and put it inside dust3r-main/videos folder.
https://drive.google.com/drive/folders/1eYUmLZ7XSYHcpjRxa9c8HronRv2kSPdO?dmr=1&ec=wgc-drive-hero-goto

4. Download the model parameters and put it inside dust3r-main/naver folder.
https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_linear.pth 

5. Install all the required packages for dust3r, which is described in their README.md file.

6. Change the model_name to the respective path: ~/computer-vision-2/epipolar_inconsistency/dust3r-main/naver. 

7. Run the epipolar_inconsistency.ipynb or epipolar_script.py (for side-by-side videos with their respective heatmap).
