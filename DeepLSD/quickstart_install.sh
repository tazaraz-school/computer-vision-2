pip install cmake==3.18  # Specific version of cmake
cd third_party/pytlsd/pybind11 && pip install .  # Install pybind11

pip install scikit-build
pip install -r requirements.txt  # Install the requirements
pip install -e .  # Install DeepLSD

mkdir weights
wget https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_wireframe.tar -O weights/deeplsd_wireframe.tar
wget https://cvg-data.inf.ethz.ch/DeepLSD/deeplsd_md.tar -O weights/deeplsd_md.tar