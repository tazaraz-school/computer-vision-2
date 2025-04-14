echo "Downloading paper datasets"
cd dataset

ZIPS=(
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Deepfloyd_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Deepfloyd_Outdoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Kandinsky_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Kandinsky_Outdoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Pixart_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_Pixart_Outdoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_SDXL_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Recent_SDXL_Outdoor.zip

)
for zip in "${ZIPS[@]}"; do
    echo "Downloading $zip"
    wget --no-verbose "$zip"
done

for zip in *.zip; do
    echo "Processing $zip"
    unzip "$zip"
    rm "$zip"
done