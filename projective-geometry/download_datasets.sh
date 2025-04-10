echo "Downloading paper datasets"
cd dataset

ZIPS=(
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Deepfloyd_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Deepfloyd_Outdoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/FLUX_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Kandinsky_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Kandinsky_Outdoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/PixArt_Indoor.zip
    https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/PixArt_Outdoor.zip

)
for zip in $ZIPS; do
    echo "Downloading $zip"
    wget --no-verbose "$zip"
    unzip "$zip"
    rm "$zip"
done