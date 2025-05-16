#!/bin/bash
# Cd to the location of the file
cd "$(dirname "$0")"

source activate computer-vision-2
echo "Downloading datasets and model $1"

# Generic function should one want to do some extra extraction
function post_extract {
    :
}

case $1 in
    line_segment)
        download=(
            https://drive.google.com/drive/folders/1VWXUlk8O4XEhRBR1-o-mJPjwFfHIU1af
        )
        model=https://drive.google.com/drive/folders/1y-NThQQGo4_TH2QExjoRwnonG1RQYXGd
        ;;
    object_shadow)
        download=(
            https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Kandinsky_Indoor_OS.zip
            https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Kandinsky_Outdoor_OS.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Deepfloyd_Indoor_OS.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Deepfloyd_Outdoor_OS.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Pixart_Indoor_OS.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry-OS/resolve/main/Pixart_Outdoor_OS.zip
        )
        model=https://drive.google.com/drive/folders/1pg6pW1A7n-UGb0HXkm0a8p0HDkc79sDS
        function post_extract {
            mv $1 ./*
        }
        ;;
    perspective_fields)
        echo "Dataset it too large for $1. Exiting"
        exit 1
        model=https://drive.google.com/drive/folders/1vRlOyXRVdS9lKseOmc1SwQl1YX5QBirk
        ;;
    prequalifier)
        download=(
            https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Kandinsky_Indoor.zip
            https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Kandinsky_Outdoor.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Deepfloyd_Indoor.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Deepfloyd_Outdoor.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Pixart_Indoor.zip
            # https://huggingface.co/datasets/amitabh3/Projective-Geometry/resolve/main/Pixart_Outdoor_OS.zip
        )
        model=https://drive.google.com/drive/folders/1QGQm9SjJ2SyB0FJObOsS2tpVULyHXMCZ
        ;;
    *)
        echo "Unkown model $1. Choose either: line_segment object_shadow perspective_fields prequalifier"
        exit 1
esac

mkdir -p dataset
cd dataset

echo "Downloading datasets"
for url in $download; do
    args=""
    if [[ "$url" != *.zip ]]; then
        args+=" --folder"
    fi
    gdown $args $url
done

for zip in ./*.zip; do
    unzip -q $zip
    post_extract ${zip%.zip}
    rm $zip
done

cd ../$1/checkpoints
echo "Downloading pretrained models"

gdown $model
