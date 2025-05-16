import os
import cv2
import torch
import pickle
import argparse

from deeplsd.utils.tensor import batch_to_device
from deeplsd.models.deeplsd_inference import DeepLSD
from deeplsd.geometry.viz_2d import plot_images, plot_lines


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_model():
    # Model config
    conf = {
        'detect_lines': True,  # Whether to detect lines or only DF/AF
        'line_detection_params': {
            'merge': False,  # Whether to merge close-by lines
            'filtering': True,  # Whether to filter out lines based on the DF/AF. Use 'strict' to get an even stricter filtering
            'grad_thresh': 1,
            'grad_nfa': True,  # If True, use the image gradient and the NFA score of LSD to further threshold lines. We recommand using it for easy images, but to turn it off for challenging images (e.g. night, foggy, blurry images)
        }
    }

    # Load the model
    ckpt = 'weights/deeplsd_md.tar'
    ckpt = torch.load(str(ckpt), map_location='cpu')
    net = DeepLSD(conf)
    net.load_state_dict(ckpt['model'])
    net = net.to(device).eval()
    return net


def image_to_lines(net, image_path):
    img = cv2.imread(image_path)[:, :, ::-1]
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Detect (and optionally refine) the lines
    inputs = {'image': torch.tensor(gray_img, dtype=torch.float, device=device)[None, None] / 255.}
    with torch.no_grad():
        out = net(inputs)
        pred_lines: torch.Tensor = out['lines'][0]

    return pred_lines.reshape(-1, 4)


def video_to_frames(video_path):
    dir = video_path + "_frames"
    os.makedirs(dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_path_list = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(dir, f'frame_{frame_count:04d}.jpg')
        cv2.imwrite(frame_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_path_list.append(frame_path)
        frame_count += 1

    return frame_path_list


def video_to_lines(net, storage, video_path):
    print("Processing video")
    frames = video_to_frames(video_path)

    print(f"Extracting lines from {len(frames)} frames")
    for frame_path in frames:
        print(f"Processing {frame_path}", end='\r')
        lines = image_to_lines(net, frame_path)
        storage[frame_path] = lines


if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description='Extract lines from a video using DeepLSD.')
    parser.add_argument('--print', action='store_true', help='Print the content of the pickle file.')
    parser.add_argument('--video', type=str, required=True, help='Path to the video file.')
    parser.add_argument('--storage', type=str, default='lines.pkl', help='Path to the output pickle file.')
    args = parser.parse_args()

    if args.print:
        if os.path.exists(args.storage):
            with open(args.storage, 'rb') as f:
                pickled = pickle.load(f)
                # Print the content of the pickle file
                print(f"Content of the pickle file:")
                for key, value in pickled.items():
                    print(f"{key}: \t{value[0]}    \t... and {len(value) - 1} more lines")
        exit(0)


    # Build the model
    print("Building model")
    net = build_model()
    storage = {}

    # Process the video and extract lines
    video_to_lines(net, storage, args.video)

    # Update the storage with possibly updated data
    if os.path.exists(args.storage):
        with open(args.storage, 'rb') as f:
            # Read the contents of the file
            old_storage = pickle.load(f)
            # Merge the old storage with the new storage
            storage.update(old_storage)

    # Save the lines to a pkl file
    with open(args.storage, 'wb') as f:
        # Save the updated storage
        pickle.dump(storage, f)
