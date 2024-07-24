import cv2
import numpy as np

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
# from tqdm import tqdm

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet


# @app.route('/depth', methods=['POST'])
def depth_image(image):
    # image_file = request.files['image']
    # print(image_file)

    # Read and preprocess the image
    # image_stream = image_file.read()
    # image_array = np.frombuffer(image_stream, dtype=np.uint8)
    # original_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(original_image.shape)

    image = original_image.copy() / 255.0


    h, w = image.shape[:2]
    
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        depth = depth_anything(image)
    
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    
    depth = depth.cpu().numpy().astype(np.uint8)
    return depth
    # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

    # cv2.imwrite("test.png", depth_color)

    # # Convert the color mask image to a base64-encoded string
    # depth_color_base64 = image_to_base64(depth_color)


    #     return jsonify({"result": "Depth successful", "depth": depth.tolist()})
    # except Exception as e:
    #     return jsonify({"error": str(e)})

def capture_video():
    # Open a connection to the webcam (0 is the default camera)
    cap = cv2.VideoCapture(0)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly, ret is True
        if not ret:
            print("Error: Can't receive frame (stream end?). Exiting ...")
            break

        # Display the resulting frame
        # cv2.imshow('Video Capture', frame)

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)
        cv2.imshow('RGB', flipped_frame)

        depth_frame = depth_image(flipped_frame)
        cv2.imshow('Depth', depth_frame)


        # Break the loop on 'q' key press
        if cv2.waitKey(1) == ord('q'):
            break

    # Release the capture and close any OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    
    depth_anything = DPT_DINOv2(encoder='vits', features=64, out_channels=[48, 96, 192, 384], localhub=True).cuda()

    checkpoint_path = './checkpoints/depth_anything_vits14.pth'
    depth_anything.load_state_dict(torch.load(checkpoint_path, map_location='cuda'), strict=True)
    
    # depth_anything.eval()

    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    capture_video()


