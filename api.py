import sys
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm

from depth_anything.dpt import DPT_DINOv2
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

app = Flask(__name__)


def image_to_base64(image):
    # Convert image to base64-encoded string
    _, buffer = cv2.imencode('.png', image)  # Use the image directly without conversion
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

@app.route('/depth', methods=['POST'])
def depth_image():
    try:
        image_file = request.files['image']
        # print(image_file)

        # Read and preprocess the image
        image_stream = image_file.read()
        image_array = np.frombuffer(image_stream, dtype=np.uint8)
        original_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
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
        # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)

        # cv2.imwrite("test.png", depth_color)

        # # Convert the color mask image to a base64-encoded string
        # depth_color_base64 = image_to_base64(depth_color)

        # Process segmentation results as needed

        return jsonify({"result": "Depth successful", "depth": depth.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

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


    app.run(debug=True, port=6000)
