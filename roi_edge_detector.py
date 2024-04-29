import cv2 as cv
import argparse
import numpy as np

parser = argparse.ArgumentParser(
        description='This sample shows how to define custom OpenCV deep learning layers in Python. '
                    'Holistically-Nested Edge Detection (https://arxiv.org/abs/1504.06375) neural network '
                    'is used as an example model. Find a pre-trained model at https://github.com/s9xie/hed.')
parser.add_argument('--input', help='Path to image or video. Skip to capture frames from camera')
parser.add_argument('--prototxt', help='Path to deploy.prototxt', required=True)
parser.add_argument('--caffemodel', help='Path to hed_pretrained_bsds.caffemodel', required=True)
parser.add_argument('--savefile', help='Specifies the output video path', default='output.jpg', type=str)
args = parser.parse_args()

class CropLayer(object):
    def __init__(self, params, blobs):
        self.xstart = 0
        self.xend = 0
        self.ystart = 0
        self.yend = 0

    # Our layer receives two inputs. We need to crop the first input blob
    # to match a shape of the second one (keeping batch size and number of channels)
    def getMemoryShapes(self, inputs):
        inputShape, targetShape = inputs[0], inputs[1]
        batchSize, numChannels = inputShape[0], inputShape[1]
        height, width = targetShape[2], targetShape[3]

        self.ystart = int((inputShape[2] - targetShape[2]) / 2)
        self.xstart = int((inputShape[3] - targetShape[3]) / 2)
        self.yend = self.ystart + height
        self.xend = self.xstart + width

        return [[batchSize, numChannels, height, width]]

    def forward(self, inputs):
        return [inputs[0][:,:,self.ystart:self.yend,self.xstart:self.xend]]

# Load the model.
net = cv.dnn.readNetFromCaffe(args.prototxt, args.caffemodel)
cv.dnn_registerLayer('Crop', CropLayer)

image = cv.imread(args.input)

# Bounding box coordinates
x1 = 1390
y1 = 1430
x2 = 1920
y2 = 2450

# Crop the region inside the bounding box
cropped_image = image[y1:y2, x1:x2]

inp = cv.dnn.blobFromImage(cropped_image, scalefactor=1.0, size=(cropped_image.shape[1], cropped_image.shape[0]),
                           mean=(104.00698793, 116.66876762, 122.67891434),
                           swapRB=False, crop=False)
net.setInput(inp)
out = net.forward()

out = out[0, 0]
out = cv.resize(out, (cropped_image.shape[1], cropped_image.shape[0]))

out = cv.cvtColor(out, cv.COLOR_GRAY2BGR)
out = 255 * out
out = out.astype(np.uint8)

# Overlay the edges on the original image
image[y1:y2, x1:x2] = out

cv.imwrite(args.savefile, image)
