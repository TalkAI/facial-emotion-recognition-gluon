import numpy as np
from mms.utils.mxnet import image
from mms.model_service.mxnet_model_service import MXNetBaseService
from skimage import transform
import mxnet as mx
import cv2 as cv

# One time initialization of Haar Cascade Classifier to extract and crop out face
face_detector = cv.CascadeClassifier('haarcascade_frontalface.xml')
# Classifier parameter specifying how much the image size is reduced at each image scale
scale_factor = 1.3
# Classifier parameter how many neighbors each candidate rectangle should have to retain it
min_neighbors = 5

def crop_face(image):
    """Attempts to identify a face in the input image.

    Parameters
    ----------
    image : array representing a BGR image

    Returns
    -------
    array
        The cropped face, transformed to grayscale. If no face found returns None

    """
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    face_roi_list = face_detector.detectMultiScale(gray_image, scale_factor, min_neighbors)
    
    if (len(face_roi_list) > 0):
        (x,y,w,h) = face_roi_list[0]
        return gray_image[y:y+h,x:x+w]
    else:
        return None

class FERService(MXNetBaseService):
    """
    Defines custom pre and post processing for the Facial Emotion Recognition model
    """

    def _preprocess(self, data):
        """
        Pre-process requests by attempting to extract face image, and transforing to fit the model's input

        Parameters
        ----------
        data : list of input images
            Raw inputs from request.

        Returns
        -------
        list of NDArray
            Processed images in the model's expected input shape
        """
        img_list = []

        # Iterate over all input images provided with the request, transform and append for inference
        for idx, img in enumerate(data):          
            input_shape = self.signature['inputs'][idx]['data_shape']
            [h, w] = input_shape[2:]
            img_arr = image.read(img).asnumpy()
            
            # Try to identify face to crop
            face = crop_face(img_arr)
            if (face is not None):
                face = transform.resize(face, (h,w))            
            # If no face identified - use the entire input image
            else:
                face = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            # Transform image into tensor of the required shape 
            face = np.resize(face, input_shape)
            face = mx.nd.array(face)
            img_list.append(face)
        
        return img_list

    def _postprocess(self, data):
        """
        Post-process inference result to normalize probabilities and render with labels

        Parameters
        ----------
        data : list of NDArray
            Inference output.

        Returns
        -------
        list of object
            list of outputs to be sent back.
        """
        response = []
        
        # Iterating over inference results to render the normalized probabilities
        for inference_result in data:
            softmax_result = inference_result.softmax().asnumpy()
            for idx, label in enumerate(self.labels):
                response.append({label: float(softmax_result[0][idx])})
        return response
