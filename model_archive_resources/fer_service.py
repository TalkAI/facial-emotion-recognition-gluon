import numpy as np
from mms.utils.mxnet import image
from skimage import transform
import mxnet as mx
import cv2 as cv
from mxnet_model_service import MXNetModelService

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
    
    if len(face_roi_list) > 0:
        (x,y,w,h) = face_roi_list[0]
        return gray_image[y:y+h,x:x+w]
    else:
        return None

def compute_norm_matrix(width, height):
    # normalization matrix used in image pre-processing
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)
    X = X.flatten()
    Y = Y.flatten()
    A = np.array([X * 0 + 1, X, Y]).T
    A_pinv = np.linalg.pinv(A)
    return A, A_pinv

def normalize(img, width, height):
    A, A_pinv = compute_norm_matrix(width, height)

    # compute image histogram
    img_flat = img.flatten()
    img_flat = img_flat.astype(int)
    img_hist = np.bincount(img_flat, minlength=256)

    # cumulative distribution function
    cdf = img_hist.cumsum()
    cdf = cdf * (2.0 / cdf[-1]) - 1.0  # normalize

    # histogram equalization
    img_eq = cdf[img_flat]

    diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))

    # after plane fitting, the mean of diff is already 0
    std = np.sqrt(np.dot(diff, diff) / diff.size)
    if std > 1e-6:
        diff = diff / std

    return diff.reshape(img.shape)

class FERService(MXNetModelService):
    """
    Defines custom pre and post processing for the Facial Emotion Recognition model
    """

    def preprocess(self, request):
        """
        Pre-process requests by attempting to extract face image, and transforming to fit the model's input

        Returns
        -------
        list of NDArray
            Processed images in the model's expected input shape
        """
        img_list = []
        input_shape = self.signature['inputs'][0]['data_shape']
        [height, width] = input_shape[2:]
        param_name = self.signature['inputs'][0]['data_name']

        # Iterate over all input images provided with the request, transform and append for inference
        for idx, data in enumerate(request):

            # Extract the input image
            img = data.get(param_name)
            if img is None:
                img = data.get("body")
            if img is None:
                img = data.get("data")
            if img is None or len(img) == 0:
                self.error = "Empty image input"
                return None

            try:
                img_arr = image.read(img).asnumpy()
            except Exception as e:
                logging.warning(e, exc_info=True)
                self.error = "Corrupted image input"
                return None
            
            # Try to identify face to crop
            face = crop_face(img_arr)
            if face is not None:
                face = transform.resize(face, (height, width))
            # If no face identified - use the entire input image
            else:
                face = cv.cvtColor(img_arr, cv.COLOR_BGR2GRAY)

            # Transform image into tensor of the required shape 
            face = np.resize(face, input_shape)
            face = normalize(face, height, width)
            face = mx.nd.array(face)
            img_list.append(face)
        
        return img_list

    def postprocess(self, data):
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
        if self.error is not None:
            return [self.error]

        # Iterating over inference results to render the normalized probabilities
        response = []
        for inference_result in data:
            softmax_result = inference_result.softmax().asnumpy()
            for idx, label in enumerate(self.labels):
                response.append({label: float(softmax_result[0][idx])})
        return [response]


_service = FERService()


def handle(data, context):
    """
    Entry point for the service, called by MMS for every incoming inference request
    """

    # Lazy initialization, so that we preserve resources until model is actually needed
    if not _service.initialized:
        _service.initialize(context)

    if data is None:
        return None

    return _service.handle(data, context)
