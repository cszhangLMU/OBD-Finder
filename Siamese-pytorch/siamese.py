import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image

from nets.siamese import Siamese as siamese
from utils.utils import letterbox_image, preprocess_input, cvtColor, show_config


# ---------------------------------------------------#
#   If you're using your own trained model, modify the model_path parameter.
# ---------------------------------------------------#
class Siamese(object):
    _defaults = {
        # -----------------------------------------------------#
        #   If using your own trained model, make sure to change model_path
        #   model_path points to the weights file in the logs folder
        # -----------------------------------------------------#
        "model_path": '/logs3/best_epoch_weights.pth',
        # -----------------------------------------------------#
        #   Input image size.
        # -----------------------------------------------------#
        "input_shape": [105, 105],
        # --------------------------------------------------------------------#
        #   This variable controls whether to use letterbox_image for resizing input images without distortion
        #   Otherwise, a CenterCrop is applied to the image.
        # --------------------------------------------------------------------#
        "letterbox_image": False,
        # -------------------------------#
        #   Whether to use CUDA
        #   If there is no GPU, set this to False
        # -------------------------------#
        "cuda": True
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   Initialize Siamese
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        self.generate()

        show_config(**self._defaults)

    # ---------------------------------------------------#
    #   Load the model
    # ---------------------------------------------------#
    def generate(self):
        # ---------------------------#
        #   Load the model and weights
        # ---------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = siamese(self.input_shape)
        model.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = model.eval()
        print('{} model loaded.'.format(self.model_path))

        if self.cuda:
            self.net = torch.nn.DataParallel(self.net)
            cudnn.benchmark = True
            self.net = self.net.cuda()

    def letterbox_image(self, image, size):
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)

        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128, 128, 128))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        if self.input_shape[-1] == 1:
            new_image = new_image.convert("L")
        return new_image

    # ---------------------------------------------------#
    #   Detect image
    # ---------------------------------------------------#
    def detect_image(self, image_1, image_2):
        # ---------------------------------------------------------#
        #   Convert images to RGB to prevent errors when using grayscale images during prediction.
        # ---------------------------------------------------------#
        image_1 = cvtColor(image_1)
        image_2 = cvtColor(image_2)

        # ---------------------------------------------------#
        #   Resize input images without distortion
        # ---------------------------------------------------#
        image_1 = letterbox_image(image_1, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)
        image_2 = letterbox_image(image_2, [self.input_shape[1], self.input_shape[0]], self.letterbox_image)

        # ---------------------------------------------------------#
        #   Normalize and add batch_size dimension
        # ---------------------------------------------------------#
        photo_1 = preprocess_input(np.array(image_1, np.float32))
        photo_2 = preprocess_input(np.array(image_2, np.float32))

        with torch.no_grad():
            # ---------------------------------------------------#
            #   Add batch dimension to inputs before passing through the network
            # ---------------------------------------------------#
            photo_1 = torch.from_numpy(np.expand_dims(np.transpose(photo_1, (2, 0, 1)), 0)).type(torch.FloatTensor)
            photo_2 = torch.from_numpy(np.expand_dims(np.transpose(photo_2, (2, 0, 1)), 0)).type(torch.FloatTensor)

            if self.cuda:
                photo_1 = photo_1.cuda()
                photo_2 = photo_2.cuda()

            # ---------------------------------------------------#
            #   Get the prediction result, output is the probability
            # ---------------------------------------------------#
            output = self.net([photo_1, photo_2])[0]
            output = torch.nn.Sigmoid()(output)

        # plt.subplot(1, 2, 1)
        # plt.imshow(np.array(image_1))

        # plt.subplot(1, 2, 2)
        # plt.imshow(np.array(image_2))
        # plt.text(-12, -12, 'Similarity:%.3f' % output, ha='center', va= 'bottom',fontsize=11)
        # plt.show()
        return output
