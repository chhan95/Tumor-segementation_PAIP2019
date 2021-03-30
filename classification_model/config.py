import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa


####
class Config(object):
    def __init__(self):
        self.seed = 5
        self.init_lr = 1.0e-4
        self.lr_steps = 60  # decrease at every n-th epoch
        self.train_batch_size = 10
        self.infer_batch_size = 10
        self.nr_epochs = 100
        self.nr_classes = 2

        # nr of processes for parallel processing input
        self.nr_procs_train = 8
        self.nr_procs_valid = 4

        self.nr_fold = 5
        self.fold_idx = 4
        self.cross_valid = False

        self.load_network = False
        self.save_net_path = ""

        self.data_size = [1500, 1500]
        self.input_size = [1024, 1024]

        #
        self.dataset = 'PAIP'
        # v1.0.3.0 test classifying cancer only
        self.logging = True  # for debug run only
        self.log_path = 'log/%s/' % self.dataset

        self.chkpts_prefix = 'model'
        self.model_name = 'v1.0.0.3'
        self.log_dir = self.log_path + self.model_name

    def train_augmentors(self):
        shape_augs = [
            iaa.Resize({
                "height": self.data_size[0],
                "width": self.data_size[1]}
            ),
            iaa.Affine(
                # scale images to 80-120% of their size, individually per axis
                scale={"x": (0.8, 1.2),
                       "y": (0.8, 1.2)},
                # translate by -A to +A percent (per axis)
                translate_percent={"x": (-0.01, 0.01),
                                   "y": (-0.01, 0.01)},
                rotate=(-179, 179),  # rotate by -179 to +179 degrees
                shear=(-5, 5),  # shear by -5 to +5 degrees
                order=[0],  # use nearest neighbour
                backend='cv2'  # opencv for fast processing
            ),
            iaa.Fliplr(0.5),  # horizontally flip 50% of all images
            iaa.CropToFixedSize(self.input_size[0], self.input_size[1],
                                position='center'),
        ]
        #
        input_augs = [
            iaa.OneOf([
                iaa.GaussianBlur((0, 3.0)),  # gaussian blur with random sigma
                iaa.MedianBlur(k=(3, 5)),  # median with random kernel sizes
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            ]),
            iaa.Sequential([
                iaa.Add((-26, 26)),
                iaa.AddToHueAndSaturation((-10, 10)),
                iaa.LinearContrast((0.8, 1.2), per_channel=1.0),
            ], random_order=True),
        ]
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            iaa.Resize({
                "height": self.input_size[0],
                "width": self.input_size[1]}
            ),
        ]
        return shape_augs, None
############################################################################