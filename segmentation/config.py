import imgaug  # https://github.com/aleju/imgaug
from imgaug import augmenters as iaa


####
class Config(object):
    def __init__(self):
        self.seed = 5
        self.init_lr = 1.0e-4
        self.lr_steps = 60  # decrease at every n-th epoch
        self.train_batch_size = 8

        self.infer_batch_size = 8
        self.nr_epochs = 120
        self.nr_classes =2

        # nr of processes for parallel processing input
        self.nr_procs_train = 12
        self.nr_procs_valid = 8

        self.load_network = False
        self.save_net_path = "log/v1.0.0.1_VIABLE_FOCAL/model_net_18.pth"

        self.data_size = [1024, 1024]
        self.input_size = [512, 512]

        self.logging = True  # for debug run only
        self.log_path = 'log/'
        self.chkpts_prefix = 'model'
        self.model_name = 'v1.0.0.3_epoch100'
        self.log_dir = self.log_path + self.model_name

        # self.inf_manual_chkpts = True
        # # self.inf_model_path  = self.log_dir + '/model_net_120.pth'
        # self.inf_model_path = 'pretrained/model_net_119.pth'
        # self.inf_eval_metric = 'valid_acc'
        #
        # inf_tma_code = 'BT140005'
        # inf_output_path = '/media/vqdang/Data_3/infer/KBSMC/BREAST/epi_str/'
        # # self.inf_output_dir = '%s/v1.0.0.4/%s/' % (inf_output_path, inf_tma_code)
        # self.inf_output_dir = '%s/%s/' % (inf_output_path, inf_tma_code)
        #
        # self.inf_imgs_ext = '.jpg'
        # inf_input_path = '/media/vqdang/Data/Workspace/KBSMC/BREAST/'
        # self.inf_imgs_dir = '%s/%s/imgs_x/' % (inf_input_path, inf_tma_code)

    def train_augmentors(self):
        shape_augs = [
            iaa.PadToFixedSize(
                self.data_size[0],
                self.data_size[1],
                pad_cval=255,
                position='center',
                deterministic=True),
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
            iaa.Flipud(0.5),  # vertically flip 20% of all images
            iaa.CropToFixedSize(self.input_size[0], self.input_size[1],
                                position='center',
                                deterministic=True),
            # iaa.CropToFixedSize(1440, 1440)
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
                iaa.AddToHueAndSaturation((-20, 20)),
                iaa.LinearContrast((0.75, 1.25), per_channel=1.0),
            ], random_order=True),
        ]
        return shape_augs, input_augs

    ####
    def infer_augmentors(self):
        shape_augs = [
            # iaa.Scale(
            #     {"height": 1280, 
            #      "width" : 1280},
            #     interpolation='nearest',
            #     ),
            iaa.PadToFixedSize(
                               self.data_size[0],
                               self.data_size[1],
                               pad_cval=255,
                               position='center',
                               deterministic=True),
            iaa.CropToFixedSize(self.input_size[0], self.input_size[1],
                                position='center',
                                deterministic=True),
        ]
        return shape_augs

############################################################################
