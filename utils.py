from tensorpack import *
from tensorpack.utils.viz import *
import tensorflow as tf
import numpy as np

SHAPE = 128
BATCH = 16
TEST_BATCH = 32
NF = 64  # channel size


def INReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
    x = InstanceNorm('inorm', x)
    return LeakyReLU(x, name=name)


def BNLReLU(x, name):
    x = BatchNorm('bn', x)
    return LeakyReLU(x)


def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=8, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.03
    K2 = 0.05
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    sigma1_sq = tf.abs(sigma1_sq)
    sigma2_sq = tf.abs(sigma2_sq)
    sigma12 = tf.abs(sigma12)
    if cs_map:

        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    #From NCHW to NHWC
    img1 = tf.transpose(img1, [0, 2, 3, 1])
    img2 = tf.transpose(img2, [0, 2, 3, 1])

    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.stack(mssim, axis=0)
    mcs = tf.stack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_dssim(img1, img2):
    img1 = tf.unstack(tf.expand_dims(img1, axis=2), axis=1)
    img2 = tf.unstack(tf.expand_dims(img2, axis=2), axis=1)
    value = tf.stack([tf_ms_ssim(i1, i2) for i1, i2 in zip(img1, img2)], axis=0)
    return tf.subtract(1.0, tf.reduce_sum(value)/3, name='DSSIM_loss')


def get_celebA_data(datadir):
    dfA = ImageFromFile(glob(datadir + "/trainA/*.jpg"), channel=3, shuffle=True)
    dfB = ImageFromFile(glob(datadir + "/trainB/*.jpg"), channel=3, shuffle=True)
    df = JoinData([dfA, dfB])
    augs = [imgaug.Resize(SHAPE)]
    df = AugmentImageComponents(df, augs, (0,1))
    df = BatchData(df, BATCH)
    df = PrefetchDataZMQ(df, 5)
    return df


def get_data(datadir, isTrain=True):
    if isTrain:
        resize_range = (0.9, 1.1)
        augs = [
            imgaug.Flip(horiz=True),
            imgaug.ResizeShortestEdge(int(SHAPE * 1.12)),
            imgaug.Rotation(30),
            imgaug.RandomCrop(int(SHAPE * 1.12)),
            imgaug.RandomResize(resize_range, resize_range,
                aspect_ratio_thres=0),
            imgaug.RandomCrop(SHAPE),
        ]
    else:
        augs = [imgaug.ResizeShortestEdge(int(SHAPE * 1.12)),
            imgaug.CenterCrop(SHAPE)
        ]

    def get_image_pairs(dir1, dir2):
        def get_df(dir):
            files = sorted(glob(os.path.join(dir, '*.jpg')) +
                glob(os.path.join(dir, '*.png')))
            df = ImageFromFile(files, channel=3, shuffle=isTrain)
            return AugmentImageComponent(df, augs)
        return JoinData([get_df(dir1), get_df(dir2)])

    names = ['trainA', 'trainB'] if isTrain else ['testA', 'testB']
    df = get_image_pairs(*[os.path.join(datadir, n) for n in names])
    df = BatchData(df, BATCH if isTrain else TEST_BATCH)
    df = PrefetchDataZMQ(df, 8 if isTrain else 1)
    return df

class VisualizeTestSet(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['inputA', 'inputB'], ['viz_A_recon', 'viz_B_recon'])

    def _before_train(self):
        global args
        self.val_ds = get_data(args.data, isTrain=False)

    def _trigger(self):
        idx = 0
        for iA, iB in self.val_ds.get_data():
            vizA, vizB = self.pred(iA, iB)
            self.trainer.monitors.put_image('testA-{}'.format(idx), vizA)
            self.trainer.monitors.put_image('testB-{}'.format(idx), vizB)
            idx += 1
