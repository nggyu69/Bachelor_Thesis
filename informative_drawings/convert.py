import os
import torch
from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import shutil

from informative_drawings.model import Generator, GlobalGenerator2, InceptionV3
from informative_drawings.dataset import UnpairedDepthDataset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='name of this experiment')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where the model checkpoints are saved')
parser.add_argument('--results_dir', type=str, default='results', help='where to save result images')
parser.add_argument('--geom_name', type=str, default='feats2Geom', help='name of the geometry predictor')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--depthroot', type=str, default='', help='dataset of corresponding ground truth depth maps')

parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--geom_nc', type=int, default=3, help='number of channels of geometry data')
parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for the geometry loss')
parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
parser.add_argument('--midas', type=int, default=0, help='use midas depth map')

parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--size', type=int, default=640, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation', default=True)
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')

parser.add_argument('--mode', type=str, default='test', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=640, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=640, help='then crop to this size')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='none', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')

parser.add_argument('--predict_depth', type=int, default=0, help='run geometry prediction on the generated images')
parser.add_argument('--save_input', type=int, default=0, help='save input image')
parser.add_argument('--reconstruct', type=int, default=0, help='get reconstruction')
parser.add_argument('--how_many', type=int, default=1000000, help='number of images to test')

opt = parser.parse_args()

def process_images(control_path, output_path, model_name, opt=opt):
    
    opt.name = model_name
    opt.checkpoints_dir = "Bachelor_Thesis/Info_Drawing_Files/checkpoints"

    opt.dataroot = control_path
    opt.results_dir = output_path

    with torch.no_grad():
        
        net_G = 0
        net_G = Generator(3, 1, 3)
        net_G.cuda()

        net_G.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_latest.pth')))
        net_G.eval()

        transforms_r = [transforms.Resize(640, Image.BICUBIC),
                        transforms.ToTensor()]
        
        test_data = UnpairedDepthDataset(opt.dataroot, '', opt, transforms_r=transforms_r, 
                mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)
        
        dataloader = DataLoader(test_data, batch_size=opt.batchSize, shuffle=False)

        split_name = opt.dataroot.split('/')[-2]
        
        full_output_dir = os.path.join(opt.results_dir, opt.name, split_name)

        # if not os.path.exists(full_output_dir+"/images"):
        #     os.makedirs(full_output_dir+"/images")

        # if not os.path.exists(full_output_dir+"/labels"):
        #     os.makedirs(full_output_dir+"/labels")

        for i, batch in enumerate(dataloader):

            img_r  = Variable(batch['r']).cuda()

            real_A = img_r

            name = batch['name'][0]
            
            input_image = real_A
            image = net_G(input_image)
            save_image(image.data, full_output_dir+f'/images/{name}.png')

            
            shutil.copyfile(os.path.split(opt.dataroot)[0]+f'/labels/{name}.txt', full_output_dir+f'/labels/{name}.txt')
            print('Saved %s' % name)

