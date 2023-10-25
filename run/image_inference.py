import subprocess
import sys

import os
import sys
import tqdm
import torch
import shutil 
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data.dataloader import DataLoader
import time

filepath = os.path.split(os.path.abspath(__file__))[0]
repopath = os.path.split(filepath)[0]
sys.path.append(repopath)

from lib import *
from utils.misc import *
from data.dataloader import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def process_input_image(pred_pil): # PIL image
    lr = np.array(pred_pil.convert('RGB'))
    lr = lr[::].astype(np.float32).transpose([2, 0, 1]) / 255.0
    inputs = torch.as_tensor([lr])
    return inputs

def process_output_image(result, file_name=None):
    result = result.data.cpu().numpy()
    result = result[0].transpose((1, 2, 0)) 
    
    print("max(result)_before", np.max(result))
    print("mean(result)_before", np.mean(result))
    print("min(result)_before", np.min(result))

    result = (result - np.min(result)) / (np.max(result) - np.min(result)) * 255

    print("max(result)_after", np.max(result))
    print("mean(result)_after", np.mean(result))
    print("min(result)_after", np.min(result))

    # result = np.array(Image.fromarray(result))#[:,:,:1]
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)[:,:,:1]#.astype(np.uint8)
    # result = ImageLoader.save_image(result, f'{file_name}.png')                        
    return result

def test(opt, args):
    model = eval(opt.Model.name)(**opt.Model)
    
    if opt.Inference.Checkpoint.device == "gpu":
        print("Using gpu")
        model.load_state_dict(torch.load(os.path.join(opt.Inference.Checkpoint.checkpoint_dir)), strict=True)
        model.cuda()
    else:
        print("Using cpu")
        model.load_state_dict(torch.load(os.path.join(opt.Inference.Checkpoint.checkpoint_dir), map_location='cpu'), strict=True)

    model.eval()
    
    save_path = opt.Inference.Dataset.dest

    print('------Loading dataset-----')

    test_dataset = eval(opt.Inference.Dataset.type)(opt.Inference.Dataset.source, opt.Inference.Dataset.transforms)
    test_loader  = DataLoader(dataset=test_dataset, batch_size=1, num_workers=opt.Inference.Dataloader.num_workers, pin_memory=opt.Inference.Dataloader.pin_memory)

    if args.verbose is True:
        samples = tqdm.tqdm(test_loader, desc='Test', total=len(test_loader),
                            position=1, leave=False, bar_format='{desc:<30}{percentage:3.0f}%|{bar:50}{r_bar}')
    else:
        samples = test_loader

    print('-----Inference-----')

    for i, sample in enumerate(samples):
        if opt.Inference.Checkpoint.device == "gpu":
            sample = to_cuda(sample)

        with torch.no_grad():
            out = model(sample)

        pred = to_numpy(out['pred'], (384, 384))
        pred = (pred * 255).astype(np.uint8)
        
        width_new, height_new = sample['shape'][0].item(), sample['shape'][1].item() 
        
        pred = cv2.cvtColor(pred,cv2.COLOR_GRAY2RGB)
        
        pred_pil = Image.fromarray(pred)
        
        results = []
        
        ###################################################
        ## Not working
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "/content/InSPyReNet/opencv_sr_models/EDSR_x4.pb"
        # sr.readModel(path)
        # sr.setModel("edsr",4)
        # t = time.time()
        # result = sr.upsample(pred).astype("uint8")
        # print("Time: ", time.time()-t)
        # # cv2.imwrite(os.path.join(save_path, 'edsr.jpeg'), result)
        # results.append(result)
        ###################################################
        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        path = "/content/InSPyReNet/opencv_sr_models/ESPCN_x4.pb"
        sr.readModel(path)
        sr.setModel("espcn",4)
        t = time.time()

        result = sr.upsample(pred).astype("uint8")

        # two times in a row
        result = sr.upsample(result).astype("uint8")
        # result = cv2.resize(result, (result.shape[0]*4, result.shape[1]*4), interpolation=cv2.INTER_LANCZOS4)

        # if max(width_new, height_new) > 1536:
        #   path = "/content/InSPyReNet/opencv_sr_models/ESPCN_x2.pb"
        #   sr.readModel(path)
        #   sr.setModel("espcn",2)
        #   result = sr.upsample(result).astype("uint8")

        print("Time: ", time.time()-t)
        # cv2.imwrite(os.path.join(save_path, f'espcn_{i}.jpeg'), result)
        results.append(result)
        # ###################################################
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "/content/InSPyReNet/opencv_sr_models/FSRCNN_x4.pb"
        # sr.readModel(path)
        # sr.setModel("fsrcnn",4)
        # t = time.time()
        # result = sr.upsample(pred).astype("uint8")
        # print("Time: ", time.time()-t)
        # #cv2.imwrite(os.path.join(save_path, 'fsrcnn.jpeg'), result)
        # results.append(result)
        ###################################################
        ## Not working
        # sr = cv2.dnn_superres.DnnSuperResImpl_create()
        # path = "/content/InSPyReNet/opencv_sr_models/LapSRN_x4.pb"
        # sr.readModel(path)
        # sr.setModel("lapsrn",4)
        # t = time.time()
        # result = sr.upsample(pred).astype("uint8")
        # print("Time: ", time.time()-t)
        # # cv2.imwrite(os.path.join(save_path, 'lapsrn.jpeg'), result)
        # results.append(result)
        ###################################################
        # t = time.time()
        # model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=4)
        # print("Phase 1: ", time.time()-t)

        # t = time.time()
        # inputs = process_input_image(pred_pil)
        # result = model(inputs)
        # print("Phase 2: ", time.time()-t)
        # # result = process_output_image(result)
        # #cv2.imwrite(os.path.join(save_path, 'edsr-base.jpeg'), result)
        # process_output_image(result, "edsr-base")
        # results.append(result)
        # # ###################################################
        # t = time.time()
        # model = A2nModel.from_pretrained('eugenesiow/a2n', scale=4)
        # print("Phase 1: ", time.time()-t)

        # t = time.time()
        # inputs = process_input_image(pred_pil)
        # result = model(inputs)
        # print("Phase 2: ", time.time()-t)
        # result = process_output_image(result)
        # #cv2.imwrite(os.path.join(save_path, 'a2n.jpeg'), result)

        # results.append(result)
        # # ###################################################
        # t = time.time()
        # model = PanModel.from_pretrained('eugenesiow/pan', scale=4)
        # print("Phase 1: ", time.time()-t)

        # t = time.time()
        # inputs = process_input_image(pred_pil)
        # result = model(inputs)
        # print("Phase 2: ", time.time()-t)
        # result = process_output_image(result)
        # #cv2.imwrite(os.path.join(save_path, 'pan.jpeg'), result)
        # results.append(result)
        # # ###################################################
        # t = time.time()
        # model = AwsrnModel.from_pretrained('eugenesiow/awsrn-bam', scale=4)
        # print("Phase 1: ", time.time()-t)

        # t = time.time()
        # inputs = process_input_image(pred_pil)
        # result = model(inputs)
        # print("Phase 2: ", time.time()-t)
        # result = process_output_image(result)
        # #cv2.imwrite(os.path.join(save_path, 'awsrn-bam.jpeg'), result)
        # results.append(result)
        # ###################################################
        # size = pred_pil.size
        # result = np.array(pred_pil.resize((size[0]*4, size[1]*4), Image.LANCZOS)).astype("uint8")
        # #cv2.imwrite(os.path.join(save_path, 'lanczos.jpeg'), result)
        # results.append(result)
        # ###################################################

        for i, pred_np in enumerate(results):  
            print("index:", i)
            threshold_otsu, pred_np = cv2.threshold(pred_np[:,:,0].astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            # print("threshold_otsu:", threshold_otsu)

            pred_np = cv2.GaussianBlur(pred_np,(21,21),0)

            pred_np = cv2.resize(pred_np, dsize=(height_new, width_new), interpolation=cv2.INTER_AREA)

            pred_pil = Image.fromarray(np.uint8(pred_np))

            # pred_pil = pred_pil.resize((height_new, width_new), Image.LANCZOS)

            # pred_np = np.array(pred_pil)

            # if max(width_new, height_new) < 1000:
            #     krn = 3
            #     itr = 1
            # elif max(width_new, height_new) >= 1000 or max(width_new, height_new) < 2000:
            #     krn = 5
            #     itr = 1
            # else:
            #     krn = 7
            #     itr = 2

            # kernel = np.ones((krn, krn), np.uint8)
            # pred_np = cv2.erode(pred_np, kernel, iterations=itr)

            # pred_pil = Image.fromarray(np.uint8(pred_np))

            image = Image.open(os.path.join(opt.Inference.Dataset.source, sample['name'][0]+'.jpg')).convert('RGB')
            final_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2RGBA)
            final_image[:,:,3] = pred_np

            final_image = Image.fromarray(final_image)
            # final_image.save(os.path.join(save_path, 'transparent' + sample['name'][0] + f'_{i}.jpg'))

            # white background 
            new_image = Image.new("RGBA", final_image.size, "WHITE")
            new_image.paste(final_image, (0, 0), final_image)
            new_image.convert('RGB').save(os.path.join(save_path, sample['name'][0] + '.png'))

            #mask only 
            #Image.fromarray((pred * 255).astype(np.uint8)).save(os.path.join(save_path, 'mask' + sample['name'][0] + '.png'))
        
if __name__ == "__main__":
    args = parse_args()
    opt = load_config(args.config)
    test(opt, args)


