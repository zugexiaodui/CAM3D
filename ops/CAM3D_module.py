import torch
import cv2
from torch.nn import functional as F
from PIL import Image
import numpy as np

class ClassAttentionMapping():
    def __init__(self, model, finalconv_name,param_idx,norm_param):
        self.net = model
        self.net.eval()
        self.finalconv_name = finalconv_name
        self.norm_param = norm_param
        print("norm_param:",norm_param)
        self.features_blobs = []
        self.param_idx = -1
        if type(param_idx) == str:
            for i,(n,p) in enumerate(self.net.named_parameters()):
                if n==param_idx: self.param_idx = i
        elif type(param_idx) == int:
            self.param_idx = param_idx
        else:
            raise TypeError(type(param_idx))
        assert self.param_idx > 0,"param_idx:{} is not in mode.parameters() or it's not a valid number.".format(param_idx)

    def get_output_feature(self, layer_name):
        features_blobs = []
        def hook_feature(module, input, output): features_blobs.append(output.data.cpu().numpy())
        layer = self.net._modules['base_model']._modules['layer4']
        layer.register_forward_hook(hook_feature)
        self.features_blobs = features_blobs
    
    def unnormalize(self,img_tensor):
        src_img = img_tensor.squeeze(0).cpu().numpy()
        tc,h,w = src_img.shape
        src_img = src_img.reshape(-1,3,h,w).transpose(0,2,3,1)
        for ti in range(src_img.shape[0]):
            for c,(m,s) in enumerate(zip(*self.norm_param)):
                src_img[ti,:,:,c]*=s
                src_img[ti,:,:,c]+=m
                src_img[ti,:,:,c]*=255
            src_img[ti] = cv2.cvtColor(np.uint8(np.round(src_img[ti])),cv2.COLOR_RGB2BGR)
        return src_img

    def returnCAM(self, feature_conv, class_weight):
        size_upsample = (256, 256)
        t, nc, h, w = feature_conv.shape
        feature_conv = np.transpose(feature_conv,(1,0,2,3)).reshape(nc, t*h*w)
        cam = class_weight.dot(feature_conv)
        cam = cam.reshape(t, h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam = np.zeros((t,*size_upsample),dtype=np.uint8)
        for i in range(t):
            output_cam[i] = cv2.resize(cam_img[i], size_upsample)
        return output_cam

    def __call__(self, img_tensor, class_idx=None):
        self.get_output_feature(self.finalconv_name)
        logit = self.net(img_tensor)
        h_x = F.softmax(logit, dim=1).data.squeeze()
        idx = None
        if type(class_idx) == type(None):
            probs, idxes = h_x.sort(0, True)
            prob = probs.cpu().numpy()[0]
            idx = idxes.cpu().numpy()[0]
            print("idx={}, prob={:.4f}".format(idx, prob))
        else:
            idx = class_idx
        class_weight = np.squeeze(list(self.net.parameters())[self.param_idx].data.cpu().numpy())[idx]
        cam_imgs = self.returnCAM(self.features_blobs[0],class_weight)
        src_imgs = self.unnormalize(img_tensor)
        t,w,h,c = src_imgs.shape
        dst_imgs = np.zeros((t,h,w,c),dtype=np.uint8)
        for ti in range(t):
            heatmap = cv2.applyColorMap(cv2.resize(cam_imgs[ti],(w,h)), cv2.COLORMAP_JET)
            dst_imgs[ti] = heatmap * 0.3 + src_imgs[ti] * 0.5
        return (src_imgs,dst_imgs)