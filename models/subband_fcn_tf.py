import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, PReLU
from tensorflow.keras import backend as k
import tensorflow_probability as tfp

import numpy as np
from scipy.io import savemat
import sys 

from os.path import abspath, dirname

sys.path.append(dirname(abspath(__file__)))

from models.bit_estimator import bit_estimator
from models.analysis_prior import Analysis_prior_net
from models.synthesis_prior import Synthesis_prior_net

from utils.references_tf import (reference_P3_42_AG_tf_par_old,
                           reference_P2_42_2D_AG_tf_par, 
                           reference_P1_42_2D_AG_tf_par,
                           reference_U_tf_par,
                           target_P1_P2_tf_par, 
                           target_P3_tf_par,
                           target_U_tf_par_v2,
                           ideal_approx_2)

from losses import (L_beta, Log_L_beta, quant_entropy)

class weighted_model(tf.keras.Model):
    def __init__(self, model1 = None, model2 = None, model3 = None):
        super(weighted_model, self).__init__()
        if model1 == None:
            self.model_X3 = self.fcn_layers(40)
        else:
            self.model_X3 = self.fcn_layers(40)
            for i in range(len(self.model_X3.trainable_weights)):
                self.model_X3.trainable_weights[i] = model3.trainable_weights[i]
        if model2 == None:
            self.model_X2 = self.fcn_layers(22)
        else:
            self.model_X2 = self.fcn_layers(22)
            for i in range(len(self.model_X2.trainable_weights)):
                self.model_X2.trainable_weights[i] = model2.trainable_weights[i]
        if model1 == None:
            self.model_X1 = self.fcn_layers(14)
        else:
            self.model_X1 = self.fcn_layers(14)
            for i in range(len(self.model_X1.trainable_weights)):
                self.model_X1.trainable_weights[i] = model1.trainable_weights[i]
                
        self.reference_P3_42_AG_tf_par_old = reference_P3_42_AG_tf_par_old
        self.reference_P2_42_2D_AG_tf_par = reference_P2_42_2D_AG_tf_par
        self.reference_P1_42_2D_AG_tf_par = reference_P1_42_2D_AG_tf_par
        self.target_P1_P2_tf_par = target_P1_P2_tf_par
        self.target_P3_tf_par = target_P3_tf_par
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, image):
       
        x0 = image[::2,::2]
        x1 = image[::2,1::2]
        x2 = image[1::2,::2]
        x3 = image[1::2,1::2]
        
        ref_P3 = self.reference_P3_42_AG_tf_par_old(x0, x1, x2)
        y_P3 = self.target_P3_tf_par(x3)
        
        pred_X3 = self.model_X3(ref_P3)
        
        x_dd = (y_P3 - pred_X3)
        x_dd = tf.reshape(x_dd, [x0.shape[0],x0.shape[1]])

        ref_P1 = self.reference_P1_42_2D_AG_tf_par(x0, x_dd)
        ref_P2 = self.reference_P2_42_2D_AG_tf_par(x0, x1, x_dd)

        y_P1, y_P2 = self.target_P1_P2_tf_par(x1, x2)

        pred_X1 = self.model_X1(ref_P1)
        pred_X2 = self.model_X2(ref_P2)
        
        return pred_X1, pred_X2, pred_X3, y_P1, y_P2, y_P3
    
    
class multires_weighted_model(tf.keras.Model):
    def __init__(self, Weights, weights_CW, betas, args, levels=3, backward = False):
        super(multires_weighted_model, self).__init__()
        self.priorEncoder = Analysis_prior_net()
        self.priorDecoder = Synthesis_prior_net()
        self.backward = backward
        self.ssim_loss = args.ssim_loss
        self.args = args
        self.balle18 = False#True
        self.Weights = Weights
        self.Weights_CW = weights_CW
        self.betas = betas
        self.quant_image = args.quant_image
        self.end_to_end = args.end_to_end
        if self.end_to_end:
            #self.bit_estimator = bit_estimator()
            #self.bit_estimator_approx = bit_estimator()
            self.bit_estimator = None
        else:
            self.bit_estimator = None
        
        self.model_X3_1 = self.fcn_layers(40)
        self.model_X3_2 = self.fcn_layers(40)
        self.model_X3_3 = self.fcn_layers(40)
        
        self.model_X2_1 = self.fcn_layers(22)
        self.model_X2_2 = self.fcn_layers(22)
        self.model_X2_3 = self.fcn_layers(22)
        
        self.model_X1_1 = self.fcn_layers(14)
        self.model_X1_2 = self.fcn_layers(14)
        self.model_X1_3 = self.fcn_layers(14)
        
        self.model_U_1 = self.fcn_layers(8)
        self.model_U_2 = self.fcn_layers(8)
        self.model_U_3 = self.fcn_layers(8)
                
        self.reference_P3_42_AG_tf_par_old = reference_P3_42_AG_tf_par_old
        self.reference_P2_42_2D_AG_tf_par = reference_P2_42_2D_AG_tf_par
        self.reference_P1_42_2D_AG_tf_par = reference_P1_42_2D_AG_tf_par
        self.reference_U_tf_par = reference_U_tf_par
        self.target_P1_P2_tf_par = target_P1_P2_tf_par
        self.target_P3_tf_par = target_P3_tf_par
        self.target_U_tf_par_v2 = target_U_tf_par_v2
        self.use_ideal_approx = args.use_ideal_approx
        self.kernel = self.get_kernel()
        self.levels = levels
        
    def get_kernel(self):
        N = 15
        t = np.arange(-N,N+1)
        h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
        h_2d = h*np.transpose(h)
        return tf.convert_to_tensor(h_2d)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, image):
        outs = []
        approx = image[:]
        ideal_approx2 = ideal_approx_2(image, tf.cast(self.kernel, image.dtype))
        subband = []
        H, W = image.shape
        for l in range(self.levels):
            x0 = approx[::2,::2]
            x1 = approx[::2,1::2]
            x2 = approx[1::2,::2]
            x3 = approx[1::2,1::2]

            ref_P3 = self.reference_P3_42_AG_tf_par_old(x0, x1, x2)
            y_P3 = self.target_P3_tf_par(x3)
            
            if l == 0:
                pred_X3 = self.model_X3_1(ref_P3)
            elif l == 1:
                pred_X3 = self.model_X3_2(ref_P3)
            elif l == 2:
                pred_X3 = self.model_X3_3(ref_P3)
            
            x_dd = (y_P3 - pred_X3)
            x_dd = tf.reshape(x_dd, [x0.shape[0],x0.shape[1]])
            
            ref_P1 = self.reference_P1_42_2D_AG_tf_par(x0, x_dd)
            ref_P2 = self.reference_P2_42_2D_AG_tf_par(x0, x1, x_dd)

            y_P1, y_P2 = self.target_P1_P2_tf_par(x1, x2)
            
            if l == 0:
                pred_X1 = self.model_X1_1(ref_P1)
                pred_X2 = self.model_X2_1(ref_P2)
            elif l == 1:    
                pred_X1 = self.model_X1_2(ref_P1)
                pred_X2 = self.model_X2_2(ref_P2)
            elif l == 2:    
                pred_X1 = self.model_X1_3(ref_P1)
                pred_X2 = self.model_X2_3(ref_P2)
            
            x_dh = (y_P1 - pred_X1)
            x_dh = tf.reshape(x_dh, [x0.shape[0],x0.shape[1]])
            x_dv = (y_P2 - pred_X2)
            x_dv = tf.reshape(x_dv, [x0.shape[0],x0.shape[1]])
            
            
            ref_U = reference_U_tf_par(x_dd, x_dv, x_dh)
            if l == 2 and self.use_ideal_approx:
                y_U = target_U_tf_par_v2(ideal_approx2, x0, tf.cast(self.kernel, ideal_approx2.dtype))
            else:
                y_U = target_U_tf_par_v2(approx, x0, tf.cast(self.kernel, approx.dtype))
            
            if l == 0:
                pred_U = self.model_U_1(ref_U)
            elif l == 1:     
                pred_U = self.model_U_2(ref_U)
            elif l == 2:       
                pred_U = self.model_U_3(ref_U)

            approx = x0 + tf.reshape(pred_U, [x0.shape[0],x0.shape[1]])
            
            
            if (not self.end_to_end) and (not self.backward):
                outs = [pred_X1, pred_X2, pred_X3,
                        pred_U, y_P1, y_P2, 
                        y_P3, y_U, ref_U] + outs
            if self.backward:
                if l == self.levels - 1:
                    subband = [approx, x_dh, x_dv, x_dd] + subband
                    del approx, x_dh, x_dv, x_dd
                else:
                    subband = [x_dh, x_dv, x_dd] + subband  
                    del x_dh, x_dv, x_dd
        if not self.backward:
            return [approx] + outs
        else:
            if self.end_to_end:
                return self.mse_rate_loss(image, subband)
            else:
                #outs = [approx] + outs + [subband]
                if self.args.wav_loss == 'entropy':
                    wav_loss = self.entropy_loss(subband)
                elif self.args.wav_loss == 'norm':
                    wav_loss = self.norm_loss(subband)
                elif self.args.wav_loss == 'log':
                    wav_loss = self.log_loss(subband)
                mse_loss, ssim_loss = self.reconstruction_loss(image, subband)
                return mse_loss, ssim_loss, wav_loss
            
    #@staticmethod
    def subband_list_to_image(self, subband):
        if self.levels >=1:
            dec_im_level3 = tf.concat([tf.concat([subband[0], subband[1]], 1), 
                                       tf.concat([subband[2], subband[3]], 1)], 0)
        if self.levels >=2:
            dec_im_level2 = tf.concat([tf.concat([dec_im_level3, subband[4]],1),
                                       tf.concat([subband[5], subband[6]], 1)],0)
        if self.levels >=3:
            dec_im = tf.concat([tf.concat([dec_im_level2, subband[7]],1),
                                tf.concat([subband[8], subband[9]], 1)],0)

        return dec_im
    
    def backwards_single_level(self, dec_im, l):
        approx = dec_im[:dec_im.shape[0]//2,:dec_im.shape[1]//2]
        x_dh = dec_im[:dec_im.shape[0]//2,dec_im.shape[1]//2:]
        x_dv = dec_im[dec_im.shape[0]//2:,:dec_im.shape[1]//2]
        x_dd = dec_im[dec_im.shape[0]//2:,dec_im.shape[1]//2:]
        
        ref_U = reference_U_tf_par(x_dd, x_dv, x_dh)
        if l == 0:
            pred_U = self.model_U_1(ref_U)
        elif l == 1:     
            pred_U = self.model_U_2(ref_U)
        elif l == 2:       
            pred_U = self.model_U_3(ref_U)

        x0 = approx - tf.reshape(pred_U, [approx.shape[0],approx.shape[1]])
        
        del ref_U, pred_U
        
        ref_P1 = self.reference_P1_42_2D_AG_tf_par(x0, x_dd)
        if l == 0:
            pred_X1 = self.model_X1_1(ref_P1)
        elif l == 1:    
            pred_X1 = self.model_X1_2(ref_P1)
        elif l == 2:    
            pred_X1 = self.model_X1_3(ref_P1)
            
        x1 = x_dh + tf.reshape(pred_X1, [x0.shape[0],x0.shape[1]])
        
        del x_dh, pred_X1, ref_P1
        
        ref_P2 = self.reference_P2_42_2D_AG_tf_par(x0, x1, x_dd)
        if l == 0:
            pred_X2 = self.model_X2_1(ref_P2)
        elif l == 1:    
            pred_X2 = self.model_X2_2(ref_P2)
        elif l == 2:
            pred_X2 = self.model_X2_3(ref_P2)
        
        x2 = x_dv + tf.reshape(pred_X2, [x0.shape[0],x0.shape[1]])
        
        del x_dv, pred_X2, ref_P2 
        
        ref_P3 = self.reference_P3_42_AG_tf_par_old(x0, x1, x2)
        if l == 0:
            pred_X3 = self.model_X3_1(ref_P3)
        elif l == 1:
            pred_X3 = self.model_X3_2(ref_P3)
        elif l == 2:
            pred_X3 = self.model_X3_3(ref_P3)
        
        x3 = x_dd + tf.reshape(pred_X3, [x0.shape[0],x0.shape[1]])
        
        del ref_P3, pred_X3, x_dd
        
        return self.merge_subbands(x0,x1,x2,x3)
    
    @staticmethod
    def merge_subbands(a,b,c,d):
        image1 = tf.reshape(tf.stack([a, b], axis=-1),[tf.shape(a)[0], tf.shape(a)[1]*2])
        image2 = tf.reshape(tf.stack([c, d], axis=-1),[tf.shape(c)[0], tf.shape(c)[1]*2])
        image1_t = tf.transpose(image1, [1,0])
        image2_t = tf.transpose(image2, [1,0])
        image = tf.reshape(tf.stack([image1_t, image2_t], axis=-1),[tf.shape(c)[1]*2, tf.shape(c)[0]*2])
        image = tf.transpose(image, (1,0))
        return image
    
    def backwards(self, dec_im):
        H, W = dec_im.shape
        H = H//(2**self.levels)
        W = W//(2**self.levels)
        for l in range(self.levels-1,-1,-1):
            if l == self.levels-1:
                Is = tf.concat([tf.concat([dec_im[:H,:W], dec_im[:H,W:W*2]],1),
                               tf.concat([dec_im[H:H*2,:W], dec_im[H:H*2,W:W*2]],1)],0)
            else:
                Is = tf.concat([tf.concat([origin_img_in_progress, dec_im[:H,W:W*2]],1),
                               tf.concat([dec_im[H:H*2,:W], dec_im[H:H*2,W:W*2]],1)],0)
            origin_img_in_progress = self.backwards_single_level(Is, l)
            H, W = H*2, W*2
            
        return origin_img_in_progress
    
    def reconstruction_loss(self, image, subbands, which_subband = 'all'):
        #approx, x_dh, x_dv, x_dd] + subband
        loss = tf.zeros([1])
        ssim_loss = tf.zeros([1])

        if which_subband == 'all':
            for i in range(4):
                if i == 0:
                    subbands[7] = tf.zeros(subbands[7].shape)
                elif i == 1:
                    subbands[8] = subbands[9] = tf.zeros(subbands[8].shape)
                elif i == 2:
                    subbands[6] = subbands[5] = subbands[4] = tf.zeros(subbands[6].shape)
                elif i == 3:
                    subbands[3] = subbands[2] = subbands[1] = tf.zeros(subbands[3].shape)
                    
                quant_dec_im = self.subband_list_to_image(subbands)
                rec_im = self.backwards(quant_dec_im)
                if not self.ssim_loss:
                    loss += k.mean(k.square(image - rec_im)) 
                if self.ssim_loss:
                    x = tf.cast(tf.expand_dims(tf.expand_dims(image,0),3), dtype=tf.float32)
                    y = tf.expand_dims(tf.expand_dims(rec_im,0),3)
                    ssim_loss += 1 - tf.image.ssim(y, x, max_val = 1, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03)

            return loss, ssim_loss

        elif which_subband == 'approx':
            for subbands in [subbands4]:
                quant_dec_im = self.subband_list_to_image(subbands)
                rec_im = self.backwards(quant_dec_im)
                loss += k.mean(k.square(image - rec_im)) 
            return loss, rec_im
        elif which_subband == 'approx_and_all_details':
            for subbands in [subbands1]:
                quant_dec_im = self.subband_list_to_image(subbands)
                rec_im = self.backwards(quant_dec_im)
                loss += k.mean(k.square(image - rec_im)) 
            return loss, rec_im
    
    def entropy_loss(self, subbands):
        return self.Weights[2]*L_beta(subbands[-1], None, self.betas[2])+\
               (1.43*Log_L_beta(subbands[-1], None, self.betas[2], self.Weights_CW[2]))/4+\
               self.Weights[1]*L_beta(subbands[-2], None, self.betas[1])+\
               (1.43*Log_L_beta(subbands[-2], None, self.betas[1], self.Weights_CW[1]))/4+\
               self.Weights[0]*L_beta(subbands[-3], None, self.betas[0])+\
               (1.43*Log_L_beta(subbands[-3], None, self.betas[0], self.Weights_CW[0]))/4+\
               self.Weights[5]*L_beta(subbands[-4], None, self.betas[5])+\
               (1.43*Log_L_beta(subbands[-4], None, self.betas[5], self.Weights_CW[5]))/16+\
               self.Weights[4]*L_beta(subbands[-5], None, self.betas[4])+\
               (1.43*Log_L_beta(subbands[-5], None, self.betas[4], self.Weights_CW[4]))/16+\
               self.Weights[3]*L_beta(subbands[-6], None, self.betas[3])+\
               (1.43*Log_L_beta(subbands[-6], None, self.betas[3], self.Weights_CW[3]))/16+\
               self.Weights[8]*L_beta(subbands[-7], None, self.betas[8])+\
               (1.43*Log_L_beta(subbands[-7], None, self.betas[8], self.Weights_CW[8]))/64+\
               self.Weights[7]*L_beta(subbands[-8], None, self.betas[7])+\
               (1.43*Log_L_beta(subbands[-8], None, self.betas[7], self.Weights_CW[7]))/64+\
               self.Weights[6]*L_beta(subbands[-9], None, self.betas[6])+\
               (1.43*Log_L_beta(subbands[-9], None, self.betas[6], self.Weights_CW[6]))/64+\
               self.Weights[9]*L_beta(subbands[-10], None, 2)+\
               (1.43*Log_L_beta(subbands[-10], None, 2, self.Weights_CW[9]))/64
    
    def norm_loss(self, subbands):
        loss = self.Weights[2]*(L_beta(subbands[-1], 0, self.betas[2]))+\
               self.Weights[1]*(L_beta(subbands[-2], 0, self.betas[1]))+\
               self.Weights[0]*(L_beta(subbands[-3], 0, self.betas[0]))+\
               self.Weights[5]*(L_beta(subbands[-4], 0, self.betas[5]))+\
               self.Weights[4]*(L_beta(subbands[-5], 0, self.betas[4]))+\
               self.Weights[3]*(L_beta(subbands[-6], 0, self.betas[3]))+\
               self.Weights[8]*(L_beta(subbands[-7], 0, self.betas[8]))+\
               self.Weights[7]*(L_beta(subbands[-8], 0, self.betas[7]))+\
               self.Weights[6]*(L_beta(subbands[-9], 0, self.betas[6]))+\
               self.Weights[9]*(L_beta(subbands[-10], 0, 2))
        return loss
    
    def quant_entropy_loss(self, subbands):
        loss = self.Weights[2]*(quant_entropy(subbands[-1], self.betas[2]))+\
               self.Weights[1]*(quant_entropy(subbands[-2], self.betas[1]))+\
               self.Weights[0]*(quant_entropy(subbands[-3], self.betas[0]))+\
               self.Weights[5]*(quant_entropy(subbands[-4], self.betas[5]))+\
               self.Weights[4]*(quant_entropy(subbands[-5], self.betas[4]))+\
               self.Weights[3]*(quant_entropy(subbands[-6], self.betas[3]))+\
               self.Weights[8]*(quant_entropy(subbands[-7], self.betas[8]))+\
               self.Weights[7]*(quant_entropy(subbands[-8], self.betas[7]))+\
               self.Weights[6]*(quant_entropy(subbands[-9], self.betas[6]))+\
               self.Weights[9]*(quant_entropy(subbands[-10], 2))
        return loss
    
        
    def log_loss(self, subbands):
        return (Log_L_beta(subbands[-1], None, self.betas[2], self.Weights_CW[2]))/4+\
               (Log_L_beta(subbands[-2], None, self.betas[1], self.Weights_CW[1]))/4+\
               (Log_L_beta(subbands[-3], None, self.betas[0], self.Weights_CW[0]))/4+\
               (Log_L_beta(subbands[-4], None, self.betas[5], self.Weights_CW[5]))/16+\
               (Log_L_beta(subbands[-5], None, self.betas[4], self.Weights_CW[4]))/16+\
               (Log_L_beta(subbands[-6], None, self.betas[3], self.Weights_CW[3]))/16+\
               (Log_L_beta(subbands[-7], None, self.betas[8], self.Weights_CW[8]))/64+\
               (Log_L_beta(subbands[-8], None, self.betas[7], self.Weights_CW[7]))/64+\
               (Log_L_beta(subbands[-9], None, self.betas[6], self.Weights_CW[6]))/64+\
               (Log_L_beta(subbands[-10], None, 2, self.Weights_CW[9]))/64
    
    def estimate_bits_z(self, z, approx= False):
        if approx:
            prob = self.bit_estimator_approx(z + self.args.quant_step/2) - self.bit_estimator_approx(z - self.args.quant_step/2)
        else:
            prob = self.bit_estimator(z + self.args.quant_step/2) - self.bit_estimator(z - self.args.quant_step/2)
        total_bits = tf.math.reduce_sum(tf.clip_by_value(-1.0 * tf.math.log(prob + 1e-10) / tf.math.log(2.0), 0, 50))
        return total_bits#, prob
    
    def feature_probs_based_sigma(self,feature, sigma):
        if sigma.shape !=feature.shape:
            H,W = feature.shape
            sigma = sigma[:H,:W]
        mu = tf.zeros_like(sigma)
        sigma = tf.clip_by_value(sigma,1e-10, 1e10)
        gaussian = tfp.distributions.Laplace(mu, sigma)
        prob = gaussian.cdf(tf.expand_dims(tf.expand_dims(feature,0),0) + self.args.quant_step/2) - \
        gaussian.cdf(tf.expand_dims(tf.expand_dims(feature,0),0) - self.args.quant_step/2)
        total_bits = tf.math.reduce_sum(tf.clip_by_value(-1.0 * tf.math.log(prob + 1e-10) / tf.math.log(2.0), 0, 50))
        return total_bits
    
    def get_bpp_subband(self, quant_subband, H, W, approx= False):
        total_bits_dec = self.estimate_bits_z(quant_subband, approx)
        return total_bits_dec / (H*W)
    
    def get_bpp_subband_18(self, quant_subband, H, W):  
        z = self.priorEncoder(tf.expand_dims(tf.expand_dims(quant_subband,0),0))
        quant_noise_z = tf.random.uniform(z.shape,-self.args.quant_step/2,self.args.quant_step/2)
        compressed_z = z + quant_noise_z
        recon_sigma = self.priorDecoder(compressed_z)
        total_bits_feature = self.feature_probs_based_sigma(quant_subband, recon_sigma) 
        total_bits_z = self.estimate_bits_z(compressed_z)
        return (total_bits_feature+total_bits_z)/(H*W)
    
    def get_rate_18(self, dec_im, H, W):
        bpp = 0
        for l in range(self.levels):
            dh = dec_im[:H//2,W//2:W]
            dv = dec_im[H//2:H,:W//2]
            dd = dec_im[H//2:H,W//2:W]
            bpp_dh = self.get_bpp_subband_18(dh, H//2, W//2)
            bpp_dv = self.get_bpp_subband_18(dv, H//2, W//2)
            bpp_dd = self.get_bpp_subband_18(dd, H//2, W//2)
            
            bpp += (1/(4**(l+1)))*(bpp_dh+bpp_dv+bpp_dd)#self.Weights[3*l] *self.Weights[(3*l)+1]* self.Weights[(3*l)+2]*
            if l == self.levels-1:
                approx = dec_im[:H//2,:W//2]
                bpp_approx = self.get_bpp_subband_18(approx, H//2, W//2)
                bpp += (1/(4**(l+1)))*bpp_approx#self.Weights[3*self.levels]*
                
            H=H//2
            W=W//2
        del dec_im
        return bpp
    
    def get_bpp_subband_mfabien(self, quant_subband, H, W):
        bc_train = self.pc.bitcost(pc_in, enc_out_train.symbols, is_training=True, pad_value=pc.auto_pad_value(ae))
        
        
    def get_rate_mfabien(self, dec_im, H, W):
        bpp = 0
        for l in range(self.levels):
            dh = dec_im[:H//2,W//2:W]
            dv = dec_im[H//2:H,:W//2]
            dd = dec_im[H//2:H,W//2:W]
            bpp_dh = self.get_bpp_subband_mfabien(dh, H//2, W//2)
            bpp_dv = self.get_bpp_subband_mfabien(dv, H//2, W//2)
            bpp_dd = self.get_bpp_subband_mfabien(dd, H//2, W//2)
            
            bpp += (1/(4**(l+1)))*(bpp_dh+bpp_dv+bpp_dd)
            if l == self.levels-1:
                approx = dec_im[:H//2,:W//2]
                bpp_approx = self.get_bpp_subband_mfabien(approx, H//2, W//2, True)
                bpp += (1/(4**(l+1)))*bpp_approx
                
            H=H//2
            W=W//2
        
        return bpp
    
    def get_rate(self, dec_im, H, W):
        bpp = 0
        for l in range(self.levels):
            dh = dec_im[:H//2,W//2:W]
            dv = dec_im[H//2:H,:W//2]
            dd = dec_im[H//2:H,W//2:W]
            bpp_dh = self.get_bpp_subband(dh, H//2, W//2)
            bpp_dv = self.get_bpp_subband(dv, H//2, W//2)
            bpp_dd = self.get_bpp_subband(dd, H//2, W//2)
            
            bpp += (1/(4**(l+1)))*(bpp_dh+bpp_dv+bpp_dd)
            if l == self.levels-1:
                approx = dec_im[:H//2,:W//2]
                bpp_approx = self.get_bpp_subband(approx, H//2, W//2, True)
                bpp += (1/(4**(l+1)))*bpp_approx
                
            H=H//2
            W=W//2
        
        return bpp
    
            
    def get_distortion(self, image, quant_dec_im, H, W):
        rec_im = self.backwards(quant_dec_im)
        return k.mean(k.square(image - rec_im))
    
    def get_quant_dec_im(self, subbands, H, W):
        dec_im = self.subband_list_to_image(subbands)
        quant_noise_dec_im = tf.random.uniform([H,W],-self.args.quant_step/2,self.args.quant_step/2)
        return dec_im + quant_noise_dec_im
    
    def get_quant_per_subband(self, subbands):
        for i in range(len(subbands)):
            quant_noise_dec_im = tf.random.uniform([subbands[i].shape[0],subbands[i].shape[1]],-self.args.quant_step/2,self.args.quant_step/2)
            subbands[i] += quant_noise_dec_im
        return self.subband_list_to_image(subbands)
    
    def mse_rate_loss(self, image, subbands):
        H,W = image.shape
        
        if self.quant_image:
            quant_dec_im = self.get_quant_dec_im(subbands, H, W)
        else:
            quant_dec_im = self.get_quant_per_subband(subbands)
        
        mse = self.get_distortion(image, quant_dec_im, H, W)
        #norm_loss = self.norm_loss(subbands)
        rate_loss = self.quant_entropy_loss(subbands)
        #bpp = self.get_bpp_subband(quant_dec_im, H, W)
        #if self.balle18:
        #    bpp = self.get_rate_18(quant_dec_im, H, W)
            #bpp = self.get_bpp_subband_18(quant_dec_im, H, W)
        #else:
        #    bpp = self.get_rate(quant_dec_im, H, W)
        return mse, rate_loss#bpp, norm_loss
        
    
class U(tf.keras.Model):
    def __init__(self):
        super(U, self).__init__()
        self.model_U = self.fcn_layers(8)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, ref_U):
        pred_U = self.model_U(ref_U)
        return pred_U
    
class X3(tf.keras.Model):
    def __init__(self):
        super(X3, self).__init__()
        self.model_X3 = self.fcn_layers(40)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, ref_X3):
        pred_X3 = self.model_X3(ref_X3)
        return pred_X3
    
class X2(tf.keras.Model):
    def __init__(self):
        super(X2, self).__init__()
        self.model_X2 = self.fcn_layers(22)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, ref_X2):
        pred_X2 = self.model_X2(ref_X2)
        return pred_X2
    
class X1(tf.keras.Model):
    def __init__(self):
        super(X1, self).__init__()
        self.model_X1 = self.fcn_layers(14)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', input_dim = input_size))
        model.add(PReLU())
        model.add(Dense(64, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(32, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(16, kernel_initializer='normal'))
        model.add(PReLU())
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, ref_X1):
        pred_X1 = self.model_X1(ref_X1)
        return pred_X1
    
    
    
class weighted_model_tanh(tf.keras.Model):
    def __init__(self):
        super(weighted_model_tanh, self).__init__()
        self.model_X3 = self.fcn_layers(40)
        self.model_X2 = self.fcn_layers(22)
        self.model_X1 = self.fcn_layers(14)
        self.reference_P3_42_AG_tf_par_old = reference_P3_42_AG_tf_par_old
        self.reference_P2_42_2D_AG_tf_par = reference_P2_42_2D_AG_tf_par
        self.reference_P1_42_2D_AG_tf_par = reference_P1_42_2D_AG_tf_par
        self.target_P1_P2_tf_par = target_P1_P2_tf_par
        self.target_P3_tf_par = target_P3_tf_par
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', activation = 'tanh', input_dim = input_size))
        model.add(Dense(64, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(32, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(16, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, image):
        
        x0 = image[::2,::2]
        x1 = image[::2,1::2]
        x2 = image[1::2,::2]
        x3 = image[1::2,1::2]
        
        ref_P3 = self.reference_P3_42_AG_tf_par_old(x0, x1, x2)
        y_P3 = self.target_P3_tf_par(x3)
        
        pred_X3 = self.model_X3(ref_P3)
        
        x_dd = (y_P3 - pred_X3)
        x_dd = tf.reshape(x_dd, [x0.shape[0],x0.shape[1]])

        ref_P1 = self.reference_P1_42_2D_AG_tf_par(x0, x_dd)
        ref_P2 = self.reference_P2_42_2D_AG_tf_par(x0, x1, x_dd)

        y_P1, y_P2 = self.target_P1_P2_tf_par(x1, x2)

        pred_X1 = self.model_X1(ref_P1)
        pred_X2 = self.model_X2(ref_P2)
        
        return pred_X1, pred_X2, pred_X3, y_P1, y_P2, y_P3
    
class U_tanh(tf.keras.Model):
    def __init__(self):
        super(U_tanh, self).__init__()
        self.model_U = self.fcn_layers(8)
        
    def fcn_layers(self, input_size):
        model = Sequential()
        model.add(Dense(128, kernel_initializer='normal', activation = 'tanh', input_dim = input_size))
        model.add(Dense(64, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(32, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(16, kernel_initializer='normal', activation = 'tanh'))
        model.add(Dense(1, kernel_initializer='normal', activation='linear'))
        return model
            
        
    def call(self, ref_U):
        pred_U = self.model_U(ref_U)
        return pred_U

    
    
    
    
"""
subbands1 = subbands[:]       
subbands1[7] = subbands1[7] - subbands1[7]
subbands2 = subbands1[:]
subbands2[8] = subbands2[8] - subbands2[8]
subbands2[9] = subbands2[9] - subbands2[9]
subbands3 = subbands2[:]
subbands3[6] = subbands3[6] - subbands3[6]
subbands3[5] = subbands3[5] - subbands3[5]
subbands3[4] = subbands3[4] - subbands3[4]
subbands4 = subbands3[:]
subbands4[3] = subbands4[3] - subbands4[3]
subbands4[2] = subbands4[2] - subbands4[2]
subbands4[1] = subbands4[1] - subbands4[1]
"""