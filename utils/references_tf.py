import tensorflow as tf

import numpy as np
from scipy import signal
import sys

def reference_P3_42_AG_tf_par_old(x0, x1, x2):
    
    x0_ = tf.concat((tf.expand_dims(x0[:,2], axis = 1), x0), 1)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[:,-1], axis=1)), 1)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[:,-3], axis=1)), 1)
    x0_ = tf.concat((tf.expand_dims(x0_[2,:], axis = 0), x0_), 0)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[-1,:], axis = 0)),0)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[-3,:], axis = 0)),0)
    
    x1_ = tf.concat((tf.expand_dims(x1[:,1], axis=1), x1),1)
    x1_ = tf.concat((x1_, tf.expand_dims(x1[:,-2], axis=1)),1)
    x1_ = tf.concat((tf.expand_dims(x1_[2,:], axis = 0),x1_),0)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[-1,:], axis = 0)),0)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[-3,:], axis = 0)),0)
    
    x2_ = tf.concat((tf.expand_dims(x2[1,:], axis = 0),x2),0)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[-2,:], axis = 0)),0)
    x2_ = tf.concat((tf.expand_dims(x2_[:,2], axis = 1), x2_),1)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[:,-1], axis = 1)),1)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[:,-3], axis = 1)),1)
    
    x0_1 = tf.expand_dims(tf.reshape(x0_[0:-3,0:-3],[-1]), axis = 1)
    x0_2 = tf.expand_dims(tf.reshape(x0_[0:-3,1:-2],[-1]), axis = 1)
    x0_3 = tf.expand_dims(tf.reshape(x0_[0:-3,2:-1],[-1]), axis = 1)
    x0_4 = tf.expand_dims(tf.reshape(x0_[0:-3,3:],[-1]), axis = 1)
    x0_5 = tf.expand_dims(tf.reshape(x0_[1:-2,0:-3],[-1]), axis = 1)
    x0_6 = tf.expand_dims(tf.reshape(x0_[1:-2,1:-2],[-1]), axis = 1)
    x0_7 = tf.expand_dims(tf.reshape(x0_[1:-2,2:-1],[-1]), axis = 1)
    x0_8 = tf.expand_dims(tf.reshape(x0_[1:-2,3:],[-1]), axis = 1)
    x0_9 = tf.expand_dims(tf.reshape(x0_[2:-1,0:-3],[-1]), axis = 1)
    x0_10 = tf.expand_dims(tf.reshape(x0_[2:-1,1:-2],[-1]), axis = 1)
    x0_11 = tf.expand_dims(tf.reshape(x0_[2:-1,2:-1],[-1]), axis = 1)
    x0_12 = tf.expand_dims(tf.reshape(x0_[2:-1,3:],[-1]), axis = 1)
    x0_13 = tf.expand_dims(tf.reshape(x0_[3:,0:-3],[-1]), axis = 1)
    x0_14 = tf.expand_dims(tf.reshape(x0_[3:,1:-2],[-1]), axis = 1)
    x0_15 = tf.expand_dims(tf.reshape(x0_[3:,2:-1],[-1]), axis = 1)
    x0_16 = tf.expand_dims(tf.reshape(x0_[3:,3:],[-1]), axis = 1)
    
    x1_1 = tf.expand_dims(tf.reshape(x1_[0:-3,0:-2],[-1]), axis = 1)
    x1_2 = tf.expand_dims(tf.reshape(x1_[1:-2,0:-2],[-1]), axis = 1)
    x1_3 = tf.expand_dims(tf.reshape(x1_[2:-1,0:-2],[-1]), axis = 1)
    x1_4 = tf.expand_dims(tf.reshape(x1_[3:,0:-2],[-1]), axis = 1)
    x1_5 = tf.expand_dims(tf.reshape(x1_[0:-3,1:-1],[-1]), axis = 1)
    x1_6 = tf.expand_dims(tf.reshape(x1_[1:-2,1:-1],[-1]), axis = 1)
    x1_7 = tf.expand_dims(tf.reshape(x1_[2:-1,1:-1],[-1]), axis = 1)
    x1_8 = tf.expand_dims(tf.reshape(x1_[3:,1:-1],[-1]), axis = 1)
    x1_9 = tf.expand_dims(tf.reshape(x1_[0:-3,2:],[-1]), axis = 1)
    x1_10 = tf.expand_dims(tf.reshape(x1_[1:-2,2:],[-1]), axis = 1)
    x1_11 = tf.expand_dims(tf.reshape(x1_[2:-1,2:],[-1]), axis = 1)
    x1_12 = tf.expand_dims(tf.reshape(x1_[3:,2:],[-1]), axis = 1)
    
    x2_1 = tf.expand_dims(tf.reshape(x2_[0:-2,0:-3],[-1]), axis = 1)
    x2_2 = tf.expand_dims(tf.reshape(x2_[0:-2,1:-2],[-1]), axis = 1)
    x2_3 = tf.expand_dims(tf.reshape(x2_[0:-2,2:-1],[-1]), axis = 1)
    x2_4 = tf.expand_dims(tf.reshape(x2_[0:-2,3:],[-1]), axis = 1)
    x2_5 = tf.expand_dims(tf.reshape(x2_[1:-1,0:-3],[-1]), axis = 1)
    x2_6 = tf.expand_dims(tf.reshape(x2_[1:-1,1:-2],[-1]), axis = 1)
    x2_7 = tf.expand_dims(tf.reshape(x2_[1:-1,2:-1],[-1]), axis = 1)
    x2_8 = tf.expand_dims(tf.reshape(x2_[1:-1,3:],[-1]), axis = 1)
    x2_9 = tf.expand_dims(tf.reshape(x2_[2:,0:-3],[-1]), axis = 1)
    x2_10 = tf.expand_dims(tf.reshape(x2_[2:,1:-2],[-1]), axis = 1)
    x2_11 = tf.expand_dims(tf.reshape(x2_[2:,2:-1],[-1]), axis = 1)
    x2_12 = tf.expand_dims(tf.reshape(x2_[2:,3:],[-1]), axis = 1)
    
    ref_P3 = tf.concat((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x0_13, x0_14, x0_15, x0_16, x1_1, 
                        x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9, x1_10,
                        x1_11, x1_12, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, 
                        x2_8, x2_9, x2_10, x2_11, x2_12), axis = 1)
    return ref_P3

def reference_P3_42_AG_tf_par(x0, x1, x2):
    
    x0_ = tf.concat((tf.expand_dims(x0[:,2], axis = 1), x0), 1)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[:,-1], axis = 1)), 1)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[:,-3], axis = 1)), 1)
    x0_ = tf.concat((tf.expand_dims(x0_[2,:], axis = 0), x0_), 0)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[-1,:], axis = 0)),0)
    x0_ = tf.concat((x0_, tf.expand_dims(x0_[-3,:], axis = 0)),0)
    
    x1_ = tf.concat((tf.expand_dims(x1[:,1], axis = 1), x1),1)
    x1_ = tf.concat((x1_, tf.expand_dims(x1[:,-2], axis = 1)),1)
    x1_ = tf.concat((tf.expand_dims(x1_[2,:], axis = 0), x1_),0)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[-1,:], axis = 0)),0)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[-3,:], axis = 0)),0)
    
    x2_ = tf.concat((tf.expand_dims(x2[1,:], axis = 0), x2),0)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[-2,:], axis = 0)),0)
    x2_ = tf.concat((tf.expand_dims(x2_[:,2], axis = 1), x2_),1)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[:,-1], axis = 1)),1)
    x2_ = tf.concat((x2_, tf.expand_dims(x2_[:,-3], axis = 1)),1)
    
    x0_1 = tf.expand_dims(tf.reshape(x0_[0:-3,0:-3],[-1]), axis = 1)
    x0_2 = tf.expand_dims(tf.reshape(x0_[0:-3,1:-2],[-1]), axis = 1)
    x0_3 = tf.expand_dims(tf.reshape(x0_[0:-3,2:-1],[-1]), axis = 1)
    x0_4 = tf.expand_dims(tf.reshape(x0_[0:-3,3:],[-1]), axis = 1)
    x0_5 = tf.expand_dims(tf.reshape(x0_[1:-2,0:-3],[-1]), axis = 1)
    x0_6 = tf.expand_dims(tf.reshape(x0_[1:-2,1:-2],[-1]), axis = 1)
    x0_7 = tf.expand_dims(tf.reshape(x0_[1:-2,2:-1],[-1]), axis = 1)
    x0_8 = tf.expand_dims(tf.reshape(x0_[1:-2,3:],[-1]), axis = 1)
    x0_9 = tf.expand_dims(tf.reshape(x0_[2:-1,0:-3],[-1]), axis = 1)
    x0_10 = tf.expand_dims(tf.reshape(x0_[2:-1,1:-2],[-1]), axis = 1)
    x0_11 = tf.expand_dims(tf.reshape(x0_[2:-1,2:-1],[-1]), axis = 1)
    x0_12 = tf.expand_dims(tf.reshape(x0_[2:-1,3:],[-1]), axis = 1)
    x0_13 = tf.expand_dims(tf.reshape(x0_[3:,0:-3],[-1]), axis = 1)
    x0_14 = tf.expand_dims(tf.reshape(x0_[3:,1:-2],[-1]), axis = 1)
    x0_15 = tf.expand_dims(tf.reshape(x0_[3:,2:-1],[-1]), axis = 1)
    x0_16 = tf.expand_dims(tf.reshape(x0_[3:,3:],[-1]), axis = 1)
    
    x1_1 = tf.expand_dims(tf.reshape(x1_[0:-3,0:-2],[-1]), axis = 1)
    x1_2 = tf.expand_dims(tf.reshape(x1_[0:-3,1:-1],[-1]), axis = 1)
    x1_3 = tf.expand_dims(tf.reshape(x1_[0:-3,2:],[-1]), axis = 1)
    x1_4 = tf.expand_dims(tf.reshape(x1_[1:-2,0:-2],[-1]), axis = 1)
    x1_5 = tf.expand_dims(tf.reshape(x1_[1:-2,1:-1],[-1]), axis = 1)
    x1_6 = tf.expand_dims(tf.reshape(x1_[1:-2,2:],[-1]), axis = 1)
    x1_7 = tf.expand_dims(tf.reshape(x1_[2:-1,0:-2],[-1]), axis = 1)
    x1_8 = tf.expand_dims(tf.reshape(x1_[2:-1,1:-1],[-1]), axis = 1)
    x1_9 = tf.expand_dims(tf.reshape(x1_[2:-1,2:],[-1]), axis = 1)
    x1_10 = tf.expand_dims(tf.reshape(x1_[3:,0:-2],[-1]), axis = 1)
    x1_11 = tf.expand_dims(tf.reshape(x1_[3:,1:-1],[-1]), axis = 1)
    x1_12 = tf.expand_dims(tf.reshape(x1_[3:,2:],[-1]), axis = 1)
    
    x2_1 = tf.expand_dims(tf.reshape(x2_[0:-2,0:-3],[-1]), axis = 1)
    x2_2 = tf.expand_dims(tf.reshape(x2_[0:-2,1:-2],[-1]), axis = 1)
    x2_3 = tf.expand_dims(tf.reshape(x2_[0:-2,2:-1],[-1]), axis = 1)
    x2_4 = tf.expand_dims(tf.reshape(x2_[0:-2,3:],[-1]), axis = 1)
    x2_5 = tf.expand_dims(tf.reshape(x2_[1:-1,0:-3],[-1]), axis = 1)
    x2_6 = tf.expand_dims(tf.reshape(x2_[1:-1,1:-2],[-1]), axis = 1)
    x2_7 = tf.expand_dims(tf.reshape(x2_[1:-1,2:-1],[-1]), axis = 1)
    x2_8 = tf.expand_dims(tf.reshape(x2_[1:-1,3:],[-1]), axis = 1)
    x2_9 = tf.expand_dims(tf.reshape(x2_[2:,0:-3],[-1]), axis = 1)
    x2_10 = tf.expand_dims(tf.reshape(x2_[2:,1:-2],[-1]), axis = 1)
    x2_11 = tf.expand_dims(tf.reshape(x2_[2:,2:-1],[-1]), axis = 1)
    x2_12 = tf.expand_dims(tf.reshape(x2_[2:,3:],[-1]), axis = 1)
    
    ref_P3 = tf.concat((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x0_13, x0_14, x0_15, x0_16, x1_1, 
                        x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9, x1_10,
                        x1_11, x1_12, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, 
                        x2_8, x2_9, x2_10, x2_11, x2_12),1)
    return ref_P3
    

def reference_P2_42_2D_AG_tf_par(x0, x1, x_dd):
    
    x02 = tf.concat((tf.expand_dims(x0[2,:], axis = 0), x0), 0)
    x02 = tf.concat((x02, tf.expand_dims(x02[-1,:], axis = 0)),0)
    x02 = tf.concat((x02, tf.expand_dims(x02[-3,:], axis = 0)),0)
    x02 = tf.concat((tf.expand_dims(x02[:,1], axis = 1), x02),1)
    x02 = tf.concat((x02, tf.expand_dims(x02[:,-2], axis = 1)),1)
    x0_ = x02
    x1_ = tf.concat((tf.expand_dims(x1[:,0], axis = 1), x1),1)
    x1_ = tf.concat((tf.expand_dims(x1_[:,2], axis = 1), x1_),1)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[:,-3], axis = 1)),1)
    x1_ = tf.concat((x1_, tf.expand_dims(x1_[-1,:], axis = 0)),0)
    
    x_dd2 = tf.concat((tf.expand_dims(x_dd[:,0], axis = 1), x_dd),1)
    
    x0_1 = tf.expand_dims(tf.reshape(x0_[0:-3,0:-2],[-1]), axis = 1)
    x0_2 = tf.expand_dims(tf.reshape(x0_[0:-3,1:-1],[-1]), axis = 1)
    x0_3 = tf.expand_dims(tf.reshape(x0_[0:-3,2:],[-1]), axis = 1)
    x0_4 = tf.expand_dims(tf.reshape(x0_[1:-2,0:-2],[-1]), axis = 1)
    x0_5 = tf.expand_dims(tf.reshape(x0_[1:-2,1:-1],[-1]), axis = 1)
    x0_6 = tf.expand_dims(tf.reshape(x0_[1:-2,2:],[-1]), axis = 1)
    x0_7 = tf.expand_dims(tf.reshape(x0_[2:-1,0:-2],[-1]), axis = 1)
    x0_8 = tf.expand_dims(tf.reshape(x0_[2:-1,1:-1],[-1]), axis = 1)
    x0_9 = tf.expand_dims(tf.reshape(x0_[2:-1,2:],[-1]), axis = 1)
    x0_10 = tf.expand_dims(tf.reshape(x0_[3:,0:-2],[-1]), axis = 1)
    x0_11 = tf.expand_dims(tf.reshape(x0_[3:,1:-1],[-1]), axis = 1)
    x0_12 = tf.expand_dims(tf.reshape(x0_[3:,2:],[-1]), axis = 1)
    
    x1_1 = tf.expand_dims(tf.reshape(x1_[0:-1,0:-3],[-1]), axis = 1)
    x1_2 = tf.expand_dims(tf.reshape(x1_[0:-1,1:-2],[-1]), axis = 1)
    x1_3 = tf.expand_dims(tf.reshape(x1_[0:-1,2:-1],[-1]), axis = 1)
    x1_4 = tf.expand_dims(tf.reshape(x1_[0:-1,3:],[-1]), axis = 1)
    x1_5 = tf.expand_dims(tf.reshape(x1_[1:,0:-3],[-1]), axis = 1)
    x1_6 = tf.expand_dims(tf.reshape(x1_[1:,1:-2],[-1]), axis = 1)
    x1_7 = tf.expand_dims(tf.reshape(x1_[1:,2:-1],[-1]), axis = 1)
    x1_8 = tf.expand_dims(tf.reshape(x1_[1:,3:],[-1]), axis = 1)
    
    x_dd2_1 = tf.expand_dims(tf.reshape(x_dd2[:,0:-1],[-1]), axis = 1)
    x_dd2_2 = tf.expand_dims(tf.reshape(x_dd2[:,1:],[-1]), axis = 1)
    
    ref_P2 = tf.concat((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, 
                        x1_7, x1_8, x_dd2_2, x_dd2_1), axis = 1)
    
    return ref_P2

def reference_P1_42_2D_AG_tf_par(x0, x_dd):
    
    x01 = tf.concat((tf.expand_dims(x0[:,2], axis = 1), x0), 1)
    x01 = tf.concat((x01, tf.expand_dims(x01[:,-1], axis = 1)), 1)
    x01 = tf.concat((x01, tf.expand_dims(x01[:,-3], axis = 1)), 1) 
    x01 = tf.concat((tf.expand_dims(x01[1,:], axis = 0), x01),0)
    x01 = tf.concat((x01, tf.expand_dims(x01[-2,:], axis = 0)),0)
    x0_ = x01
    x_dd1 = tf.concat((tf.expand_dims(x_dd[0,:], axis = 0), x_dd),0)
    
    x0_1 = tf.expand_dims(tf.reshape(x0_[0:-2,0:-3], [-1]), axis = 1)
    x0_2 = tf.expand_dims(tf.reshape(x0_[0:-2,1:-2], [-1]), axis = 1)
    x0_3 = tf.expand_dims(tf.reshape(x0_[0:-2,2:-1], [-1]), axis = 1)
    x0_4 = tf.expand_dims(tf.reshape(x0_[0:-2,3:], [-1]), axis = 1)
    x0_5 = tf.expand_dims(tf.reshape(x0_[1:-1,0:-3], [-1]), axis = 1)
    x0_6 = tf.expand_dims(tf.reshape(x0_[1:-1,1:-2], [-1]), axis = 1)
    x0_7 = tf.expand_dims(tf.reshape(x0_[1:-1,2:-1], [-1]), axis = 1)
    x0_8 = tf.expand_dims(tf.reshape(x0_[1:-1,3:], [-1]), axis = 1)
    x0_9 = tf.expand_dims(tf.reshape(x0_[2:,0:-3], [-1]), axis = 1)
    x0_10 = tf.expand_dims(tf.reshape(x0_[2:,1:-2], [-1]), axis = 1)
    x0_11 = tf.expand_dims(tf.reshape(x0_[2:,2:-1], [-1]), axis = 1)
    x0_12 = tf.expand_dims(tf.reshape(x0_[2:,3:], [-1]), axis = 1)
    
    x_dd1_1 = tf.expand_dims(tf.reshape(x_dd1[0:-1,:], [-1]), axis = 1)
    x_dd1_2 = tf.expand_dims(tf.reshape(x_dd1[1:,:], [-1]), axis = 1)
    
    ref_P1 = tf.concat((x0_1, x0_2, x0_3, x0_4, x0_5, x0_6, x0_7, x0_8, x0_9,
                        x0_10, x0_11, x0_12, x_dd1_2, x_dd1_1),1)
    return ref_P1

def target_P3_tf_par(x3):
    
    y_P3 = tf.expand_dims(tf.reshape(x3, [-1]), axis = 1)
    return y_P3

def target_P1_P2_tf_par(x1,x2):
    
    y_P1 = tf.expand_dims(tf.reshape(x1, [-1]), axis = 1)
    y_P2 = tf.expand_dims(tf.reshape(x2, [-1]), axis = 1)
    return y_P1, y_P2

def convolve(x, kernel):
    """convolve image and kernel.

    Parameters
    ----------
    x : tf tensor image.
    kernel : tf tensor with odd hight and width.

    Returns
    -------
    tf tensor
    convolution output.

    """
    
    h, w = kernel.shape
    assert h % 2 == 1 and w % 2 ==1
    x_h, x_w = x.shape 
    x = tf.pad(x, [[(h-1)//2,(h-1)//2],[(w-1)//2,(w-1)//2]], "SYMMETRIC")
    concat = tf.concat([tf.concat(
        [tf.expand_dims(tf.reshape(x[i:x.shape[0]-(h-(i+1)),j:x.shape[1]-(w-(j+1))], [-1]), axis = 1)\
                          for j in range(w)],1) for i in range(h)],1)
    kernel = tf.expand_dims(tf.reshape(kernel, [-1]), axis = 0)
    convolved = tf.expand_dims(tf.reduce_sum(concat * kernel, axis = 1), axis = 1)
    
    return tf.reshape(convolved, [x_h, x_w])

def sparse_convolve(x, kernel):
    """convolve image and kernel.

    Parameters
    ----------
    x : tf tensor image.
    kernel : tf tensor with odd hight and width.

    Returns
    -------
    tf tensor
    convolution output.

    """
    
    h, w = kernel.shape
    assert h % 2 == 1 and w % 2 ==1
    x_h, x_w = x.shape 
    x = tf.pad(x, [[(h-1)//2,(h-1)//2],[(w-1)//2,(w-1)//2]], "SYMMETRIC")
    concat = tf.concat([tf.concat(
        [tf.expand_dims(tf.reshape(x[i:x.shape[0]-(h-(i+1)):2,j:x.shape[1]-(w-(j+1)):2], [-1]), axis = 1)\
                          for j in range(w)],1) for i in range(h)],1)
    kernel = tf.expand_dims(tf.reshape(kernel, [-1]), axis = 0)
    convolved = tf.expand_dims(tf.reduce_sum(concat * kernel, axis = 1), axis = 1)
    
    return tf.reshape(convolved, [x_h//2, x_w//2])
    
    
def target_U_tf_par_v2(image, x0, h_2d):

    y_tild = sparse_convolve(image, h_2d)
    y_U = tf.expand_dims(tf.reshape(y_tild, [-1]), axis = 1)-tf.expand_dims(tf.reshape(x0, [-1]), axis = 1)
    
    return y_U

def ideal_approx_2(image, h_2d):
    ideal_approx1 = sparse_convolve(image, h_2d)
    ideal_approx2 = sparse_convolve(ideal_approx1, h_2d)
    return ideal_approx2

def target_U_tf_par(image, x0):
               
    N = 15
    t = np.arange(-N,N+1)
    h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
    h_2d = h*np.transpose(h)
    h_2d = h_2d.astype(image.numpy().dtype)
    y_tild_ = signal.convolve2d(image.numpy(), h_2d , boundary='symm', mode='same')
    
    y_tild = tf.convert_to_tensor(y_tild_[::2,::2])
    
    y_U = tf.expand_dims(tf.reshape(y_tild, [-1]), axis = 1)-tf.expand_dims(tf.reshape(x0, [-1]), axis = 1)
    
    return y_U

def reference_U_tf_par(x_dd, x_dv, x_dh):
    
    x_dd1 = tf.concat((tf.expand_dims(x_dd[0,:], axis = 0), x_dd), 0)
    x_dd1 = tf.concat((tf.expand_dims(x_dd1[:,0], axis = 1), x_dd1), 1)
    x_dh1 = tf.concat((tf.expand_dims(x_dh[:,0], axis = 1), x_dh), 1)
    x_dv1 = tf.concat((tf.expand_dims(x_dv[0,:], axis = 0), x_dv), 0)
    
    x_dd_1 = tf.expand_dims(tf.reshape(x_dd1[:-1,:-1], [-1]), axis = 1)
    x_dd_2 = tf.expand_dims(tf.reshape(x_dd1[:-1,1:], [-1]), axis = 1)
    x_dd_3 = tf.expand_dims(tf.reshape(x_dd1[1:,:-1], [-1]), axis = 1)
    x_dd_4 = tf.expand_dims(tf.reshape(x_dd1[1:,1:], [-1]), axis = 1)
    
    x_dh_1 = tf.expand_dims(tf.reshape(x_dh1[0:,0:-1], [-1]), axis = 1)
    x_dh_2 = tf.expand_dims(tf.reshape(x_dh1[0:,1:], [-1]), axis = 1)
    
    x_dv_1 = tf.expand_dims(tf.reshape(x_dv1[0:-1,0:], [-1]), axis = 1)
    x_dv_2 = tf.expand_dims(tf.reshape(x_dv1[1:,:], [-1]), axis = 1)
 
    ref_U = tf.concat((x_dh_2, x_dh_1, x_dv_2, x_dv_1, x_dd_4, x_dd_3, x_dd_2, x_dd_1), axis = 1)
    
    return ref_U

    
if __name__ == '__main__':

    image_np = np.arange(32*32).reshape(32,32)
    x_dd_np = np.arange(16*16).reshape(16,16)
    x_dv_np = np.arange(16*16).reshape(16,16)
    x_dh_np = np.arange(16*16).reshape(16,16)

    x0_np = image_np[::2,::2]
    x1_np = image_np[::2,1::2]
    x2_np = image_np[1::2,::2]
    x3_np = image_np[1::2,1::2]
    
    x_dd = tf.convert_to_tensor(x_dd_np, dtype = tf.float64)
    x_dv = tf.convert_to_tensor(x_dv_np, dtype = tf.float64)
    x_dh = tf.convert_to_tensor(x_dh_np, dtype = tf.float64)
    image = tf.convert_to_tensor(image_np, dtype = tf.float64)
    
    x0 = image[::2,::2]
    x1 = image[::2,1::2]
    x2 = image[1::2,::2]
    x3 = image[1::2,1::2]
    
    N = 15
    t = np.arange(-N,N+1)
    h = np.expand_dims(0.5*np.sinc(t/2),axis = 1)
    h_2d = h*np.transpose(h)
    h_2d = h_2d.astype(image.numpy().dtype)
    h_2d = tf.convert_to_tensor(h_2d)
    y_U = target_U_tf_par_v2(image, x0, h_2d)
    y_U_np = target_U(image_np, x0_np)
    
    #print('y_U', (y_U_np == y_U.numpy()).all())
    #print(np.sum(np.abs(y_U_np-y_U.numpy())**2))
    #print(y_U, y_U_np)

    #ref_P2 = reference_P2_42_2D_AG(x0_np, x1_np, x_dd_np)
    #ref_P1 = reference_P1_42_2D_AG(x0_np, x_dd_np)
    #y_P3 = target_P3(x3_np)
    #y_P1, y_P2 = target_P1_P2(x1_np,x2_np)

    
    
    if False:
        ref_P3_par = reference_P3_42_AG_tf_par_old(x0, x1, x2)
        ref_P2_par = reference_P2_42_2D_AG_tf_par(x0, x1, x_dd)
        ref_P1_par = reference_P1_42_2D_AG_tf_par(x0, x_dd)
        y_P3_par = target_P3_tf_par(x3)
        y_P1_par, y_P2_par = target_P1_P2_tf_par(x1,x2)

        ref_P3_pt = reference_P3_42_AG(x0_np, x1_np, x2_np)
        ref_P2_pt = reference_P2_42_2D_AG(x0_np, x1_np, x_dd_np)
        ref_P1_pt = reference_P1_42_2D_AG(x0_np, x_dd_np)
        y_P3_pt = target_P3(x3_np)
        y_P1_pt, y_P2_pt = target_P1_P2(x1_np, x2_np)

        #ref_U_np = reference_U(x_dd_np, x_dv_np, x_dh_np)
        #ref_U = reference_U_pt_par(x_dd, x_dv, x_dh)
        #print('ref_U', (ref_U_np==ref_Unumpy()).all())
        #print(np.sum(np.abs(ref_U_np-ref_Unumpy())))

        print('ref_P3', (ref_P3_pt==ref_P3_par.numpy()).all())
        print(np.sum(np.abs(ref_P3_pt-ref_P3_par.numpy())))

        print('ref_P2', (ref_P2_pt==ref_P2_par.numpy()).all())
        print(np.sum(np.abs(ref_P2_pt-ref_P2_par.numpy())))
        print(ref_P2_pt-ref_P2_par.numpy())

        print('ref_P1', (ref_P1_pt==ref_P1_par.numpy()).all())
        print(np.sum(np.abs(ref_P1_pt-ref_P1_par.numpy())))
        print(ref_P1_pt-ref_P1_par.numpy())

        print('y_P3', (y_P3_pt==y_P3_par.numpy()).all())
        print(np.sum(np.abs(y_P3_pt-y_P3_par.numpy())))

        print('y_P2', (y_P2_pt==y_P2_par.numpy()).all())
        print(np.sum(np.abs(y_P2_pt-y_P2_par.numpy())))

        print('y_P1', (y_P1_pt==y_P1_par.numpy()).all())
        print(np.sum(np.abs(y_P1_pt-y_P1_par.numpy())))


