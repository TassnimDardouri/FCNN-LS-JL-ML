import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model

import numpy as np
import time
from scipy.io import loadmat

import sys 

from os.path import abspath, dirname

sys.path.append(dirname(abspath(__file__)))

from utils.data_loader import full_loader
from utils.train_parser import Options
from utils.helpers import (get_args, create_directory, get_loss_weights, get_betas)
from utils.losses import quant_entropy, Log_L_beta, L_beta, L2

from models.subband_fcn_tf import multires_weighted_model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

args = get_args()
opt = Options(args)

train_dataset = full_loader(opt, opt.train_data_dir)
val_dataset = full_loader(opt, opt.val_data_dir)

betas = get_betas(args.loss)
weights = get_loss_weights(args.loss, w_type = args.weight_type, dataset = args.dataset)
weights = weights/np.asarray([4,4,4,16,16,16,64,64,64,64])
weights_CW = get_loss_weights(args.loss, w_type = 'compute_weight_sqrt', dataset = args.dataset)
print(weights, betas)

model = multires_weighted_model(weights, weights_CW, betas, args)
"""
if args.loss == 'L1':
    method = 'fcn_L1_42_AG_16n'
elif args.loss == 'L_beta':
    method = 'fcn_L_const_beta_42_AG_16n'
    beta_train = loadmat('/data/tasnim/alpha_beta/beta_train_fcn_L1.mat')
    beta_train = beta_train['Beta']
    alpha_train = loadmat('/data/tasnim/alpha_beta/alpha_clic_train_f(beta_i).mat')
    alpha_train = alpha_train['alpha']
elif args.loss == 'L2':
    method = 'fcn_42_AG_16n'
    #alpha_train = loadmat('/data/tasnim/alpha_beta/alpha_clic_train_f(2_i).mat')
    #alpha_train = alpha_train['alpha']
#method = 'fcn_sum_log_L2_adapt_ma_164_global_tf'
#method = 'fcn_sum_log_L_beta_joint_UP_ma_164_global_tf'
L1_path = '/data/tasnim/weights/models_16n/'+method
model.model_X3_1 = load_model(L1_path+'/level1_X3_'+method+'.h5')
model.model_X3_2 = load_model(L1_path+'/level2_X3_'+method+'.h5')
model.model_X3_3 = load_model(L1_path+'/level3_X3_'+method+'.h5')
model.model_X2_1 = load_model(L1_path+'/level1_X2_'+method+'.h5')
model.model_X2_2 = load_model(L1_path+'/level2_X2_'+method+'.h5')
model.model_X2_3 = load_model(L1_path+'/level3_X2_'+method+'.h5')
model.model_X1_1 = load_model(L1_path+'/level1_X1_'+method+'.h5')
model.model_X1_2 = load_model(L1_path+'/level2_X1_'+method+'.h5')
model.model_X1_3 = load_model(L1_path+'/level3_X1_'+method+'.h5')
model.model_U_1 = load_model(L1_path+'/level1_U_'+method+'.h5')
model.model_U_2 = load_model(L1_path+'/level2_U_'+method+'.h5')
model.model_U_3 = load_model(L1_path+'/level3_U_'+method+'.h5')
"""
create_directory(opt.log_path)
create_directory(opt.save_models_path)
create_directory(opt.save_reference_path)




optimizer = keras.optimizers.Adam(learning_rate = args.lr, decay = args.decay)

best_loss = 10000
train_loss = np.zeros((opt.epochs,1))
val_loss = np.zeros((opt.epochs,1))
train_loss_U = np.zeros((opt.epochs,1))

for epoch in range(opt.epochs):
    print("\nStart of epoch %d" % (epoch,))
    begin = time.time()
    train_loss_sum = 0
    # Iterate over the batches of the dataset.
    batch_train = 0
    for step, (image) in enumerate(train_dataset): 
        if args.beta_adapt:
            if args.loss == 'L_beta':
                betas = beta_train[step,:]
            weights = ((1/alpha_train[step,:])**betas)
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables auto-differentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.

            approx, pred_X1_3, pred_X2_3, pred_X3_3, U_pred_3, y_P1_3, y_P2_3, y_P3_3, y_U_3, ref_U_3,\
            pred_X1_2, pred_X2_2, pred_X3_2, U_pred_2, y_P1_2, y_P2_2, y_P3_2, y_U_2, ref_U_2,\
            pred_X1_1, pred_X2_1, pred_X3_1, U_pred_1, y_P1_1, y_P2_1, y_P3_1, y_U_1, ref_U_1 = model(image, training=True)
            
            # Compute the loss value for this minibatch.
            
            if args.sum_log_L_beta:
                
                loss_value = weights[0]*L_beta(pred_X1_1, y_P1_1, betas[0])+\
                               1.43*Log_L_beta(pred_X1_1, y_P1_1, betas[0], weights_CW[0])/4+\
                               weights[1]*L_beta(pred_X2_1, y_P2_1, betas[1])+\
                               1.43*Log_L_beta(pred_X2_1, y_P2_1, betas[1], weights_CW[1])/4+\
                               weights[2]*L_beta(pred_X3_1, y_P3_1, betas[2])+\
                               1.43*Log_L_beta(pred_X3_1, y_P3_1, betas[2], weights_CW[2])/4+\
                               weights[3]*L_beta(pred_X1_2, y_P1_2, betas[3])+\
                               (1.43*Log_L_beta(pred_X1_2, y_P1_2, betas[3], weights_CW[3]))/16+\
                               weights[4]*L_beta(pred_X2_2, y_P2_2, betas[4])+\
                               (1.43*Log_L_beta(pred_X2_2, y_P2_2, betas[4], weights_CW[4]))/16+\
                               weights[5]*L_beta(pred_X3_2, y_P3_2, betas[5])+\
                               (1.43*Log_L_beta(pred_X3_2, y_P3_2, betas[5], weights_CW[5]))/16+\
                               weights[6]*L_beta(pred_X1_3, y_P1_3, betas[6])+\
                               (1.43*Log_L_beta(pred_X1_3, y_P1_3, betas[6], weights_CW[6]))/64+\
                               weights[7]*L_beta(pred_X2_3, y_P2_3, betas[7])+\
                               (1.43*Log_L_beta(pred_X2_3, y_P2_3, betas[7], weights_CW[7]))/64+\
                               weights[8]*L_beta(pred_X3_3, y_P3_3, betas[8])+\
                               (1.43*Log_L_beta(pred_X3_3, y_P3_3, betas[8], weights_CW[8]))/64+\
                               (args.u_weights*L2(U_pred_3, y_U_3))/64#+\
                               # tf.cast(weights[9]*L_beta(tf.cast(approx, tf.float64),
                               # tf.zeros([approx.shape[0],approx.shape[1]], tf.float64), 2)+\
                               # (1.43*Log_L_beta(tf.cast(approx, tf.float64),
                               # tf.zeros([approx.shape[0],approx.shape[1]], tf.float64), 
                               # 2, weights_CW[9]))/16, tf.float32)
                """
                loss_value = (quant_entropy(pred_X1_1 - y_P1_1, betas[2]))/4+\
                             (quant_entropy(pred_X2_1 - y_P2_1, betas[1]))/4+\
                             (quant_entropy(pred_X3_1 - y_P3_1, betas[0]))/4+\
                             (quant_entropy(pred_X1_2 - y_P1_2, betas[5]))/16+\
                             (quant_entropy(pred_X2_2 - y_P2_2, betas[4]))/16+\
                             (quant_entropy(pred_X3_2 - y_P3_2, betas[3]))/16+\
                             (quant_entropy(pred_X1_3 - y_P1_3, betas[8]))/64+\
                             (quant_entropy(pred_X2_3 - y_P2_3, betas[7]))/64+\
                             (quant_entropy(pred_X3_3 - y_P3_3, betas[6]))/64+\
                             (quant_entropy(U_pred_3 - y_U_3, 2))/64+\
                             args.u_weights*L2(U_pred_3, y_U_3)/64
                """
            elif args.use_log:
                loss_value = Log_L_beta(pred_X1_1, y_P1_1, betas[0],1)+\
                               Log_L_beta(pred_X2_1, y_P2_1, betas[1],1)+\
                               Log_L_beta(pred_X3_1, y_P3_1, betas[2],1)+\
                               Log_L_beta(pred_X1_2, y_P1_2, betas[3],1)/4+\
                               Log_L_beta(pred_X2_2, y_P2_2, betas[4],1)/4+\
                               Log_L_beta(pred_X3_2, y_P3_2, betas[5],1)/4+\
                               Log_L_beta(pred_X1_3, y_P1_3, betas[6],1)/16+\
                               Log_L_beta(pred_X2_3, y_P2_3, betas[7],1)/16+\
                               Log_L_beta(pred_X3_3, y_P3_3, betas[8],1)/16
            else:
                loss_value = (weights[0])*L_beta(pred_X1_1, y_P1_1, betas[0])+\
                               (weights[1])*L_beta(pred_X2_1, y_P2_1, betas[1])+\
                               (weights[2])*L_beta(pred_X3_1, y_P3_1, betas[2])+\
                               (weights[3])*L_beta(pred_X1_2, y_P1_2, betas[3])+\
                               (weights[4])*L_beta(pred_X2_2, y_P2_2, betas[4])+\
                               (weights[5])*L_beta(pred_X3_2, y_P3_2, betas[5])+\
                               (weights[6])*L_beta(pred_X1_3, y_P1_3, betas[6])+\
                               (weights[7])*L_beta(pred_X2_3, y_P2_3, betas[7])+\
                               (weights[8])*L_beta(pred_X3_3, y_P3_3, betas[8])

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        
        train_loss_sum += loss_value.numpy()

        batch_train+=1
        sys.stdout.write('\r train image '+str(batch_train)+' loss '+ str(loss_value.numpy()))
        sys.stdout.flush()
        
    val_loss_sum = 0
    batch_test = 0
    for step, (image) in enumerate(val_dataset):
        
        approx, pred_X1_3, pred_X2_3, pred_X3_3, U_pred_3, y_P1_3, y_P2_3, y_P3_3, y_U_3, ref_U_3,\
        pred_X1_2, pred_X2_2, pred_X3_2, U_pred_2, y_P1_2, y_P2_2, y_P3_2, y_U_2, ref_U_2,\
        pred_X1_1, pred_X2_1, pred_X3_1, U_pred_1, y_P1_1, y_P2_1, y_P3_1, y_U_1, ref_U_1 = model(image, training=True)

        # Compute the loss value for this minibatch.
        if args.sum_log_L_beta:
            
            val_loss_value = weights[0]*L_beta(pred_X1_1, y_P1_1, betas[0])+\
                               1.43*Log_L_beta(pred_X1_1, y_P1_1, betas[0], weights_CW[0])+\
                               weights[1]*L_beta(pred_X2_1, y_P2_1, betas[1])+\
                               1.43*Log_L_beta(pred_X2_1, y_P2_1, betas[1], weights_CW[1])+\
                               weights[2]*L_beta(pred_X3_1, y_P3_1, betas[2])+\
                               1.43*Log_L_beta(pred_X3_1, y_P3_1, betas[2], weights_CW[2])+\
                               weights[3]*L_beta(pred_X1_2, y_P1_2, betas[3])+\
                               (1.43*Log_L_beta(pred_X1_2, y_P1_2, betas[3], weights_CW[3]))/4+\
                               weights[4]*L_beta(pred_X2_2, y_P2_2, betas[4])+\
                               (1.43*Log_L_beta(pred_X2_2, y_P2_2, betas[4], weights_CW[4]))/4+\
                               weights[5]*L_beta(pred_X3_2, y_P3_2, betas[5])+\
                               (1.43*Log_L_beta(pred_X3_2, y_P3_2, betas[5], weights_CW[5]))/4+\
                               weights[6]*L_beta(pred_X1_3, y_P1_3, betas[6])+\
                               (1.43*Log_L_beta(pred_X1_3, y_P1_3, betas[6], weights_CW[6]))/16+\
                               weights[7]*L_beta(pred_X2_3, y_P2_3, betas[7])+\
                               (1.43*Log_L_beta(pred_X2_3, y_P2_3, betas[7], weights_CW[7]))/16+\
                               weights[8]*L_beta(pred_X3_3, y_P3_3, betas[8])+\
                               (1.43*Log_L_beta(pred_X3_3, y_P3_3, betas[8], weights_CW[8]))/16+\
                               (args.u_weights*L2(U_pred_3, y_U_3))/16
            """
            val_loss_value = (quant_entropy(pred_X1_1 - y_P1_1, betas[2]))/4+\
                             (quant_entropy(pred_X2_1 - y_P2_1, betas[1]))/4+\
                             (quant_entropy(pred_X3_1 - y_P3_1, betas[0]))/4+\
                             (quant_entropy(pred_X1_2 - y_P1_2, betas[5]))/16+\
                             (quant_entropy(pred_X2_2 - y_P2_2, betas[4]))/16+\
                             (quant_entropy(pred_X3_2 - y_P3_2, betas[3]))/16+\
                             (quant_entropy(pred_X1_3 - y_P1_3, betas[8]))/64+\
                             (quant_entropy(pred_X2_3 - y_P2_3, betas[7]))/64+\
                             (quant_entropy(pred_X3_3 - y_P3_3, betas[6]))/64+\
                             (quant_entropy(U_pred_3 - y_U_3, 2))/64+\
                             args.u_weights*L2(U_pred_3, y_U_3)/64
            """
        elif args.use_log:
            val_loss_value = Log_L_beta(pred_X1_1, y_P1_1, betas[0],1)+\
                               Log_L_beta(pred_X2_1, y_P2_1, betas[1],1)+\
                               Log_L_beta(pred_X3_1, y_P3_1, betas[2],1)+\
                               Log_L_beta(pred_X1_2, y_P1_2, betas[3],1)+\
                               Log_L_beta(pred_X2_2, y_P2_2, betas[4],1)+\
                               Log_L_beta(pred_X3_2, y_P3_2, betas[5],1)+\
                               Log_L_beta(pred_X1_3, y_P1_3, betas[6],1)+\
                               Log_L_beta(pred_X2_3, y_P2_3, betas[7],1)+\
                               Log_L_beta(pred_X3_3, y_P3_3, betas[8],1)
        else:
            val_loss_value = (weights[0])*L_beta(pred_X1_1, y_P1_1, betas[0])+\
                               (weights[1])*L_beta(pred_X2_1, y_P2_1, betas[1])+\
                               (weights[2])*L_beta(pred_X3_1, y_P3_1, betas[2])+\
                               (weights[3])*L_beta(pred_X1_2, y_P1_2, betas[3])+\
                               (weights[4])*L_beta(pred_X2_2, y_P2_2, betas[4])+\
                               (weights[5])*L_beta(pred_X3_2, y_P3_2, betas[5])+\
                               (weights[6])*L_beta(pred_X1_3, y_P1_3, betas[6])+\
                               (weights[7])*L_beta(pred_X2_3, y_P2_3, betas[7])+\
                               (weights[8])*L_beta(pred_X3_3, y_P3_3, betas[8])


        val_loss_sum += val_loss_value.numpy()
        batch_test+=1
        sys.stdout.write('\r test image '+str(batch_test)+' loss P '+ str(val_loss_value.numpy()))
        sys.stdout.flush()
        
    train_loss_epoch = train_loss_sum/batch_train
    val_loss_epoch = val_loss_sum/batch_test
    
    print('\n Epoch: ', epoch,'train loss: ', train_loss_epoch,'test loss: ', val_loss_epoch)
    
    train_loss[epoch,:] = train_loss_epoch
    val_loss[epoch,:] = val_loss_epoch
    
    np.save(opt.train_P_global_log_path, train_loss)
    np.save(opt.test_P_global_log_path, val_loss)
    
    if train_loss_epoch < best_loss:
        
        P1_model_path = (opt.P1_model_path).replace('.pt', '.h5')
        model.model_X1_1.save(P1_model_path)
        P2_model_path = (opt.P2_model_path).replace('.pt', '.h5')
        model.model_X2_1.save(P2_model_path)
        P3_model_path = (opt.P3_model_path).replace('.pt', '.h5')
        model.model_X3_1.save(P3_model_path)
        U_model_path = (opt.U_model_path).replace('.pt', '.h5')
        model.model_U_1.save(U_model_path)
        
        
        P1_model_path = P1_model_path.replace('level1', 'level2')
        model.model_X1_2.save(P1_model_path)
    
        P2_model_path = P2_model_path.replace('level1', 'level2')
        model.model_X2_2.save(P2_model_path)

        P3_model_path = P3_model_path.replace('level1', 'level2')
        model.model_X3_2.save(P3_model_path)

        U_model_path = U_model_path.replace('level1', 'level2')
        model.model_U_2.save(U_model_path)
        

        P1_model_path = P1_model_path.replace('level2', 'level3')
        model.model_X1_3.save(P1_model_path)
 
        P2_model_path = P2_model_path.replace('level2', 'level3')
        model.model_X2_3.save(P2_model_path)

        P3_model_path = P3_model_path.replace('level2', 'level3')
        model.model_X3_3.save(P3_model_path)

        U_model_path = U_model_path.replace('level2', 'level3')
        model.model_U_3.save(U_model_path)
        
        best_loss = train_loss_epoch

    elif epoch % 20 == 0:

        P1_model_path = (opt.P1_model_path).replace('.pt', '_'+str(epoch+1)+'.h5')
        model.model_X1_1.save(P1_model_path)
        P2_model_path = (opt.P2_model_path).replace('.pt', '_'+str(epoch+1)+'.h5')
        model.model_X2_1.save(P2_model_path)
        P3_model_path = (opt.P3_model_path).replace('.pt', '_'+str(epoch+1)+'.h5')
        model.model_X3_1.save(P3_model_path)
        U_model_path = (opt.U_model_path).replace('.pt', '_'+str(epoch+1)+'.h5')
        model.model_U_1.save(U_model_path)

        P1_model_path = P1_model_path.replace('level1', 'level2')
        model.model_X1_2.save(P1_model_path)

        P2_model_path = P2_model_path.replace('level1', 'level2')
        model.model_X2_2.save(P2_model_path)

        P3_model_path = P3_model_path.replace('level1', 'level2')
        model.model_X3_2.save(P3_model_path)

        U_model_path = U_model_path.replace('level1', 'level2')
        model.model_U_2.save(U_model_path)


        P1_model_path = P1_model_path.replace('level2', 'level3')
        model.model_X1_3.save(P1_model_path)

        P2_model_path = P2_model_path.replace('level2', 'level3')
        model.model_X2_3.save(P2_model_path)

        P3_model_path = P3_model_path.replace('level2', 'level3')
        model.model_X3_3.save(P3_model_path)

        U_model_path = U_model_path.replace('level2', 'level3')
        model.model_U_3.save(U_model_path)

    end = time.time()
    print('\n\n',end-begin,' sec / epoch')
    
"""
elif args.beta_adapt:
    loss_value_P = ((1/alpha_train[step,0])**beta_train[step,0])*\
    L_beta(pred_X1_1, y_P1_1, beta_train[step,0])+\
                 ((1/alpha_train[step,1])**beta_train[step,1])*\
    L_beta(pred_X2_1, y_P2_1, beta_train[step,1])+\
                 ((1/alpha_train[step,2])**beta_train[step,2])*\
    L_beta(pred_X3_1, y_P3_1, beta_train[step,2])+\
                 (((1/alpha_train[step,3])**beta_train[step,3])*\
    L_beta(pred_X1_2, y_P1_2, beta_train[step,3]))/4+\
                 (((1/alpha_train[step,4])**beta_train[step,4])*\
    L_beta(pred_X2_2, y_P2_2, beta_train[step,4]))/4+\
                 (((1/alpha_train[step,5])**beta_train[step,5])*\
    L_beta(pred_X3_2, y_P3_2, beta_train[step,5]))/4+\
                 (((1/alpha_train[step,6])**beta_train[step,6])*\
    L_beta(pred_X1_3, y_P1_3, beta_train[step,6]))/16+\
                 (((1/alpha_train[step,7])**beta_train[step,7])*\
    L_beta(pred_X2_3, y_P2_3, beta_train[step,7]))/16+\
                 (((1/alpha_train[step,8])**beta_train[step,8])*\
    L_beta(pred_X3_3, y_P3_3, beta_train[step,8]))/16
"""