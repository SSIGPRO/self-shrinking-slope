# CREATED BY Filippo Martinini (University of Bologna)

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.callbacks import Callback 

def myCallbacks(ldir, 
                checkpoint, 
                monitor='val_loss', 
                patienceEarlyStop=100,
                patienceLR=[50,50,50],
                min_lr = 0.000001,
                min_delta = 0.00000001,
                prancingPony = False,
                changeLR = 0.2,                
                changeSlope = 1.5,                  
                patienceSlope=[40,40,40,40],
                max_slope = 100000,
                idle_state_patience = 10,
                verbose = 1,
                layer_name = [['layer_1','layer_2'],['layer_3'],['layer_4','layer_5','layer_6']],
                double_round = False,
                exitIdleIfImprove = False,
               ):
    
    callback_list=[
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.ModelCheckpoint(filepath = checkpoint,
                                           monitor = monitor,
                                           verbose = 0,
                                           save_best_only = True,
                                           save_weights_only = True,
                                           save_freq = 'epoch',
                                          ),
        tf.keras.callbacks.TensorBoard(log_dir=ldir,
                                       histogram_freq=0,
                                       write_graph=False,
                                       write_images=False,
                                       update_freq='epoch',
                                       embeddings_freq=0,
                                       embeddings_metadata=None,
                                      ),
    ] 
    
    if prancingPony==True:
        callback_list+=[HolisticCallback_prancingPony(monitor = 'val_loss',
                                                      patienceEarly=patienceEarlyStop,
                                                      patienceLR=patienceLR,
                                                      patienceSlope=patienceSlope,
                                                      changeLR=changeLR, 
                                                      changeSlope = changeSlope,
                                                      minLR = min_lr,
                                                      maxSlope = max_slope,
                                                      idlePatience = idle_state_patience,
                                                      verbose = verbose,
                                                      layer_name = layer_name,
                                                      double_round = double_round, 
                                                      exitIdleIfImprove = exitIdleIfImprove,
                                                     )  
                        
                       ]
    else:
        callback_list += [tf.keras.callbacks.ReduceLROnPlateau(monitor=monitor,
                                                               factor = 0.2,
                                                               patience = patienceLR[0],
                                                               min_lr = min_lr,
                                                               min_delta=min_delta,
                                                               verbose = 1,
                                                              ),
                          tf.keras.callbacks.EarlyStopping(monitor=monitor,
                                                           min_delta=min_delta,
                                                           patience=patienceEarlyStop,
                                                           verbose=1,
                                                           mode='auto',
                                                           baseline=None,
                                                           restore_best_weights=True,
                                                          ),
                         ]
    
    return callback_list

def setSlopeTrainability(model, layer_name, trainable = False, verbose = False):
    i = 0
    for l in model.layers:
        i = i+1
        if l.name==layer_name:
            if verbose == False:
                print('\n\n\t\t SET SLOPE TRAINABILITY')
                print('layer n°: ', i,' - ', l.name, ' - trainability:')
                print('- before : ', l.trainable)
            l.trainable = trainable
            if verbose == False:
                print('- after: ', l.trainable)
    return model


class HolisticCallback_prancingPony(Callback):
    """
    ------- DESCRIPTION -------
    Replace 'reduceLRonPlateau' and 'EarlyStop' with this callback to modify the slope of a soft-threshold
    or a soft-max after a 'patienceSlope' number of epochs pass without the loss reaching a new lowest value.
    In its basic conception it only subsitutes the 'reduceLRonPlateau' with a kind of 'reduceSLOPEonPlateau'. 
    This can be done by setting patienceLR = [x] where x>patienceEarly.
    
    If model has more than one slope to modify, the user can set how many slopes modify (also contemporary) 
    and in what order.
    
    'Patience' variables accepts list of values, so that one can control the pace of the training (for both LR and 
    slope patience). Every time 'patience[i]' is reached, the new patience to reach becomes 'patience[i+1]'.
    USE THIS OPTION ONLY IF A MANUAL PATIENCE TUNING IS REQUIRED!
    
    Finally ,one can combine a first round of self-sigmoid tuning (without reducing the LR) with a round of training
    where the best model from the first round is kept and trained with a fixed slope but varying the LR 
    (as a classic training). To activate this behaviour set 'double_round=True': it will use 'patienceLR' only 
    during the second round and the 'patienceSlope' only during the first.
     
    ------- INPUTS -------
    monitor: specifies what metric to consider, recommended: 'val_loss'
    
    patienceEarly: number of epochs without the loss reaching a new lowest values before the training stops
    
    patienceLR: (MUST BE A LIST) number of epochs without the loss reaching a new lowest values before the 
    LR is reduced.
    
    patienceSlope: (MUST BE A LIST) number of epochs without the loss reaching a new lowest values before 
    the Slope is increased
    
    changeLR: every time patienceLR is reached LR = LR * changeLR
    
    changeSlope: every time patienceLSlope is reached slope = soe * changeSlope
    
    minLR: lower bound for the LR
    
    maxSlope: higher bound for the Slope
    
    idlePatience: maximum number of epochs of cooldown after the Slope is raised, during which the counter 
    of the epochs without improving is freezed.
    
    verbose: enable printing informations on the status of the callback
    
    layer_name: (MUST BE A LIST OF LISTS) the name (string) of the layers with the slope that will be modified by 
    Prancing Pony.
    layers = [[name1, ...], [...], ...]
    All the layers of layers[k] see their slope increasing at the same time. Every time Early stop occours k = k+1,
    and best weights from previous configurations are restored.
    
    double_round: if True --> the first round of training is only for auto-tuning the slope, the second is normal
    
    exitIdleIfImprove: if True --> when in 'idle state' the first epoch with a lower loss determines 
    the end of 'idle state'
  """

    def __init__(self, 
                 monitor = 'val_loss',
                 patienceEarly=100,
                 patienceLR=[50,50,50,50],
                 patienceSlope=[40,40,40],
                 changeLR=0.2, 
                 changeSlope = 1.5,
                 minLR = 0.000001,
                 maxSlope = 100000,
                 idlePatience = 10,
                 verbose = 1,
                 layer_name = [['layer_1','layer_2'],['layer_3'],['layer_4','layer_5','layer_6']],
                 double_round = False,
                 exitIdleIfImprove = False,
                ):
        
        self.layer_name = layer_name
        
        self.lenLayer = np.shape(np.array(layer_name, dtype=object))[0]-1
        print(self.lenLayer)
        self.patienceEarly = patienceEarly
        self.patienceLR = patienceLR

        self.patienceSlope = patienceSlope
                                     
        self.changeLR = np.float32(changeLR)
        self.changeSlope = np.float32(changeSlope)
        self.monitor = monitor
        
        self.idlePatience = idlePatience
        
        self.lenLR = len(patienceLR)
        self.lenSlope = len(patienceSlope)
        
        self.best_weights = None
        self.minLR = np.float32(minLR)
        self.maxSlope = np.float32(maxSlope)
        
        self.double_round = double_round
        
        self.verbose = verbose
        
        self.exitIdleIfImprove = exitIdleIfImprove
        
        super(HolisticCallback_prancingPony, self).__init__()

    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
   
        # Initialize the best as infinity.
        self.best = np.Inf
        self.flag = False
        self.prev = np.Inf
        
        self.count = 0
        self.countIdle = 0
        self.idleLR = 0
                
        self.indexSlope = 0
        self.indexLR = 0
        self.indexLayer = 0
                
        self.state = 'INITIALIZATION'
        
        if self.verbose == 1:
            print('state: ',self.state,'\nscheadule LR : ',self.patienceLR,'\nscheadule Slope : ',self.patienceSlope,'\ndouble round : ',self.double_round)
            
        if self.double_round == True: # secure that during the first round reduceLRonPlateau is not triggered
            self.patienceLR = [x+self.patienceEarly  for x in self.patienceLR]
        
        pass
    
    

    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        
        
        if self.flag==True:
            
            if (np.less(current,self.prev) and self.exitIdleIfImprove==True) or (self.countIdle >= self.idlePatience):
                self.state = 'IDLE END'
                self.flag = False
                self.best = current                
                self.countIdle = 0
                self.wait = 0
                
            else:
                self.state = 'IDLE'
                self.prev = current
                self.countIdle += 1
                pass
            pass
        
        
        elif np.less(current, self.best):
            self.state = 'DECREASING LOSS'
            self.best = current
            self.best_weights = self.model.get_weights()
            self.wait = 0

        else:
            self.wait += 1
            self.state = 'INCREASING LOSS'
            if self.wait >= self.patienceEarly:
                
                if self.verbose == 1:
                    print("Restoring model weights from the end of the best epoch.")
                
                self.model.set_weights(self.best_weights)
                self.wait = 0
                    
                if self.indexLayer<self.lenLayer:
                    self.indexLayer = self.indexLayer + 1
                    self.state = 'CHANGE LAYERS SUBJECT TO PRANCING PONY'
                    self.indexSlope = 0
                    self.indexLR = 0
                    
                elif self.double_round == True:
                    self.double_round = False
                    self.patienceLR = [x-self.patienceEarly  for x in self.patienceLR]
                    self.patienceSlope = [x+self.patienceEarly  for x in self.patienceSlope]
                    self.state = '\n\nSTOP FIRST ROUND ---- START SECOND ROUND\n\n'
                else:
                    self.model.stop_training = True
                    self.state = 'STOP TRAINING'
                
                pass

            
            
            elif self.wait >= self.patienceLR[self.indexLR]:
                self.state = 'REDUCE LR'
                
                lr = np.float32(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
                new_lr = np.float32(lr * self.changeLR)
                # Set the value back to the optimizer before this epoch starts
                if new_lr>=self.minLR:
                    tf.keras.backend.set_value(self.model.optimizer.lr, 
                                               new_lr )
                    if self.indexLR < self.lenLR-1:
                        self.indexLR = self.indexLR + 1
                    pass
                else:
                    tf.keras.backend.set_value(self.model.optimizer.lr,
                                               self.minLR)
                    self.state = 'LR MIN VAL'
            
            elif self.wait >= self.patienceSlope[self.indexSlope]:
                self.state = 'INCREASE SLOPE'
                for layer_name in self.layer_name[self.indexLayer]:
                    sl = np.float32(tf.keras.backend.get_value(self.model.get_layer(layer_name).slope))
                    new_slope = np.float32(sl*self.changeSlope)
                    if new_slope<=self.maxSlope:
                    
                        tf.keras.backend.set_value(
                            self.model.get_layer(layer_name).slope, 
                            new_slope
                        )
                        self.flag = True
                        self.prev = current
                    else:
                        tf.keras.backend.set_value(
                            self.model.get_layer(layer_name).slope, 
                            self.maxSlope
                        )
                        self.state = 'SLOPE MAX VAL'
                        
                if self.indexSlope < self.lenSlope - 1:
                    self.indexSlope = self.indexSlope + 1
                               
            else:
                pass
            
            pass
        if self.verbose == 1:
            print('\n   ---    \n',
                  '; lr = ',np.float32(tf.keras.backend.get_value(self.model.optimizer.learning_rate)),)
            for i, layer_name_tmp in enumerate(self.layer_name[self.indexLayer]):
                slope_tmp = np.float32(tf.keras.backend.get_value(self.model.get_layer(layer_name_tmp).slope))
                print('; slope ',i,' = ',slope_tmp,' - ', layer_name_tmp)
            print('; wait = ',self.wait,
                  '; patiece slope = ',self.patienceSlope[self.indexSlope],
                  '; patience LR  = ',self.patienceLR[self.indexLR],
                  '\nstate = ', self.state,
                  '\n   ---    \n',
                 )

        pass
    

    pass


    def on_train_end(self, logs=None):
        if self.verbose==1:
            print('STOP')
        pass
    pass