#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import tensorflow as tf
import numpy as np
from numpy import array
from random import shuffle
from tensorflow.contrib import layers
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_random_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.gen_random_ops import *
import os, sys, itertools, argparse
from cnn import CNN
from Input_Ocean import Input_Ocean
import cnn_parameter


class Configuration( object ):
    '''a configuration object'''
    def __str__( self ):
        def report( a ):
            rep = '{0}={1}'.format( a, getattr(self, a) )
            if a.endswith( 'err' ): rep += '={0}'.format( sum( getattr(self, a) ) )
            return rep
        return '\n'.join( [ report( a ) for a in self.__dict__ ] )


def pack( args, subset ):
    '''
    return a list of configurations
    '''
    ret = []
    l = [ getattr(args, key) for key in subset ] + [ range(args.repeat) ]
    for line in itertools.product( *l ):
        item = Configuration()
        for key, number in zip( subset, line ):
            setattr( item, key, number )
        setattr( item, 'trial', line[-1] )
        ret.append( item )
    return ret


def report( rank ):
	if rank >= len(top): return
	print( '--== rank {0} ==--'.format( rank+1 ) )
	print( configs[top[rank]] )
	print( '' )


                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
def exp_train( dataset, config, parameters):
	tf.reset_default_graph()
	save_file_name 	= 	"/Users/chenj0e/Dropbox/tensorflow/model.ckpt"
	cnn  = CNN(parameters)
	parameters.learning_rate 	= config.lrate
	parameters.batch_size       = config.batchsize
	parameters.layer_shape      = config.shape
	parameters.training_epochs	= config.nbatch
	cost_value=cnn.training_inference(dataset,save_file_name)
	return cost_value


def exp_test( dataset, config, parameters):
	tf.reset_default_graph()
	save_file_name 	= 	"/Users/chenj0e/Dropbox/tensorflow/model.ckpt"
	cnn  = CNN(parameters)
	parameters.learning_rate 	= config.lrate
	parameters.batch_size       = config.batchsize
	parameters.layer_shape      = config.shape
	parameters.training_epochs	= config.nbatch
	cost_value=cnn.testing_inference(dataset,save_file_name)
	return cost_value



class Parameters(object):
        
    learning_rate   = 0.0001
    training_epochs = 100
    batch_size      = 100
    layer_shape     = None
    kernal_size     = None
    drop_prob       = None
    metaclass       = None
    image_size      = None
    input_channel   = None
    pool            = None
    con_strides     = [1,1,1,1]    
    con_padding     = 'SAME'
    max_pool_ksize  = [1, 2, 2, 1]
    max_pool_strides= [1, 2, 2, 1]
    max_pool_padding= 'SAME'
    reuse           = None
    resize_seperate = [8 , 1 , 1]
    display_step    = 1
    gup_on          = False



def main( args ):
    basic_arr 					= [ 'lrate', 'shape', 'nbatch', 'batchsize' ,'data_file_name']
    configs 					= pack( args,  basic_arr )
    parameters                  = Parameters()
    parameters.learning_rate    = 0.0001
    parameters.training_epochs  = 10
    parameters.batch_size       = 100
    parameters.image_size       = [28,28]
    parameters.layer_shape      = [32,64,1024,7]
    parameters.kernal_size      = [5, 5]
    parameters.drop_prob        = 0.5 
    parameters.metaclass        = 0 
    parameters.input_channel    = 1 
    parameters.dtype            = tf.float32
    parameters.pool             = 1 
    parameters.con_strides      = [1,1,1,1]
    parameters.con_padding      = 'SAME'
    parameters.max_pool_ksize   = [1, 2, 2, 1]
    parameters.max_pool_strides = [1, 2, 2, 1]
    parameters.max_pool_padding = 'SAME'
    parameters.reuse            = False        
    parameters.resize_seperate  = [8 , 1 , 1]
    parameters.display_step     = 1 
    parameters.gup_on           = False



    print( 'trying {0} configurations'.format(len(configs)) )
    for seed, config in enumerate( configs ):
    	data = Input_Ocean.inputs(config.data_file_name,parameters)
    	print('file name:',config.data_file_name)
    	print( '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~' )
    	print("trying config:", config )
    	cost_dict={}
        cost_value = exp_train(data, config, parameters )
        cost_dict.update({cost_value:parameters.layer_shape})
        cost_dict.items()
        print(cost_dict)
        data = Input_Ocean.inputs('target_data_725_Objecthuman30.csv',parameters)
        exp_test(data, config, parameters )


    # sort the models based on validation error



if __name__ == '__main__':
    parser 		= argparse.ArgumentParser( description='A Deme: cross validation' )
    # common options
    parser.add_argument( '--lrate',     nargs='+', type=float, default=[5e-4,1e-4,5e-3,1e-3], help='learning rates' )
    parser.add_argument( '--nbatch',    nargs='+', type=int,   default=[1000], help='number of batches' )
    parser.add_argument( '--batchsize', nargs='+', type=int,   default=[100],   help='batch size' )
    parser.add_argument( '--repeat',    type=int, default=1, help='number of random initializations' )
    parser.add_argument( '--data_file_name',  type=str, default=None, help='Name of input data' )
    args 		= parser.parse_args()
    args.shape 	= ([32,64,128,7],
                  [32,64,256,7],
                  [32,64,512,7],
                  [32,64,1024,7],
                  [32,64,2048,7])

    args.data_file_name = args.data_file_name,
    print(args.data_file_name) 
    for key, value in args.__dict__.items():
        print( key, ':', value )
    main(args)
