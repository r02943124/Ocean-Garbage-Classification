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

