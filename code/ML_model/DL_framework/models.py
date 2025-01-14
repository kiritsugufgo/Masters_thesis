import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Conv2D, LeakyReLU, MaxPooling2D
from tensorflow.keras.optimizers import SGD

def DNN(config):
    dnn = tf.keras.sequential()
    dnn.add(tf.kerasInput(shape=tuple(config['shape'])))
    for layer in config['layers']:
        dnn.add(tf.keras.layers.Dense(layer, activation='relu'))
    dnn.add(Dense(1,activation='sigmoid'))
    
    #Use SDG
    sdg = SDG(learning_rate=0.01, momentum=0.9)
    dnn.compile(optimizer=sdg, loss='binary_crossentropy', metrics=['accuracy'])
    
    return dnn

# def CNN(config):
    
#     inputs = Input(tuple(config['shape']))
    
#     x = conv_block(inputs, config['filters'], config['batch_norm'], config['dropouts'][0])
#     x = conv_block(x, config['filters']*2, config['batch_norm'], config['dropouts'][1])
#     x = conv_block(x, config['filters']*4, config['batch_norm'], config['dropouts'][2])
#     x = conv_block(x, config['filters']*8, config['batch_norm'], config['dropouts'][3])
#     x = GlobalMaxPooling()(x)
#     outputs = Dense(config['num_classes'], activation='softmax')(x)
    
#     model = tf.keras.Model(inputs=inpus, outputs, outpus, name='CNN')
    
#     return model
    
# def conv_block(inputs=None, n_filter=64, batch_nor,=False, dropout=0):
    
#     conv = Conv2D(n_filters, 3, padding='same')(inputs)
#     if barch_norm:
#         conv = BatchNormalization(axis=1)(conv)
#     conv = LeakyRelu(alpha=0.2)(conv)
    
#     if dropout > 0:
#         conv = Dropout(dropout)(conv)
#     conv = MaxPooling2D((2,2))(conv)
    
#     return conv

# class CNN(tf.keras.Model):
    
#     def __init__(self, num_class=6):
#         self.global_pool = tf.keras.layers.GlobalAveragePooling()
#         self.classifier = Dense(num_class, activation='softmax')
        
#     def conv_block(self, inputs=None, n_filter=64, batch_nor,=False, dropout=0):
#         conv = Conv2D(n_filters, 3, padding='same')(inputs)
#         if barch_norm:
#             conv = BatchNormalization(axis=1)(conv)
#         conv = LeakyRelu(alpha=0.2)(conv)
        
#         if dropout > 0:
#             conv = Dropout(dropout)(conv)
#         conv = MaxPooling2D((2,2))(conv)
        
#         return conv

#     def Call(self, inputs)
#         x = self.conv_block(inputs, 64, True, 0.3)
#         x = self.conv_block(inputs, 64*2, True, 0.3)
#         x = self.conv_block(x, 64*4, True, 0.3)
#         x = self.conv_block(x, 64*8, True, 0.5)
#         x= self.global_pool(x)
#         return self.classifier(x)
    
def get_complied_model(model, optimizer, loss, metric):
    
    print(model.summary())
    print("Compiling and returning model")
    
    model.compile(
        optimizer=optimizer,
        loss=[loss],
        metrics=[metrics]
    )
    
    return model