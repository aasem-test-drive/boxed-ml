from tensorflow import keras

class keras_applications:
    def __init__(self):
        pass

    def get_base_model(self,architecture,init_weights=None,img_width=224, img_height=224, img_channel=1):
        self.architecture=architecture.lower()
        self.init_weights=init_weights
        self.input_shape=img_width
        self.img_height=img_height
        self.img_channel=img_channel
        self.input_shape=(img_width,img_height,img_channel)
        
        if self.img_channel==1 and self.init_weights=='imagenet':
            print ("imagenet weights are not compatible with 1 channel images (grayscale)")        
        
        base={
            'xception':keras.applications.Xception(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'vgg16':keras.applications.VGG16(weights=self.init_weights,
                                       input_shape=(self.input_shape),
                                       include_top=False),
            'vgg19':keras.applications.VGG19(weights=self.init_weights,
                                       input_shape=(self.input_shape),
                                       include_top=False),
            'resnet50':keras.applications.ResNet50(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'resnet152v2':keras.applications.ResNet152V2(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'inceptionv3':keras.applications.InceptionV3(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'inceptionresnetv2':keras.applications.InceptionResNetV2(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'mobilenet':keras.applications.MobileNet(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'mobilenetv2':keras.applications.MobileNetV2(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'densenet121':keras.applications.DenseNet121(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'densenet201':keras.applications.DenseNet201(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'nasnetmobile':keras.applications.NASNetMobile(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'nasnetlarge':keras.applications.NASNetLarge(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'efficientnetb0':keras.applications.EfficientNetB0(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False),
            'efficientnetb1':keras.applications.EfficientNetB1(weights=self.init_weights,
                                                   input_shape=(self.input_shape),
                                                   include_top=False)
            }

        return (base[self.architecture])