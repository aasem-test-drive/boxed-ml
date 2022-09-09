#!/usr/bin/python
import sys, getopt, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" # or any {'0', '1', '2','3'}

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

from lib.lib_utils import utilities
from lib.lib_logger import logger
from lib.lib_keras import keras_applications
from lib.lib_dataset import Dataset
from lib.lib_plot import show_training_performance,show_dataset_chart


# Definitions 

def create_model(base,num_of_classes,model_name):
    #biasInitializer = tf.keras.initializers.HeNormal(seed=101)
    inputShape=base.input_shape
    inputShape=(inputShape[1],inputShape[2],inputShape[3])
    inputs = keras.Input(shape=inputShape)
    x = base(inputs, training=True)## todo
    #x = keras.layers.Conv2D(1280*2, (3, 3), strides= (1, 1), activation="relu", name="for_CAM")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(num_of_classes)(x)
    outputs = keras.layers.Softmax()(x)
    model=keras.Model(inputs, outputs,name=model_name)
    return model

def print_summary(x):
    log_txt.print_log("\t"+x,print_on_screen=True)
    #print("\t"+x)
    x=str(x)
    if x.startswith("Total params:") or x.startswith("Trainable params:") or x.startswith("Non-trainable params:"):
        token=x.split(':')
        k=token[0]
        v=token[1].strip()
        _data[k]=v


# Main()

def main(argv):
    util=utilities()
  
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile=","ofile="])
        
    except getopt.GetoptError:
        print ('train.py -i <inputfile>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print ('train.py -i <conf_yourFilename.json>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    print ('Input file is {'+ inputfile+'}')
    #USER_PARA=util.load_JSON_file('my_workspace/exp00_montgomeryset.json')    
    USER_PARA=util.load_JSON_file(inputfile)
    
    # step: Definitions
    prefix=USER_PARA['prefix']
    project_workspace=os.path.join('my_workspace',prefix)

    color_mode=USER_PARA["colorMode"]
    imgChannel= 1 if color_mode=='grayscale' else 3
    image_size=(USER_PARA['imgHeight'],USER_PARA['imgWidth'])

    image_shape=(USER_PARA['imgHeight'],USER_PARA['imgWidth'],imgChannel)

    logFilenameJSON=os.path.join(project_workspace,"log.json")
    logFilenameTXT=os.path.join(project_workspace,"log.txt")

    base_architecture=USER_PARA['architecture']

    root_dataset=USER_PARA["root_dataset"]

    batch_size=USER_PARA["batch_size"]
    dataset_tag=USER_PARA['dataset_tag']
    epochs=USER_PARA['epochs']

    callbacks_enabled={"CSVLogger":True,
                         "TensorBoard":True,
                         "ModelCheckpoint":True,
                         "EarlyStopping":True,
                         "TerminateOnNaN":True,
                         "ReduceLROnPlateau":True,
                         "LearningRateScheduler":False,
                         "RemoteMonitor":False}
    _exception={}            
    
    
    # step: directory structure and filenames
    if not os.path.isdir(project_workspace):
        os.mkdir(project_workspace)

    path_to_image=os.path.join(project_workspace,'image')
    if not os.path.isdir(path_to_image):
        os.mkdir(path_to_image)

    log_json=logger(logFilenameJSON)
    log_txt=logger(logFilenameTXT)

    log_txt.print_log("Getting Starting...",overwrite=True)
    log_txt.print_log('\tprefix:'+prefix)

    log_json.print_jsonlog({'prefix':prefix},overwrite=True)
    
    # step: Dataset
    log_txt.print_log('Loading Dataset...')
    datasetOBJ=Dataset(dataset_tag=dataset_tag,
                 path_to_dataset=root_dataset,
                 image_size=image_size, 
                 color_mode=color_mode, 
                 batch_size=batch_size
                )
    (x_train, y_train), (x_valid, y_valid)=datasetOBJ.load_dataset()
    num_of_classes=x_train.num_classes
    
    trainset_dict=datasetOBJ.as_dictionary(x_train)
    validset_dict=datasetOBJ.as_dictionary(x_valid)
    #show_chart(chart_Type="barh", data_Dictionary=trainset_dict,chart_title='Training Set',label_x='classes',label_y="samples") 
    #show_chart(chart_Type="barh", data_Dictionary=trainset_dict,chart_title='Training Set',label_x='classes',label_y="samples") 
    dataset_distribution=os.path.join(project_workspace,'image','dataset_distribution.png')
    show_dataset_chart(trainset_dict,validset_dict,save_filename=dataset_distribution,show_legend=True )
    
    _data={        
        "dataset":{
            "trainset":{"num_classes":x_train.num_classes,
                        "num_samples":x_train.samples,
                        "class_wise_count":datasetOBJ.trainSet_as_dictionary()
                      },
            "validateset":{"num_classes":x_valid.num_classes,
                        "num_samples":x_valid.samples,
                        "class_wise_count":datasetOBJ.validateSet_as_dictionary()
                      }        
            }
        }

    _d1=f'\tTrainSet: Found {_data["dataset"]["trainset"]["num_samples"]} images belonging to {_data["dataset"]["trainset"]["num_classes"]} classes.'
    _d2=f'\tValidateset: Found {_data["dataset"]["validateset"]["num_samples"]} images belonging to {_data["dataset"]["validateset"]["num_classes"]} classes.'

    log_txt.print_log(_d1,print_on_screen=False)
    log_txt.print_log(_d2,print_on_screen=False)
    log_json.print_jsonlog(_data)
    

    # step: Modeling
    keras_app=keras_applications()
    base_model=keras_app.get_base_model(base_architecture)

    log_txt.print_log("Modeling...")
    log_txt.print_log("\tbase_model: "+ base_architecture)
    
    ## create model
    model=create_model(base=base_model,
                   num_of_classes=num_of_classes,
                   model_name=prefix)
    
    _data={}
    #model.summary(print_fn=print_summary)
    _data={"model":{"base_model":base_architecture,
                    "summary":_data
                   }
    }
    log_json.print_jsonlog(_data)

    model_snap=os.path.join(project_workspace,'image','model_snap.png')
    keras.utils.plot_model(model,
                           to_file=model_snap,
                           show_shapes=True,
                           show_dtype=False,
                           show_layer_names=True,
                           rankdir='TB',
                           expand_nested=False,
                           dpi=96)    
        
    #callbacks
    # https://blog.paperspace.com/tensorflow-callbacks/
    path_csvname = os.path.join (project_workspace,"CSVLogger.csv")
    path_checkpoint = os.path.join(project_workspace,'ckpt','model_epoch{epoch:02d}_vLoss{val_loss:.2f}.hdf5')
    path_custom_file1 = os.path.join (project_workspace,"log_batchwise.txt")

    path_tensorboardLog = os.path.join (project_workspace,"tensorboard_logs")
    #path_tensorboardLog = os.path.join (project_workspace,'tensorboard_logs',"scalars" , datetime.now().strftime("%Y%m%d-%H%M%S"))


    CSVLogger_cb = tf.keras.callbacks.CSVLogger(path_csvname, 
                                 separator=',', 
                                 append=True)


    TensorBoard_cb = tf.keras.callbacks.TensorBoard(log_dir=path_tensorboardLog,
                                              histogram_freq=1,
                                              write_graph=True,
                                              write_images=True,
                                              update_freq=batch_size,
                                              #write_steps_per_second=False,
                                              profile_batch=2,
                                              embeddings_metadata=None)




    ModelCheckpoint_cb = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                    save_best_only=True, ###to save space
                                                    save_weights_only=False,
                                                    monitor='val_accuracy',
                                                    mode='max')



    EarlyStopping_cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                       min_delta=0.001,
                                                       patience=4,
                                                       verbose=0, 
                                                       mode='auto',
                                                       baseline=None,
                                                       restore_best_weights=False)

    TerminateOnNaN_cb = tf.keras.callbacks.TerminateOnNaN()

    ReduceLROnPlateau_cb = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                                factor=0.01,
                                                                patience=3,
                                                                verbose=0,
                                                                mode='auto',
                                                                min_delta=0.001,
                                                                cooldown=0,
                                                                min_lr=0)

    def scheduler(epoch, lr):
        if epoch < 15:
            return lr
        else:
            return lr * tf.math.exp(-0.0001)

    LearningRateScheduler_cb = tf.keras.callbacks.LearningRateScheduler(scheduler, 
                                                                        verbose=0)



    RemoteMonitor_cb = tf.keras.callbacks.RemoteMonitor(root='http://localhost:9000',
                                                        path='/publish/epoch/end/',
                                                        field='data',
                                                        headers=None,
                                                        send_as_json=False)


    callback_dictionary={"CSVLogger":{"callback":CSVLogger_cb},
                         "TensorBoard":{"callback":TensorBoard_cb},
                         "ModelCheckpoint":{"callback":ModelCheckpoint_cb},
                         "EarlyStopping":{"callback":EarlyStopping_cb},
                         "TerminateOnNaN":{"callback":TerminateOnNaN_cb},
                         "ReduceLROnPlateau":{"callback":ReduceLROnPlateau_cb},
                         "LearningRateScheduler":{"callback":LearningRateScheduler_cb},
                         "RemoteMonitor":{"callback":RemoteMonitor_cb}}
    callbacksList=[]
    for k in callbacks_enabled:
        if callbacks_enabled[k]:
            callbacksList.append(callback_dictionary[k]["callback"])

    # step: training
    historyIndex=0
    index_finish=0

    steps=epochs
    _metrics=['MeanSquaredError','AUC','Precision','Recall','accuracy']
    histories={}
    _data={"training":{}}

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=_metrics,
        loss_weights=None,
        weighted_metrics=None,
        run_eagerly=None,
        steps_per_execution=None
    )
    class_weight_dict=None

    _data_compile={"optimizer":"adam",
                   "loss":"CategoricalCrossentropy",
                   "metrics":["accuracy"],
                   "loss_weights":"None",
                   "weighted_metrics":"None"
                  }

    log_txt.print_log("Training...")
    log_txt.print_log("\tcompile:")
    log_txt.print_log("\t\toptimizer: "+_data_compile["optimizer"])
    log_txt.print_log("\t\tloss: "+_data_compile["loss"])
    log_txt.print_log("\t\tloss_weights: "+_data_compile["loss_weights"])   
    log_txt.print_log("\t\tweighted_metrics: "+_data_compile["weighted_metrics"])     
    
    index_start=index_finish
    index_finish+=steps
    historyIndex+=1
    _data={"training":{}}
    _data_epoch={"index":historyIndex,
                 "epoch_from":index_start,
                 "epoch_to":index_finish
                }
    _data["training"]["session"]=_data_epoch
    _data["training"]["compiler"]=_data_compile

    log_txt.print_log("\tsession:")
    log_txt.print_log(f"\t\tindex:{historyIndex}, epoch from {index_start} to {index_finish}")
    log_json.print_jsonlog(_data)

    try:
        train_session = model.fit(
            x=x_train,
            #y=y_train,
            #validation_split=0.0,
            validation_data=(x_valid),
            batch_size=batch_size,
            epochs=index_finish,
            callbacks=callbacksList,
            verbose=1,
            shuffle=True,
            class_weight=class_weight_dict,
            sample_weight=None,
            initial_epoch=index_start,
            steps_per_epoch=None,
            validation_steps=None,
            validation_batch_size=None,
            validation_freq=1,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False)

        #histories[historyIndex]=train_session.history
        log_txt.print_log("\tresults:",print_on_screen=False)
        log_txt.print_log(str(train_session.history),print_on_screen=False)

        if len(train_session.epoch)<index_finish-index_start:
            log_txt.print_log('\tTraining session completed at epoch '+str(len(train_session.epoch)))
            _data={"stopped_at_epoch":len(train_session.epoch)}
            log_json.print_jsonlog(_data)

        training_performance = os.path.join (path_to_image,"training_performance.png")
        show_training_performance(path_csvname=path_csvname,save_filename=training_performance)

    except Exception as e:
        if e.error_code==8:
            _exception={"error":
                        {"code":e.error_code,
                         "desc":"Resource exhausted",
                         "fix":["reduce batch size"]
                        }
                       }
        else:
            _exception={"error":
                        {"code":e.error_code,
                         "desc":str(e),
                        }
                       }

        log_txt.print_log(str(_exception))
        log_json.print_jsonlog(_exception)    
        
    # step: Save the model
    fname = os.path.join (project_workspace,"saved_model")
    savedModel = os.path.join(project_workspace, "saved_model")
    savedWeights = os.path.join(project_workspace,"saved_weights","weights")


    log_txt.print_log("Saving...")
    model.save(savedModel)
    log_txt.print_log("\tComplete model saved:"+ str(savedModel))
    model.save_weights(savedWeights)
    log_txt.print_log("\tWeights saved:"+ str(savedWeights))

    _data={"saved":{"full":str(savedModel),
                  "weights":str(savedWeights)}}
    log_json.print_jsonlog(_data)


if __name__ == "__main__":
    main(sys.argv[1:])
