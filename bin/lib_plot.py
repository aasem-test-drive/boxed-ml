import pandas as pd
from matplotlib import pyplot as plt

def show_dataset_chart(trainset_dict, validset_dict, save_filename="",show_legend=False):
    trainColor='orange'
    validationColor='blue'
    
    label_y='instances'
    label_x='classes'
    chart_title='Dataset'
    plt.rcParams['axes.facecolor'] = 'none'
    fig = plt.figure(dpi=200)
    fig.patch.set_facecolor('xkcd:white')

    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    ax.set_title(chart_title)
    ax.xlabel='instances'


    ax.barh(list(trainset_dict.keys()),list(trainset_dict.values()),color=trainColor,label='Train')
    ax.barh(list(validset_dict.keys()),list(validset_dict.values()),color=validationColor,label='Valid')
    ax.set_xlabel(label_y) 
    ax.set_ylabel(label_x)


    #ax.pie(data_Dictionary.values(), labels = data_Dictionary.items(),autopct='%1.1f%%')
    plt.legend() if show_legend else None
    plt.tight_layout()
    plt.savefig(save_filename,dpi=100,pad_inches=5,) if len(save_filename)>0 else None
        
    
def show_training_performance(path_csvname, save_filename="",show_legend=False):
    
    history=pd.read_csv(path_csvname)
    trainColor='orange'
    validationColor='blue'
    figure, axis = plt.subplots(nrows=2, ncols=2, 
                                figsize=(8,4),
                                dpi=100,sharey=False,sharex=False, 
                                squeeze=True)
    plt.subplots_adjust(top = 0.99, bottom=0.01, hspace=0.4, wspace=0.4)

    # accuracy
    axis[0, 0].plot(history['accuracy'],color=trainColor,label='Accuracy (Train)')
    axis[0, 0].plot(history['val_accuracy'],color=validationColor,label='Accuracy (Valid)')
    axis[0, 0].set_title("Accuracy")
    axis[0, 0].legend() if show_legend else None

    

    # loss
    axis[1, 0].plot(history['loss'],color=trainColor,linestyle='dashed',label='Loss (Train)')
    axis[1, 0].plot(history['val_loss'],color=validationColor,linestyle='dashed',label='Loss (Train)')
    axis[1, 0].set_title("Loss")
    axis[1, 0].legend() if show_legend else None

    
    # Summary
    axis[0, 1].set_title("Summary")
    axis[0, 1].plot(history['accuracy'],color=trainColor,label='Accuracy (Train)')
    axis[0, 1].plot(history['val_accuracy'],color=validationColor,label='Accuracy (Valid)')
    axis[0, 1].plot(history['loss'],color=trainColor,linestyle='dashed',label='Loss (Train)')
    axis[0, 1].plot(history['val_loss'],color=validationColor,linestyle='dashed',label='Loss (Train)')
    axis[0, 1].legend() if show_legend else None

    # learning rate
    if "lr" in history:
        axis[1, 1].set_title("learning rate")
        axis[1, 1].plot(history['lr'],color="red")
    else:
        epoch = list(history.iloc[:, 0])
        accuracy = list(history.iloc[:, 1])
        val_accuracy = list(history.iloc[:, 3])
        loss = list(history.iloc[:, 2])
        val_loss = list(history.iloc[:, 4]) 
        
        axis[1, 1].bar(epoch, accuracy, color='orange',label='Accuracy (Train)')
        axis[1, 1].bar(epoch, val_accuracy, color='blue',label='Accuracy (Valid)')
        axis[1, 1].set_title("Training Performance")
        axis[1, 1].legend() if show_legend else None
        
      
        #plt.setp(axis[1, 1], xlabel='epochs')
    

    if len(save_filename)>0:
        plt.savefig(save_filename,dpi=100,pad_inches=5,)
    #plt.legend()
    plt.show()

    
def show_chart(chart_Type='barh', data_Dictionary={}, chart_title="title",label_x='classes', label_y='instances'):
    plt.rcParams['axes.facecolor'] = 'none'
    fig = plt.figure(dpi=200)
    fig.patch.set_facecolor('xkcd:white')

    ax = fig.add_subplot(1, 1, 1) # nrows, ncols, index
    ax.set_title(chart_title)
    ax.xlabel='instances'

    if chart_Type=='bar':
        ax.bar(*zip(*data_Dictionary.items()))
        ax.set_xlabel(label_x) 
        ax.set_ylabel(label_y)


    elif chart_Type=='barh':
        ax.barh(list(data_Dictionary.keys()),list(data_Dictionary.values()))
        ax.set_xlabel(label_y) 
        ax.set_ylabel(label_x)

    elif chart_Type=='pie':
        ax.pie(data_Dictionary.values(), labels = data_Dictionary.items(),autopct='%1.1f%%')      
    plt.tight_layout()