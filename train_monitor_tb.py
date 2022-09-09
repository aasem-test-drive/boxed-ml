
import sys, getopt, os
from lib.lib_utils import utilities

#!kill $(ps aux | grep 'tensorboard_logs' | awk '{print $2}')
def main(argv):
    util=utilities()
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile=","ofile="])
        
    except getopt.GetoptError:
        print ('train_monitor_tb.py -i <inputfile>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print ('train_monitor_tb.py -i <conf_yourFilename.json>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    print ('Input file is {'+ inputfile+'}')    
    USER_PARA=util.load_JSON_file(inputfile)
    
    
    # step: Definitions
    prefix=USER_PARA['prefix']
    project_workspace=os.path.join('my_workspace',prefix)
    path_tensorboardLog = os.path.join (project_workspace,"tensorboard_logs")
    print (path_tensorboardLog)

    print('\nTensorboard setup...')
    print('\tCopy and execute:')
    print(f'\t  tensorboard --logdir='+path_tensorboardLog+' --host 0.0.0.0 --port 6006')
    print('\tthen open this in browser:')
    print('\t  http://localhost:6006')

    os.system('tensorboard --logdir='+path_tensorboardLog+' --host 0.0.0.0 --port 6006')


if __name__ == "__main__":
    main(sys.argv[1:])
