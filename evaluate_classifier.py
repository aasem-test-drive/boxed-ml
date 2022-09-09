
import sys, getopt, os
from lib.lib_utils import utilities

def main(argv):
    util=utilities()
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile=","ofile="])
        
    except getopt.GetoptError:
        print ('evaluate_classifier.py -i <inputfile>')
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print ('evaluate_classifier.py -i <conf_yourFilename.json>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

   
    USER_PARA=util.load_JSON_file(inputfile)
    

    # step: Definitions
    prefix=USER_PARA['prefix']
    project_workspace=os.path.join('my_workspace',prefix)
    path_tensorboardLog = os.path.join (project_workspace,"tensorboard_logs")
    print (path_tensorboardLog)


    print ('loading dataset...')
    



if __name__ == "__main__":
    main(sys.argv[1:])
