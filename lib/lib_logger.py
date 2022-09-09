import os
from datetime import datetime
import json

class logger:
    def __init__(self,path_to_file):
        self.path_to_file=path_to_file
        
    def print_log(self, string_text, print_on_screen=True, overwrite=False):
        newline='\n'
        print (string_text) if print_on_screen else None
        mode='w' if overwrite else 'a'

        with open (os.path.join (self.path_to_file), mode) as f:
            f.write (str(string_text)+newline) 

            
    def print_jsonlog(self,_dict,overwrite=False):
        path=self.path_to_file
        ts=str(datetime.now())
        if overwrite:
            if os.path.isfile(path):
                os.unlink(path)

        # https://stackoverflow.com/questions/12994442/how-to-append-data-to-a-json-file
        with open(path, 'ab+') as f:
            f.seek(0,2)                                #Go to the end of file    
            if f.tell() == 0:             #Check if file is empty
                f.write(json.dumps([_dict]).encode())  #If empty, write an array
            else :
                f.seek(-1,2)           
                f.truncate()                           #Remove the last character, open the array
                f.write(' , '.encode())                #Write the separator
                f.write(json.dumps(_dict).encode())    #Dump the dictionary
                f.write(']'.encode())                  #Close the array          