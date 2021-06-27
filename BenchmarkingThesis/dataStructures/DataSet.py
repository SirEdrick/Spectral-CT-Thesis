'''
Created on 19 sept. 2018

@author: Wail Mustafa
'''
import sys
import os
import string
#from oct2py import *
import json
from dataStructures import CTData
import numpy
import itertools

class DataSet():
 
    
    
    def __init__(self, datapath_head):
       self.datapath_head = datapath_head

       self.loadGeometry()

       self.initDataTypes()
        
    def __del__(self):
        del self.geostruct

    def initDataTypes(self):
        self.raw_data = CTData.RawData(self.datapath_head)
        self.corrected_data = CTData.CorrectedData(self.datapath_head)
        self.sinogram_data = CTData.SinogramData(self.datapath_head)
        self.reconstruction_data = CTData.ReconstructionData(self.datapath_head)
    
    def loadGeometry(self):
        #geostruct = octave.load_geometry(self.datapath_head)
        
        with open(self.datapath_head+'/geometry.json', 'r') as fp:
            geostruct= json.load( fp)

        #with open(self.datapath_head+'/geometry.json', 'w') as fp:
            #json.dump(geostruct, fp)
        
        
        print (geostruct)
        linesPerSlice=1
        #if geostruct.has_key("acc_proj"):
        if 'acc_proj' in  geostruct:
           geostruct["acc_proj"]=int(geostruct["acc_proj"]);
           geostruct["nproj"]=int(geostruct["nproj"])
           geostruct["ndet"]=int(geostruct["ndet"])
           geostruct["nElem"]=int(geostruct["nElem"])
           #linesPerSlice=1
           if "nSliceLines" in geostruct:
              if(geostruct["vol"] and (geostruct["nSliceLines"]!=None)):
                 geostruct["nSliceLines"]=int(geostruct["nSliceLines"])
                 linesPerSlice = geostruct["nSliceLines"]
        print ("geometry loaded from class.")
        print (geostruct)
        

        self.geostruct = geostruct
        self.linesPerSlice = linesPerSlice
