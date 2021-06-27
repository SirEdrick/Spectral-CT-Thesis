#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 1 13:25:08 2021

@author: wamus, bruun
"""
from argparse import ArgumentParser
from dataStructures import DataSet
from dataStructures import CTData

if __name__ == '__main__':

    option_parser = ArgumentParser(description="example script for reading and writing spectral ct dataset")
    option_parser.add_argument("-d","--datasetPath",action="store",dest="datasetPath",help="full path to the dataset")
    options = option_parser.parse_args()
    arg_dict = vars(options)

    #N=100 # image dimesion (NxNxN) = (x,y,x)., For now, all dimensions have to be the same as object generator expects, let's see if we can change that
    #sigma=0.03 # noise added to the simulated objects
    object_scale = 1 # This is a scaling factor for reconsucted objects, keep at 1
   
    datapath_head = arg_dict["datasetPath"]
    data_set =  DataSet.DataSet(datapath_head)
    
    #sinogram shape after loading: (# of spectal channels,# of detector pixels, # of slices, # of projections)
    # because the data files are very large, sometimes (especially for testing you don't want to read all the data but perhaps one slice)
    '''
    data_set.sinogram_data.setLoadChannels(start=0,stop=128,step=1) # comment out this line if you want to load all channels
    data_set.sinogram_data.setLoadSliceZ(start=0,stop=300,step=1) # comment out this line if you want to load all slices
    '''
    #data_set.sinogram_data.loadData('h5')
    
    #data_set.sinogram_data.saveData('h5') # to save the data as H5 files
    #data_set.sinogram_data.saveData('png') # to save the data as png images

    ## reconstruction is the process where we obtain images from sinograms
    #reconstruction (image) shape after loading: (X,Y, Z, # of spectal channels)
    
    #data_set.reconstruction_data.setLoadChannels(start=0,stop=128,step=1)
    #data_set.reconstruction_data.setLoadSliceZ(start=0,stop=300,step=10)

    data_set.reconstruction_data.file_name = 'reconstruction_NN_9'
    data_set.reconstruction_data.loadData('h5')

    import numpy
    data_set.reconstruction_data.data = numpy.delete(data_set.reconstruction_data.data, slice(290, 300), 2)

    data_set.reconstruction_data.saveData('png') # to save the data as png images
    
    #data_set.reconstruction_data.save_images_montage() # another way to visualize the data; save as gif
    
    # segmentation information 

    label_data = CTData.LabelData(datapath_head, file_name='segmented_manual') 
    
    #label_data.setLoadSliceZ(start=0,stop=300,step=10)
    
    label_data.loadData('h5')
    label_data.data = label_data.data.transpose(1, 0, 2)
    label_data.data = label_data.data[::-1, :, :]
    label_data.data = label_data.data[2:98, 2:98, :]

    label_data.loadLabelMap()
    label_data.saveData('png')

    print("--------------------------new-------------")
    
    label_data.extractSegmentValues(data_set.reconstruction_data)
    label_data.saveExtractedValue()