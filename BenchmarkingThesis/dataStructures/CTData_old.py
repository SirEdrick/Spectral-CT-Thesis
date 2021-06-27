'''
Created on 19 sept. 2018
@author: Wail Mustafa
'''
import sys
import os
import string
import h5py
#from libtiff import TIFF
from PIL import Image
import numpy
import itertools
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import imageio

class CTData(object):
    def __init__(self, datapath_head, sub_dir, file_name):
        self.datapath_head = datapath_head
        self.sub_dir = sub_dir
        self.file_name = file_name
        #self.file_ext = 'h5'
        
        # data order inside the class should be unified
        # any manipulation of dimeisons should be done right before it is needed
        # also, when loaded we should ensure that the dimensions are respected
        # projection data: [Energy][Pixel][Slice][Angle]
        # reconstuction data: [X][Y][Slice][Energy]
        # This how it seems to be done now but we can consider change that to make Energey and Slice at the same dimension for both projection & reconstruction
        self.data = None  # consider make this protected
        self.z_load_slice = slice(None,None,None) # for partial (random access) loading
        self.load_channels = slice(None,None,None) # for partial (random access) loading
        self.load_angles = slice(None,None,None) # for partial (random access) loading
        self.processed_dir = 'processed' # put everything under 'processed' directory

    def setLoadSliceZ(self, start, stop, step=None):
        if stop <0:
            return 
        self.z_load_slice = slice(start, stop, step)
        
    def setLoadChannels(self, start, stop, step=None):
        if stop <0:
            return 
        self.load_channels = slice(start, stop, step)
        
    def getDirPath(self):
        return self.datapath_head + os.path.sep + self.processed_dir + os.path.sep + self.sub_dir 
        
    def getFilePath(self, file_ext):
        return self.getDirPath() + os.path.sep + self.file_name + "." + file_ext

    #def useH5(self):
    #    self.file_ext = 'h5'

    #def useMat(self):
    #    self.file_ext = 'mat'

    def loadData(self, file_ext):
        if file_ext is None:
           file_ext = "h5"
	
        if "h5" in file_ext:
            try:
                self.loadDataH5(self.getFilePath(file_ext))
            except:
                print(self.getFilePath(file_ext))
                warnings.warn('Cound not open file, trying .mat')
                #self.loadDataMat()
                
        elif "mat" in file_ext:
            self.loadDataMat()
        else:
            raise ValueError('File extension is not recognized.')

        print (self.__class__.__name__ + "  loaded dims" + str(self.data.shape))
        
            #-------------------------------checked
    def loadDataH5Silce(self, value):
        
        self.data = numpy.array(value[self.load_channels,self.z_load_slice,:], order='F').transpose()
        
    
    def loadDataH5(self, data_path):
        print ('loading: '+data_path)
        print("here...")
        f = h5py.File(data_path,'r')
        print(f)
        value = f['data']['value']
        print ("file   data shape: " + str(value.shape))
        #raise SystemExit
        #self.data = numpy.array(value[:,self.z_load_slice,:], order='F').transpose()
        self.loadDataH5Silce(value)
        #self.data = numpy.array(value[:,1:5,:], order='F').transpose()
        #self.data = self.data.reshape((1,) + self.data.shape)
        #self.data = numpy.array(value[:,self.z_load_slice,:,:], order='F').transpose()
        print ("loaded data shape: " + str(self.data.shape))
        print("THIS WAS THE SHAPE OF H5 DATA LOADED")
        f.close()
        return True
        
    def saveData(self, file_ext = None):
        minVal = numpy.nanmin(self.data)
        maxVal = numpy.nanmax(self.data)
        print("saveData minVal: " + str(minVal))
        print("saveData maxVal: " + str(maxVal))
        if file_ext is None:
           file_ext = "h5"
                   
        dir_path = self.getDirPath()
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
                
        if "h5" in file_ext:
            self.saveDataH5(self.getFilePath(file_ext))
        elif "tiff" or 'png' in file_ext:
            dir_path_images = dir_path + '/' + self.file_name
            if not os.path.exists(dir_path_images):
               os.makedirs(dir_path_images)
            self.saveDataAsImages(dir_path_images, file_ext)
        else:
            raise ValueError('File extension is not recognized.')
            
    def saveDataH5(self, data_path):
        f = h5py.File(data_path,'w')
        data_group = f.create_group('data')
        data_group.create_dataset('value', data=self.data.transpose())  
        f.close()

    #@abc.abstractmethod
    #TODO: consider using abc package to define functions as abstract
    def loadDataMat(self):
        raise NotImplementedError("Please Implement this method")
   
    # The following methods has to be abstract now because reconstruction data has a different dimensionality 
    def convertTo4D(self):
        raise NotImplementedError("Please Implement this method")

    def selectSlices(self, slices):
        raise NotImplementedError("Please Implement this method")

    def selectChannels(self, channels):
        raise NotImplementedError("Please Implement this method")
        
    def averageChannels(self):
        raise NotImplementedError("Please Implement this method")
    
    def removeNans(self):
        raise NotImplementedError("Please Implement this method")
        
    def __del__(self):
        del self.data
        
    def saveImageAsTiff(self, channelPath, I, sliceIndex):
        fname = channelPath + os.path.sep + ("image_z%04d.tif" % (sliceIndex)) 
        sliceFile = TIFF.open(fname, 'w')
        sliceFile.write_image(I.astype(numpy.float16))
        sliceFile.close()
        
    def saveImageAsGreyScale(self, channelPath, I, sliceIndex):
        #self.i = self.i+1
        fname = channelPath + os.path.sep  + ("image_z%04d.png" % (sliceIndex) ) 
        #fname = channelPath + os.path.sep + '_'+ ("image_z%04d_%01d.png" % (sliceIndex, self.i) ) 
        #I8 = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(numpy.uint8)
        #I = I.clip(min=0) 
        #I8 = (I*255.9).astype(numpy.uint8)
        I8 = (((I - I.min()) / (I.max() - I.min())) * 256).astype(numpy.uint8)
        I8 = 255-I8
        img = Image.fromarray(I8)
        img.save(fname)
        
    def show_image_ch_basic(self, ax, plot_slice_id, ch, plot_max):
        
        myplt = ax.imshow(self.data[:,:,plot_slice_id,ch].squeeze(), cmap='gist_yarg',norm=mcolors.PowerNorm(gamma=0.6))
        #for spine in ax.spines.values():
        #    if ch == 8 or ch == 20:
        #        spine.set_edgecolor('red')
        myplt.set_clim(0,plot_max)        
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.colorbar(myplt, ax=ax)
        #ax.axis('off')
        #generate_patches(patches, patches_colors, ax)
        return myplt


    def show_image_ch_mont(self, ax, plot_slice_id, ch, plot_max):
        myplot = self.show_image_ch_basic(ax, plot_slice_id, ch, plot_max)
        #sub_text = 'TV = ' + str(int(image_data.TV[plot_slice_id,ch])) 
        #sub_text = sub_text + '\n' + 'MAE = ' + str(int(100*image_data.MAE[plot_slice_id,ch])) 
        #sub_text = sub_text + '\n' + 'SSIM = ' + str(int(100*image_data.SSIM[plot_slice_id,ch])) 
        #ax.set_xlabel(sub_text)
        ax.set_visible(True)
        return myplot
        
    def plot_images_montage(self, axs, row_no, plot_slice_id, plot_max, kevs_used, ch_step=4):
        #image_data.data = image_data.data[2:98,2:98,:,:]
        ch= self.data.shape[3]
        #print(image_data.data.shape)
        #print(ch)
        
        fig_counter = 0
        for ch_i in itertools.islice(itertools.count(),0,ch,ch_step):
            #print(ch_i)
            ax = axs[row_no,fig_counter]
           
            #ax = plt.subplot(fig_rows, fig_cols, fig_counter)
            energy = str(kevs_used[ch_i]) + ' keV'
            myplot = self.show_image_ch_mont(ax, plot_slice_id, ch_i, plot_max)
            if fig_counter==0:
                if not hasattr(self, 'Lname'):
                    self.Lname = ''
                ax.set_ylabel(self.Lname, fontsize=12)
            if row_no==0:
                ax.set_title(energy, fontsize=12)
            fig_counter+=1
            
        return myplot
    def save_images_montage_slice(self, plot_slice_id, kevs_used):
        ch_step = 1
        #ch = 15#30
        #imagenr = plot_slice_id
          
        fig_rows = 2
        fig_cols = 32
        
        #fig_data = plt.figure()
        fig, axs = plt.subplots(fig_rows,fig_cols, dpi=100)
        slice_text  = ("_slice_z%04d" % (plot_slice_id))
        fig.suptitle('slice: '+slice_text, y=0.1)
        fig_data = fig
        fig.set_figheight(2)
        fig.set_figwidth(40)
        numpy.vectorize(lambda axs:axs.set_visible(False))(axs)
        p_min, plot_max = numpy.percentile(self.data[:,:,plot_slice_id,:], (5, 99.9))
        myplot = self.plot_images_montage(axs, 0, plot_slice_id, plot_max, kevs_used, ch_step=ch_step)
        fig.colorbar(myplot, ax=axs.ravel().tolist(), fraction=0.046, pad=0.04)
        #dir_path = self.getDirPath()
        #save_path = dir_path+os.path.sep+self.Lname+slice_text+'.png'
        #print(save_path)
        
        #fig.savefig(save_path, bbox_inches='tight')
        
        fig_data.canvas.draw()
        plot_image = numpy.frombuffer(fig_data.canvas.tostring_rgb(), dtype=numpy.uint8)
        print(plot_image.shape)
        plot_image = plot_image.reshape(fig_data.canvas.get_width_height()[::-1] + (3,))
        print(plot_image.shape)
        return plot_image
        
    def save_images_montage(self):
        slice_no = self.data.shape[2]
        #slice_no = 2
        print(slice_no)
        
        ch_no = self.data.shape[3]
        kevs = numpy.round(numpy.linspace(start=20, stop=160, num=128),decimals=1)
        channels_used = list(numpy.linspace(start=self.load_channels.start, stop=self.load_channels.stop, num=ch_no).astype(int))
        kevs_used = kevs[channels_used]
        
        plot_images = list()
        for slice_i in itertools.islice(itertools.count(),0,slice_no,1):
            plot_image = self.save_images_montage_slice(slice_i, kevs_used)
            plot_images.append(plot_image)
            
        fps = 100
        imageio.mimwrite(self.getDirPath()+ '/reconstruction_movie' + '.gif', plot_images, fps=fps)
        

class ProjectionData(CTData):

      def reduceProjNo(self, new_no, geostruct):
          projection_resolution = geostruct["range_angle"]/geostruct["nproj"]; # angles between consecutive projections before reduction
          projection_resolution_new = geostruct["range_angle"]/new_no;
          sampling_step = projection_resolution_new/projection_resolution;
 
          new_indices = [0] * new_no
          for j in range(new_no):
              new_indices[j] = int(sampling_step*j)
     
          #print(new_indices)
          #new_indices = round(new_indices);
          #print(new_indices)
  
          self.data = self.data[:,:,:,new_indices]
          geostruct["nproj"] = new_no

      def setLoadAngles(self, geostruct, ang_no, ang_start=0, ang_sep=None):
        if ang_sep == None:
          projection_resolution = geostruct["range_angle"]/geostruct["nproj"] # angles between consecutive projections before reduction
          print("projection_resolution",projection_resolution)
          projection_resolution_new = geostruct["range_angle"]/ang_no
          print("projection_resolution_new",projection_resolution_new)
          ang_sep = int(projection_resolution_new/projection_resolution)
          print("ang_sep",ang_sep)
          
        #ang_start = 1;
        #ang_sep = 15;
        #ang_no = 12;
        ang_span = ang_sep*(ang_no-1)+ang_start
        ang_span = int(ang_span)
        #ang_start = 359-ang_span
        #ang_span = 359
        self.load_angles = slice(ang_start, ang_span+1, ang_sep)
        geostruct["nproj"] = ang_no
        geostruct["range_angle"] = ang_span
        print("load_angles",self.load_angles)
        print(geostruct)
      
      def loadDataH5Silce(self, value):
        self.data = numpy.array(value[self.load_angles, self.z_load_slice,:,self.load_channels], order='F').transpose()
        
      def convertTo4D(self):
          if len(self.data.shape) == 3: # 2D data
             self.data = self.data.reshape((self.data.shape[0], self.data.shape[1], 1 , self.data.shape[2]))
             print(self.__class__.__name__ + " coverted to 4D, new dims" + str(self.data.shape))

      def selectSlices(self, slices = [200]):
          if set(slices).issubset(set(range(0, self.data.shape[2]))):
             self.data = self.data[:,:,slices,:]

      def selectChannels(self, channels = [63]):
          if set(channels).issubset(set(range(0, self.data.shape[0]))):
             self.data = self.data[channels,:,:,:]
          else: 
               raise SystemExit
               
      def averageChannels(self):
          self.data = self.data.mean(0, keepdims = True)
    
      def removeNansChris(self):
          # christian impelmentation
          # remove Nans and Infs
          nshape = self.data.shape
          valueCorrections = numpy.zeros(nshape[0],dtype=numpy.long)
          nanArray = numpy.isnan(self.data)
          infArray = numpy.isinf(self.data)
        
          for E_index in itertools.islice(itertools.count(),0,nshape[0]):
              mask = numpy.isfinite(self.data[E_index,:,:,:])
              maxVal = numpy.nanmax(self.data[E_index,:,:,:][mask])
              for D_index in itertools.islice(itertools.count(),0,nshape[1]):
                  for Z_index in itertools.islice(itertools.count(),0,nshape[2]):
                      for P_index in itertools.islice(itertools.count(),0,nshape[3]):
                          if infArray[E_index,D_index,Z_index,P_index]==True:
                             #self.data[E_index,D_index,Z_index,P_index]=-2000
                             self.data[E_index,D_index,Z_index,P_index]=maxVal
                             valueCorrections[E_index]+=1
                          if nanArray[E_index,D_index,Z_index,P_index]==True:
                             self.data[E_index,D_index,Z_index,P_index]=0
                             valueCorrections[E_index]+=1 

      def removeNans(self):
          minVal = numpy.nanmin(self.data)
          maxVal = numpy.nanmax(self.data)
          print("before removeNans minVal: " + str(minVal))
          print("before removeNans maxVal: " + str(maxVal))
          # remove Nans and Infs
          nshape = self.data.shape        
          for E_index in itertools.islice(itertools.count(),0,nshape[0]):
              mask = numpy.isfinite(self.data[E_index,:,:,:])
              maxVal = numpy.nanmax(self.data[E_index,:,:,:][mask])
              #minVal = numpy.nanmin(self.data[E_index,:,:,:][mask])
              nanArray = numpy.isnan(self.data[E_index,:,:,:])
              infArray = numpy.isinf(self.data[E_index,:,:,:])
              #negInfArray = numpy.isneginf(self.data[E_index,:,:,:])
              self.data[E_index,:,:,:][nanArray] = 0
              self.data[E_index,:,:,:][infArray] = maxVal 
              #self.data[E_index,:,:,:][negInfArray] = minVal 
          
          self.data[self.data < 0.0] = 0
          minVal = numpy.nanmin(self.data)
          maxVal = numpy.nanmax(self.data)
          print("after removeNans minVal: " + str(minVal))
          print("after removeNans maxVal: " + str(maxVal))

      def plotAsImage(self):
          import pylab
          pylab.figure(1)
          pylab.imshow(self.data[10,200,:,:].squeeze())
          pylab.show()
          
      def saveDataAsImages(self, data_path, file_ext):
          print (self.data.shape)
          numChannels = self.data.shape[0]
          numSlices = self.data.shape[2]
          for channelIndex in itertools.islice(itertools.count(),0,numChannels):
              channelPath = data_path+os.path.sep+("channel_%04d" % (channelIndex))
              if(os.path.exists(channelPath) == False):
                 os.mkdir(channelPath)
              for sliceIndex in itertools.islice(itertools.count(),0,numSlices):                  
                  I = self.data[channelIndex,:,sliceIndex,:]
                  if "tiff" in file_ext:
                      self.saveImageAsTiff(channelPath,I,sliceIndex)
                  if "png" in file_ext:
                      self.saveImageAsGreyScale(channelPath,I,sliceIndex)

class RawData(ProjectionData):

      def __init__(self, datapath_head, file_name = "raw"):
          super(RawData,self).__init__(datapath_head = datapath_head, sub_dir="raw", file_name = file_name)

      def loadData(self, file_ext = None):
          super(RawData,self).loadData(file_ext)
          #print ("corrected data loaded dims" + str(self.data.shape))
          self.convertTo4D()
          
          
      def getNumberOfSlices(self, filepath):
        import struct
        #nlines=1
        #nslices=1
        #nlinesPerSlice=1
        with open(filepath, "rb") as binary_file:
            binary_file.seek(60);
            nlines_bytes = binary_file.read(4)
            #nslices = int.from_bytes(nslices_bytes, signed=True)
            nlines = struct.unpack("@I",nlines_bytes)[0]
        return nlines
    
      def loadDataMultix(self,geostruct):
        import glob
        import re
        #import loadRawData_Cstyle
        from dataStructures import loadRawData_Cstyle
        
        fnames = glob.glob1(self.datapath_head, "*.bin")
        if(len(fnames)<2):
            exit(1);
            
          # sort the names
        digits = re.compile(r'(\d+)')
        def tokenize(filename):
            return tuple(int(token) if match else token
                         for token, match in
                         ((fragment, digits.search(fragment))
                          for fragment in digits.split(filename)))
        
        # Now you can sort your file names like so:
        fnames.sort(key=tokenize)
            
        nproj = int(geostruct["nproj"])
        nenergy = int(128)
        ndet = int(geostruct["ndet"])
        nslices = 1
        #==> ???? nslices = ???? <===#
        nlines = self.getNumberOfSlices(self.datapath_head+os.path.sep+fnames[0])
        
        linesPerSlice=1
        linesPerSlice = nlines
        #if(geostruct.has_key("nSliceLines")):
        if not ('vol' in geostruct):
            linesPerSlice = nlines
        elif 'nSliceLines' in geostruct:
          if(geostruct["vol"] and (geostruct["nSliceLines"]!=None)):
              geostruct["nSliceLines"]=int(geostruct["nSliceLines"])
              linesPerSlice = geostruct["nSliceLines"]
              
        if(linesPerSlice>1):
            nslices = int(nlines / linesPerSlice)
        else:
            nslices = nlines
        # we need to get the number of slices to resize the field correctly. Any way to determine this upfront ? #
        self.data = numpy.zeros((nenergy,ndet,nslices,nproj+1), dtype=numpy.double, order="F")
        
        #raw_data = octave.load_raw_tomography(datapath_head,geostruct)
        #raw_data,data_param = octave.feval("load_raw_tomography", datapath_head, geostruct, nout=2)
        #rather replicate 'load_raw_data' file in python, working on file-by-file basis
        volData = False
        if(geostruct["vol"]>0):
            volData=True
        #get all files with .bin ending in folder 'datapath_head'
        
        for i in range(0, len(fnames)):
            ## OCTAVE SOLUTION ##
            #loaded = octave.read_multix_bin_files(datapath_head, fnames[i])
            #raw_data[:,:,:,i]=octave.process_multi_lines(loaded["mltdata"], loaded["DataPara"]["NumIntTime"],volData)
            
            ## PYTHON NATIVE SOLUTION
            reader = loadRawData_Cstyle.loadRawData_Cstyle()
            ##reader.read_multix_bin_file(os.path.join(datapath_head, fnames[i]), linesPerSlice)
            ##reader.process_multi_lines(True)
            ##raw_data[:,:,:,i]=reader.getRawData()
            #
            print(nslices)
            print(linesPerSlice)
            ret = reader.read_multix_bin_file(os.path.join(self.datapath_head, fnames[i]))
            if ret==False:
                print ("Error reading file '"+fnames[i]+"'. EXITING ...")
                exit(1)
            reader.process_multi_lines(True)
            
            reader.average_lines(nslices, linesPerSlice)
            
#            if(linesPerSlice>1):
#                cnt=0
#                binVal = linesPerSlice
#                lineData = reader.getRawData()
#                dims = lineData.shape
#                for sliceIndex in itertools.islice(itertools.count(),0,nslices):
#                    stIndex = cnt
#                    edIndex = min(cnt+binVal, dims[2])
#                    #raw_data[:,:,sliceIndex,i] = numpy.squeeze(numpy.median(reader.getRawData()[:,:,stIndex:edIndex], axis=2))
#                    raw_data[:,:,sliceIndex,i] = numpy.nanmean(lineData[:,:,stIndex:edIndex], axis=2, keepdims=False)
#                    cnt = cnt+binVal
#            else:
#                raw_data[:,:,:,i]=reader.getRawData()
            #raw_data[:,:,:,i]=reader.getRawData()
            data_avg = reader.getAveragedData_FortranOrder()
            print ("data_avg.shape: ", data_avg.shape)
            import pylab
            pylab.figure(1)
            pylab.imshow(data_avg.squeeze())
            pylab.show()
            self.data[:,:,:,i]=data_avg
            print ("read projection %d ..." % (i))
        #raw_data = numpy.squeeze(raw_data)



class CorrectedData(ProjectionData):

      def __init__(self, datapath_head, file_name = "corrected"):
          super(CorrectedData,self).__init__(datapath_head = datapath_head, sub_dir="corrected", file_name = file_name)

      def loadData(self, file_ext = None):
          super(CorrectedData,self).loadData(file_ext)
          #print ("corrected data loaded dims" + str(self.data.shape))
          self.convertTo4D()
          #start = time.time()
          #self.data = numpy.array(numpy.transpose(self.data, [0,1,3,2])) # TODO: why we do this here?? If needed for correction do the transpose right bef
          #shape = self.data.shape
          #print(self.data.shape)
          #import pylab
          #pylab.gray()
          #pylab.figure(1)
          #print(self.data.shape)
          #print(self.data[40,:,:,0].shape)
          #pylab.imshow(self.data[40,:,:,100].squeeze())
          #pylab.show()
          #end = time.time()
          #print("transpose: "+str(end - start))
          #print ("corrected data new dims" + str(self.data.shape))

      def loadDataMat(self):
          self.data = octave.load_corrected_data(self.datapath_head)


class SinogramData(ProjectionData):

      def __init__(self, datapath_head, file_name = "sinogram"):
          super(SinogramData,self).__init__(datapath_head = datapath_head, sub_dir="sinogram", file_name = file_name)
      
      def loadData(self, file_ext = None):
          super(SinogramData,self).loadData(file_ext)
          self.convertTo4D()
          start = time.time()
          #self.removeNans()
          end = time.time()
          #print("remove Nans: "+str(end - start))
          minVal = numpy.nanmin(self.data)
          print("minVal: " + str(minVal))

      def loadDataMat(self):
          #print("loadDataMat: ")
          #self.data = octave.load_sinogram_data(self.datapath_head)
          self.data = octave.load_data(self.datapath_head, 'sinogram', 'sinogram');
          minVal = numpy.nanmin(self.data)
          print("minVal: " + str(minVal))
          #print( self.data.dtype)
          
      def compute(self,ProjectionData):
          #corrected_data = numpy.array(numpy.transpose(corrected_data, [0,1,3,2]))
          #print "correction shape after permutation: "+str(corrected_data.shape)  #should be: <e><d><p><s>
          sinoShape = (ProjectionData.data.shape[0], ProjectionData.data.shape[1], ProjectionData.data.shape[2], ProjectionData.data.shape[3]-1)
          self.data = numpy.zeros(sinoShape, dtype=numpy.double, order="F")
          
          for sliceIndex in itertools.islice(itertools.count(),0,ProjectionData.data.shape[2]):
              data_slice = ProjectionData.data[:,:,sliceIndex,:]
              if sliceIndex == 0:
                 print ("Slice shape: "+str(data_slice.shape))
              #sliceShape = (slice.shape[0], slice.shape[1], 1, slice.shape[2]-1)
              #slice_reshaped = numpy.squeeze(slice) # - here in Python, a mid-index field is already squeezed
              #sinogram_slice = octave.compute_sinograms(slice_reshaped)
              #sinogram_slice = numpy.expand_dims(sinogram_slice, 2)
              #sinogram_data[:,:,sliceIndex,:] = numpy.reshape(sinogram_slice, sliceShape, 'F')
              sinogram_slice = octave.compute_sinograms(data_slice)
              #sinogram_slice = octave.compute_sinograms3D(slice)
              #sinogram_slice = numpy.reshape(sinogram_slice, slice.shape, order='F')
              #print sinogram_slice.shape
              #flat_field = data_slice[:,:,0];
              #flat_field_rep = numpy.repeat(flat_field[:, :, numpy.newaxis], data_slice.shape[2]-1, axis=2)
              #print(flat_field_rep.shape)
              #data_slice_norm = data_slice[:,:,1:]/flat_field_rep
              #sinogram_slice = -numpy.log(1e-8+data_slice_norm)
              self.data[:,:,sliceIndex,:] = sinogram_slice
           
      def add_nooise(self, n=0):
          #print(self.data.shape[1:])
          #raise SystemExit
          data_ = numpy.copy(self.data)
          ##self.data[[8,20],:,:,:] = self.data[[8,20],:,:,:] + numpy.random.normal(0,n, size=(2,self.data.shape[1],self.data.shape[2],self.data.shape[3]))
          data_[[8,20],:,:,:] = self.data[[8,20],:,:,:] + numpy.random.normal(0,n, size=(2,self.data.shape[1],self.data.shape[2],self.data.shape[3]))
          return data_
  
class ImageData(CTData):
    
      def averageChannels(self):
          self.data = self.data.mean(3, keepdims = True)
        
      def dumpAllImagesInDir(self, dir_path):
          if(os.path.exists(dir_path) == False):
              os.mkdir(dir_path)
          numChannels = self.data.shape[3]
          for channelIndex in itertools.islice(itertools.count(),0,numChannels):
              #channelPath = data_path+os.path.sep+("channel_%04d" % (channelIndex))
              numSlices = self.data.shape[2]
              for sliceIndex in itertools.islice(itertools.count(),0,numSlices): 
                  fname = dir_path+os.path.sep+self.file_name+'_'+str(channelIndex).zfill(3)+'_'+str(sliceIndex).zfill(3)+'.png'
                  I = self.data[:,:,sliceIndex,channelIndex]
                  I8 = (((I - I.min()) / (I.max() - I.min())) * 256).astype(numpy.uint8)
                  I8 = 255-I8
                  img = Image.fromarray(I8)
                  img.save(fname)
                  
      def saveDataAs2DImages(self, file_ext, channelIndex, channelPath):
          numSlices = self.data.shape[2]
          for sliceIndex in itertools.islice(itertools.count(),0,numSlices):                  
              I = self.data[:,:,sliceIndex,channelIndex]
              if "tiff" in file_ext:
                  self.saveImageAsTiff(channelPath,I,sliceIndex)
              if "png" in file_ext:
                  self.saveImageAsGreyScale(channelPath,I,sliceIndex)
      def saveDataAs3DImage(self, data_path, channelIndex):     
            fname = data_path + os.path.sep + ("image_channel%04d.mhd" % (channelIndex))
            dimArray = numpy.array([self.data.shape[0], self.data.shape[1], self.data.shape[2]], dtype=numpy.uintc)
            spaceArray = numpy.ones(3, dtype=numpy.float32)
            arrayLen = self.data.shape[0]*self.data.shape[1]*self.data.shape[2]
            selectedArray = numpy.reshape(self.data[:,:,:,channelIndex], arrayLen, 'F')
#            sliceFile = mhd.MhdFile()
#            sliceFile.SetDimensions(dimArray)
#            sliceFile.SetSpacing(spaceArray)
#            sliceFile.setDataAsDouble(selectedArray.astype(numpy.double))
#            sliceFile.setFilename(fname)
#            sliceFile.writeFile()
                      
      def saveDataAsImages(self, data_path, file_ext):
          print (self.data.shape)
          numChannels = self.data.shape[3]
          #numSlices = self.data.shape[2]
          for channelIndex in itertools.islice(itertools.count(),0,numChannels):
              if "tiff" in file_ext or "png" in file_ext:
                  channelPath = data_path+os.path.sep+("channel_%04d" % (channelIndex))              
                  if(os.path.exists(channelPath) == False):
                     os.mkdir(channelPath)
                  self.saveDataAs2DImages(file_ext, channelIndex, channelPath)
              if "mhd" in file_ext:
                  self.saveDataAs3DImage(data_path, channelIndex)    
                  
      def compute_TV(self):
          numSlices = self.data.shape[2]
          numChannels = self.data.shape[3]
          
          self.TV = numpy.zeros(shape=(numSlices,numChannels))
          self.TV_s = numpy.zeros(numSlices)
          for sliceIndex in itertools.islice(itertools.count(),0,numSlices):
              #TV_c = numpy.zeros(numChannels)
              for channelIndex in itertools.islice(itertools.count(),0,numChannels):
                  image = self.data[:,:,sliceIndex,channelIndex]
                  g_x, g_y = numpy.gradient(image)
                  g_norm2 = g_x**2 + g_y**2 
                  TV_ = numpy.sum(numpy.sqrt(g_norm2))
                  self.TV[sliceIndex,channelIndex] = TV_
                  
                  #print(w.shape)
                  #import matplotlib.pyplot as plt
                  #print (image.shape)
                  #plt.figure()
                  #plt.imshow(image, cmap='gray')
                  #plt.show()
              TV_c_norm2 = self.TV[sliceIndex,:]**2
              self.TV_s[sliceIndex] = numpy.sqrt(numpy.sum(TV_c_norm2))
              
      def compute_error(self, ref):
          print(self.data.shape)
          print(ref.data.shape)
          assert (self.data.shape == ref.data.shape),"shape should be equal in both sets!"
          numSlices = self.data.shape[2]
          numChannels = self.data.shape[3]
          dist = numpy.linalg.norm(self.data-ref.data)
          print(dist)
          #self.TV = numpy.zeros(shape=(numSlices,numChannels))
          #self.TV_s = numpy.zeros(numSlices)
          self.SSIM = numpy.zeros(shape=(numSlices,numChannels))
          self.SSIM_s = numpy.zeros(numSlices)
          self.MAE = numpy.zeros(shape=(numSlices,numChannels))
          self.MAE_s = numpy.zeros(numSlices)
          for sliceIndex in itertools.islice(itertools.count(),0,numSlices):
              #TV_c = numpy.zeros(numChannels)
              for channelIndex in itertools.islice(itertools.count(),0,numChannels):
                  image = self.data[:,:,sliceIndex,channelIndex]
                  image_ref = ref.data[:,:,sliceIndex,channelIndex]
                  #g_x, g_y = numpy.gradient(image)
                  #g_norm2 = g_x**2 + g_y**2 
                  #TV_ = numpy.sum(numpy.sqrt(g_norm2))
                  #self.TV[sliceIndex,channelIndex] = TV_
                  #self.MAE[sliceIndex,channelIndex] = numpy.linalg.norm(image-image_ref,ord=1)
                  self.MAE[sliceIndex,channelIndex] = mae(image,image_ref)
                  self.SSIM[sliceIndex,channelIndex] = ssim(image,image_ref)
                  
                  #print(w.shape)
                  #import matplotlib.pyplot as plt
                  #print (image.shape)
                  #plt.figure()
                  #plt.imshow(image, cmap='gray')
                  #plt.show()
              #TV_c_norm2 = self.TV[sliceIndex,:]**2
              #TV_c_norm1 = abs(self.TV[sliceIndex,:])
              #self.TV_s[sliceIndex] = numpy.sqrt(numpy.sum(TV_c_norm2))
              self.MAE_s[sliceIndex] = numpy.mean(self.MAE[sliceIndex,:])
              self.SSIM_s[sliceIndex] = numpy.mean(self.SSIM[sliceIndex,:])
              #self.TV_s[sliceIndex] = numpy.sum(TV_c_norm1)
              #print(TV_c)
          #print(self.TV_s)
          #print(self.TV)
          #raise SystemExit
      #return TV_c, TV_s
      def get_metric(self, error_type, slice_id):
          if error_type == 'TV':
              return numpy.log(self.TV_s[slice_id]), numpy.log(self.TV[slice_id,:].squeeze())
          elif error_type == 'MAE':
              return self.MAE_s[slice_id], self.MAE[slice_id,:].squeeze()
              #return -numpy.log(self.MAE_s[slice_id]), -numpy.log(self.MAE[slice_id,:].squeeze())
          elif error_type == 'SSIM':
              #return -numpy.log(self.SSIM_s[slice_id]), -numpy.log(self.SSIM[slice_id,:].squeeze())
              return self.SSIM_s[slice_id], self.SSIM[slice_id,:].squeeze()
          

class ReconstructionData(ImageData):

      def __init__(self, datapath_head, file_name = "reconstruction"):
          super(ReconstructionData,self).__init__(datapath_head = datapath_head, sub_dir="reconstructed", file_name = file_name)
      
      def rearrange_data_dl(self):
          #print("rearrange_data_dl:")
          #print(self.data.shape)
          #self.data = self.data.transpose((2, 0, 1, 3))
          #print(self.data.shape)
          #self.data = self.data.transpose()
          #print(self.data.shape)
          self.data = self.data.transpose((3, 1, 0, 2))
          #print(self.data.shape)
          #raise SystemExit
      #def loadDataH5Silce(self, value):
          #self.data = numpy.array(value[:,self.z_load_slice,:,:], order='F').transpose()
      # def loadDataH5(self, data_path):
      #     f = h5py.File(data_path,'r')
      #     self.data = numpy.array(f['data']['value'], order='F').transpose()
      #     f.close()
      #     return True
      
      def saveDataForDL(self,save_dir):
          self.rearrange_data_dl()
          self.datapath_head = save_dir
          self.sub_dir = 'images/'
          self.saveData()
          
      def setLegendInfo(self, name, color, style='-'):
          self.Lname = name
          self.Lcolor = color
          self.Lstyle = style
      
class LabelData(CTData):
      def __init__(self, datapath_head, file_name = "segmented"):
          super(LabelData,self).__init__(datapath_head = datapath_head, sub_dir="segmented", file_name = file_name)
          
      #def getDirPath(self):
        #return self.datapath_head + os.path.sep + "manualSegmentation" + os.path.sep 
    
      def loadData(self, file_ext = None):
          super(LabelData,self).loadData(file_ext)
          
      def loadDataH5Silce(self, value):
          self.data = numpy.array(value[self.z_load_slice,:,:], order='F').transpose()
          
      def saveDataAsImages(self, data_path, file_ext):
          print (self.data.shape)
          print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!shape printed")
          self.data = self.data +1
          #numChannels = self.data.shape[0]
          numSlices = self.data.shape[2]
          #for channelIndex in itertools.islice(itertools.count(),0,numChannels):
              #channelPath = data_path+os.path.sep+("channel_%04d" % (channelIndex))
              #if(os.path.exists(channelPath) == False):
                 #os.mkdir(channelPath)
          for sliceIndex in itertools.islice(itertools.count(),0,numSlices):   
                              
                  I = self.data[:,:,sliceIndex]
                  if "tiff" in file_ext:
                      self.saveImageAsTiff(data_path,I,0)
                  if "png" in file_ext:
                      self.saveImageAsGreyScale(data_path,I,sliceIndex)
          
      def loadLabelMap(self):
          file = self.datapath_head + "/processed/segmented/"+'label_map.txt'
          with open(file) as f:
            content = f.readlines()
            #print(content)
            
            self.label_names = [''] *  len(content)
            self.label_ids = [0] *  len(content)
            for seg_id in itertools.islice(itertools.count(), 0, len(content)):
                content = [x.split('\n')[0] for x in content]    
                contents = content[seg_id].split(' ')
                #label_name = content[seg_id].split(' ')[1]
                self.label_ids[seg_id] =  int(contents[0])
                self.label_names[seg_id] = contents[1]
                 #content = [x.split(':')[0] for x in content]
                 #content = [x.split('\n')[0] for x in content]             
                 #return content
            print(self.label_names)
            print(self.label_ids)
            
            #----------------------------------------------------------not used
      def extractSegmentValues(self, reconstruction_data):
        ch_no = reconstruction_data.data.shape[3]
        
        self.data_all = numpy.zeros((1,ch_no))
        self.label_all = ['None']
        self.data_mean = numpy.zeros((len(self.label_ids),ch_no))
        self.data_std = numpy.zeros((len(self.label_ids),ch_no))
        for seg_id in itertools.islice(itertools.count(), 0, len(self.label_ids)):
            print("label_id is:")
            print(self.label_ids[seg_id]+1)
            x, y, z = numpy.where(self.data==(self.label_ids[seg_id]+1))
            #print(x,y,z)
            print(reconstruction_data.data.shape)
            recons_data_seg = reconstruction_data.data[x,y,z,:]
            recons_data_seg = recons_data_seg.squeeze()
            print(recons_data_seg.shape)
            #recons_data_seg = recons_data_seg.transpose()
            self.data_mean[seg_id,:] = numpy.mean(recons_data_seg, axis = 0)
            self.data_std[seg_id,:] = numpy.std(recons_data_seg, axis = 0)
            self.data_all = numpy.vstack((self.data_all, recons_data_seg))
            self.label_all = numpy.hstack((self.label_all, [self.label_names[seg_id]]*recons_data_seg.shape[0]))
            print("------------1234-------------")
            
            print(recons_data_seg.shape)
        self.data_all = self.data_all[1:,]
        self.label_all = self.label_all[1:]
        
         #-------------------------------------------------------------not used
      def saveExtractedValue(self):
        numpy.savetxt(self.datapath_head + "/processed/segmented/"+"LAC_all.csv", self.data_all, delimiter=",")
        numpy.savetxt(self.datapath_head + "/processed/segmented/"+"labels_all.txt", self.label_all, delimiter=" ", fmt="%s")
        numpy.savetxt(self.datapath_head + "/processed/segmented/"+"LAC_mean.csv", self.data_mean, delimiter=",")
        numpy.savetxt(self.datapath_head + "/processed/segmented/"+"LAC_std.csv", self.data_std, delimiter=",")
        numpy.savetxt(self.datapath_head + "/processed/segmented/"+"labels_mean.txt", self.label_names, delimiter=" ", fmt="%s")

class SynthImages(ImageData):

      def __init__(self, datapath_head, file_name = "images"):
          super(SynthImages,self).__init__(datapath_head = datapath_head, sub_dir="images", file_name = file_name)
 
      def loadData(self, file_ext = None):
          super(SynthImages,self).loadData(file_ext)
          
      def loadDataH5(self, data_path):
          f = h5py.File(data_path,'r')
          self.data = numpy.array(f['data']['value'], order='F').transpose((1, 2, 0, 3))
          f.close()
          return True
      
      def getDirPath(self):
          print("getDirPath")
          print("self.datapath_head :" + self.datapath_head)
          print("self.sub_dir :" + self.sub_dir)
          print("self.file_name :" + self.file_name)
          print (self.datapath_head + os.path.sep + self.sub_dir +  self.file_name)
          return self.datapath_head + os.path.sep + self.sub_dir +  self.file_name
