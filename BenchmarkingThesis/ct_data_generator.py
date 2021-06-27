#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 14:33:07 2018

@author: wamus
"""
import numpy as np
import sys
import astra
import itertools


class CTDataGenerator(object):
    """CTDataGenerator generates training and testing data with variations in CT acquisition."""
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.proj_no = self.sim_conf['proj_no']
        self.angles = np.linspace(0., 2*np.pi, self.proj_no , endpoint=False)
        #print(self.angles)
        self.det_width = 0.8
        self.det_count = 258 # 256 + 2(gap) 
        self.source_origin = 575 # 575*3 TODO: doubl check the exact value!
        self.origin_det = 575
        
        #M = (self.origin_det+self.source_origin)/self.source_origin; 
        #det_width = self.det_width*self.det_count;
        #scale = M/det_width
        #print ('rescaling geometry: ' + str(scale))
        
        #rescaling stuff
        #self.source_origin = scale*self.source_origin;
        #self.origin_det = scale*self.origin_det;
        #self.det_width = self.det_width*scale;
        
        #self.setProjGeom()

            
    def setProjGeom(self):         
        if self.sim_conf['geo_type'] == 'parallel':
            self.proj_geom = astra.create_proj_geom('parallel', self.det_width, self.det_count, self.angles)
        elif self.sim_conf['geo_type'] == 'fan':
            #self.proj_geom = astra.create_proj_geom('fanflat', self.det_width, self.det_count, self.angles, self.source_origin, self.origin_det)
            
            vectors = np.zeros((len(self.angles), 6)) 
            
            origin_detector = self.source_detector-self.source_origin
            for i in range(len(self.angles)):
                # source
                vectors[i,0] = self.source_origin*np.sin(self.angles[i]) + self.source_shift*np.sin(self.angles[i])
                vectors[i,1] = -self.source_origin*np.cos(self.angles[i]) + self.source_shift*np.cos(self.angles[i]);
                
                # center of detector
                vectors[i,2] = -origin_detector*np.sin(self.angles[i]) - self.detector_shift*np.cos(self.angles[i]);
                vectors[i,3] = origin_detector*np.cos(self.angles[i]) - self.detector_shift*np.sin(self.angles[i]);

                # vector from detector pixel (0,0) to (0,1)
                vectors[i,4] = np.cos(self.angles[i]) *self.det_width;
                vectors[i,5] = np.sin(self.angles[i]) *self.det_width;
            
            print("vectors.shape:")
            print(vectors.shape)
            self.proj_geom_vector = vectors;
            self.proj_geom = astra.create_proj_geom('fanflat_vec', self.det_count, vectors)
            return vectors

            
        else:
            sys.exit(" Geo type unknown")       
    
    def getDetCount(self):
        return self.det_count 
    
    def setGeometry(self, geo_struct):
        self.geo_struct = geo_struct
        self.source_origin = self.geo_struct["SAD"]*10
        self.source_detector = self.geo_struct["SDD"]*10
        self.origin_det = ((self.geo_struct["SDD"])-(self.geo_struct["SAD"]))*10
        self.source_shift = self.geo_struct["sourceCentShift"]*10
        self.detector_shift = self.geo_struct["detectCentShift"]*10
        self.det_width = self.geo_struct["pixel_size"]*10
        self.gap_no  = int(self.geo_struct["nElem"]) -1
        self.gap_size_pxs = round(self.geo_struct["Sep"]/self.geo_struct["pixel_size"])
        self.det_count = int(self.geo_struct["ndet"]) + self.gap_no*self.gap_size_pxs
        self.elem_pxs = int(self.geo_struct["ndet"])/int(self.geo_struct["nElem"])
        
        return self.setProjGeom()
        
        #print((geostruct["ndet"]))
        #(geostruct["det_space"]))
        #(geostruct["SAD"]))
        #(geostruct["SDD"]))
        #(geostruct["model"][0]))
        #(geostruct["detectCentShift"]))
        #(geostruct["sourceCentShift"]))
        #(geostruct["nElem"]))
        #(geostruct["Sep"]))
    
    def generateFromProjections(self, sinograms, image_size, object_scale):
        #image_size = [200,200];

        # input shape [Energy,Pixels,Slice,Angles]
        #print ('sinogram size: ')
        #print (sinogram['data'].shape)
        #sinogram = sinogram['data'].squeeze()'
        print("generateFromProjections")
        print(sinograms.shape)
        where_are_NaNs = np.isnan(sinograms)
        print('Nans:')
        print(where_are_NaNs.sum())
        where_are_inf = np.isinf(sinograms)
        print('Inf:')
        print(where_are_inf.sum())
        #sinogram = data.squeeze()
        #maxVal = np.nanmin(sinograms)
        #print("maxVal: " + str(maxVal))
        print(sinograms.dtype)
        
        
        #sinogram_mean = np.mean(sinogram[10:100,:,:], axis=0)

        
        #where_are_NaNs = np.isnan(sinogram_mean)
        #where_are_inf = np.isinf(sinogram_mean)
        
        #sinogram = sinogram[79,:,:]
        #sinogram = sinogram_mean
      
        
        #vol_geom = astra.create_vol_geom(100, 100) 
        
        #rec_id = astra.data2d.create('-vol', vol_geom)
        
        #self._reconstOneSino(sinogram, rec_id)
        self.images_tranformed, self.interpol_sinograms  = self._reconstruct_all(sinograms, image_size, object_scale)
        
        #astra.data2d.delete(rec_id)
        
    
    def _reconstruct_spectral(self, result, interpol_sinograms, slice_ind):
        
        for channel_ind in itertools.islice(itertools.count(), 0, interpol_sinograms.shape[0]):
                #print("j " + str(channel_ind))
                result[:,:,slice_ind,channel_ind] = self._reconstOneSino(interpol_sinograms[channel_ind,:,slice_ind,:])
        
        return result
    

    def _interpolateGaps_all(self, sinograms):
        interpol_sinograms = np.zeros((sinograms.shape[0], self.det_count, sinograms.shape[2],  sinograms.shape[3]))
        for slice_ind in itertools.islice(itertools.count(), 0, sinograms.shape[2]):
            for channel_ind in itertools.islice(itertools.count(), 0, sinograms.shape[0]):
                interpol_sinograms[channel_ind,:,slice_ind,:] = self._interpolateGaps(sinograms[channel_ind,:,slice_ind,:])
                
        return interpol_sinograms
            
            

    def _reconstruct_all(self, sinograms, image_size, object_scale):
        #s_x = image_size[0]
        #s_y = image_size[1]
        #sc = object_scale
        #image_scale
        #vol_geom = astra.create_vol_geom(s_x, s_y, -s_x/(sc*2), s_x/(sc*2), -s_y/(sc*2), s_y/(sc*2))
        #rec_id = astra.data2d.create('-vol', vol_geom)         
        
        self.setupVolume(image_size, object_scale)
        
        if(len(sinograms.shape)!=4):
            print("Input is not a 4D NumPy array - abort.")
            return None
        print(sinograms.shape)
        
        result = np.zeros((image_size[0], image_size[1], sinograms.shape[2],  sinograms.shape[0]))
        #sinogram_new = np.zeros((self.det_count,self.angles.shape[0]))
        #interpol_sinograms = np.zeros((sinograms.shape[0], self.det_count, sinograms.shape[2],  sinograms.shape[3]))
        interpol_sinograms = self._interpolateGaps_all(sinograms)
        #result = np.zeros((image_size[0], image_size[1], 1,  1))
        print(result.shape)
#        for slice_ind in itertools.islice(itertools.count(), 237, 238):
        for slice_ind in itertools.islice(itertools.count(), 0, sinograms.shape[2]):
            #print("slice_ind: " + str(slice_ind))
            #result  = self._reconstruct_spectral(result, interpol_sinograms, slice_ind)
            result  = self._reconstruct_spectral(result, interpol_sinograms, slice_ind)
            #for channel_ind in itertools.islice(itertools.count(), 0, sinograms.shape[0]):
                #print("j " + str(channel_ind))
            #    result[:,:,slice_ind,channel_ind], interpol_sinograms[channel_ind,:,slice_ind,:] = self._reconstOneSino(sinograms[channel_ind,:,slice_ind,:])
           
        #print(interpol_sinograms.shape)
        
        
        print('after _reconstruct_all')
        print(result.shape)
        where_are_NaNs = np.isnan(result)
        print('Nans:')
        print(where_are_NaNs.sum())
        where_are_inf = np.isinf(result)
        print('Inf:')
        print(where_are_inf.sum())
        
        astra.data2d.delete(self.rec_id)
        return result, interpol_sinograms
        
    def _interpolateGaps(self, sinogram):
        #sinogram[where_are_NaNs] = 0
        #sinogram[where_are_inf] = np.max(sinogram)
        #sinogram[where_are_inf] = 0
        #sinogram = np.nan_to_num(sinogram)
        #sinogram = sinogram[:,5:-5]
        #sinogram = sinogram[:,0::round(360/self.sim_conf['proj_no'])]
        sinogram_new = np.zeros((self.det_count,self.angles.shape[0]))
        #print(self.elem_pxs)
        #print(self.gap_size_pxs)
        for element_id in itertools.islice(itertools.count(), 0, int(self.geo_struct["nElem"])):
          from_id = int(element_id*(self.elem_pxs+self.gap_size_pxs))
          from_id_orig = int(element_id*(self.elem_pxs))
          end_id = int(from_id+self.elem_pxs)
          end_id_orig = int(from_id_orig+self.elem_pxs)
          #print(from_id)
          #print(from_id_orig)
          #print(end_id)
          sinogram_new[from_id:end_id,] = sinogram[from_id_orig:end_id_orig,]
          #sinogram_new[self.elem_pxs+self.gap_size_pxs:,] = sinogram[self.elem_pxs:,]
        
        for element_id in itertools.islice(itertools.count(), 0, int(self.geo_struct["nElem"]-1)):
           from_id = int(element_id*(self.elem_pxs+self.gap_size_pxs))
           end_id = int(from_id+self.elem_pxs)
           from_id_orig = int(element_id*(self.elem_pxs))
           end_id_orig = int(from_id_orig+self.elem_pxs)
           #print("_interpolateGaps")
           #print(from_id)
           #print(end_id)
           inter_data = sinogram[end_id_orig-1:end_id_orig+2,]
           #inter_data = sinogram[end_id:end_id,]
           #print(inter_data.shape)
           inter_data_mean = np.mean(inter_data, axis=0)
           #print(inter_data_mean.shape)
           #print((end_id+self.gap_size_pxs))
           sinogram_new[end_id:(end_id+self.gap_size_pxs),] = inter_data_mean
           #print(end_id+self.gap_size_pxs)
           #sinogram_new[end_id:(end_id+self.gap_size_pxs),] = 0
        
        #sinogram_new[128,] = inter_data_mean
        #sinogram_new[129,] = inter_data_mean
        return sinogram_new
    
    def _reconstOneSino(self, sinogram):

        #sinogram = self._interpolateGaps(sinogram)
        sinogram = np.float32(sinogram.transpose())
        
        #print(sinogram.shape)
        #raise SystemExit
        #if self.sinogram_id is not None:
        
        sinogram_id = astra.data2d.create('-sino', self.proj_geom, sinogram)
        self.setupProjector()
        reconst = self._reconstruct(sinogram, sinogram_id)
        astra.data2d.delete(sinogram_id)
        #sinogram = np.float32(sinogram.transpose())
        #return reconst, sinogram
        return reconst
        
    
    def setupVolume(self, image_size, object_scale):
        self.image_size = image_size;
        self.object_scale = object_scale;
        s_x = self.image_size[0]
        s_y = self.image_size[1]
        sc = self.object_scale
        #image_scale
        self.vol_geom = astra.create_vol_geom(s_x, s_y, -0.5*(s_x/sc), 0.5*(s_x/sc), -0.5*(s_y/sc), 0.5*(s_y/sc))
        #self.vol_geom = astra.create_vol_geom(s_x, s_y, -s_x/(sc*1), s_x/(sc*1), -s_y/(sc*1), s_y/(sc*1))
        self.rec_id = astra.data2d.create('-vol', self.vol_geom)     
        
        #vol_geom = astra.create_vol_geom(images[0,].shape[0], images[0,].shape[1])
        #vol_geom = astra.create_vol_geom(96, 96)
        
    def setupProjector(self):
        #self.proj_id = astra.create_projector('strip_fanflat', self.proj_geom, self.vol_geom)      
        self.proj_id = astra.create_projector('line_fanflat', self.proj_geom, self.vol_geom)    
         
    
    def generateFromImages(self, images):
            
        print (images.shape)
        self.setupVolume(images.shape,1)
        self.setupProjector()
                
        self.images_tranformed =  np.array([self._transform(item) for item in images])

        astra.data2d.delete(self.rec_id)
        astra.projector.delete(self.proj_id)
        
    def generateSinogramsFromImages(self, images, object_scale):
            
        print ('generateSinogramsFromImages')
        print (images.shape)
        self.setupVolume(images.shape, object_scale)
        self.setupProjector()
        #(1, 256, 300, 74)
        self.projections = np.zeros((images.shape[3], self.det_count, images.shape[2],  self.proj_no))
        for slice_ind in itertools.islice(itertools.count(), 0, images.shape[2]):
            #print("slice_ind: " + str(slice_ind))
            for channel_ind in itertools.islice(itertools.count(), 0, images.shape[3]):
                #print("j " + str(channel_ind))
                self.projections[channel_ind,:,slice_ind,:] = self._project(images[:,:,slice_ind,channel_ind]).transpose()
        
        #self.projections =  np.array([self._project(item) for item in images])
        
        print (self.projections.shape)

        astra.data2d.delete(self.rec_id)
        astra.projector.delete(self.proj_id)
        
        return self.projections
    
    def _project(self, image):
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id)
        return sinogram
    
    def _transform(self, image):
        
        sinogram_id, sinogram = astra.create_sino(image, self.proj_id)
        return self._reconstruct(sinogram, sinogram_id)
        
    def _reconstruct(self, sinogram, sinogram_id):
        #print(sinogram.shape) # angle, pixels
        # Set up the parameters for a reconstruction algorithm using the GPU
        cfg = astra.astra_dict('FBP_CUDA') # available algorithms: SIRT_CUDA, SART_CUDA, EM_CUDA, FBP_CUDA, BP_CUDA
        cfg['ReconstructionDataId'] = self.rec_id
        cfg['ProjectionDataId'] = sinogram_id
        #cfg['ProjectorId'] = self.proj_id # Necessary for CPU version
        
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, 20)
        
        image_rec = astra.data2d.get(self.rec_id)
        
        astra.algorithm.delete(alg_id)      
        astra.data2d.delete(sinogram_id)
        
        return image_rec

    
    def get_conf_name(self):
        name = str(list(self.sim_conf.values())) # convert dic values to string
        #name = re.sub('[!\'[]@#$]', '', name)
        name = name.replace("]","")
        name = name.replace("[","")
        name = name.replace("\'","")
        name = name.replace(",","_")
        name = name.replace(" ","")
        return name
    
    def get_data(self):
        return self.images_tranformed, self.interpol_sinograms
    
    def get_conf(self):
        return self.sim_conf
        
    

    
class CTDataGeneratorTNV(CTDataGenerator):
    def __init__(self, sim_conf):
        super(CTDataGeneratorTNV,self).__init__(sim_conf = sim_conf)
        
    def _reconstruct_spectral(self, result, interpol_sinograms, slice_ind):
            
        import odl 
        
        #print (interpol_sinograms.shape)
        #raise SystemExit
        data_corr = interpol_sinograms[:,:,slice_ind,:]
        data_corr = data_corr.transpose()
        vecs = self.proj_geom_vector
        
        print('slice id: ' + str(slice_ind))
        print('data_corr.shape')
        print(data_corr.shape)
        #height=data_corr.shape[1]
        #width=data_corr.shape[1]
        height = self.image_size[0]
        width = self.image_size[1]
    
        space = odl.uniform_discr(min_pt=[-width/2]*2, max_pt=[height/2]*2, shape=[width]*2, dtype='float32')
    
        #angle_idx = np.arange(0, 360, 1)        #angle selection
        angle_idx = np.arange(0, data_corr.shape[0], 1)        #angle selection
        #print(angle_idx)
        #angle_idx=np.linspace(0, 359, data_corr.shape[0],endpoint=False, dtype=int)
    
        vecs = vecs[angle_idx, :]
        vec_geom = odl.tomo.ConeVecGeometry(data_corr.shape[1], vectors=vecs)
        # vec_geom = odl.tomo.ParallelVecGeometry(width, vectors=vecs)
    
        ray_trafo = odl.tomo.RayTransform(space, vec_geom, impl='astra_cuda')
        
        lamb = 0.8       #lambda parameter
        #lambs=np.arange(0.1, 1, 0.1)
        # energies = [0]
        #energies = [20,60,90]         #energy channels
        #energies = [10,20,30,70]#,20,30]#np.arange(0, 104, 20)        #energy channels
        energies = np.arange(0, data_corr.shape[2], 1)
        n_energy = len(energies)
        
        #scale = 10     #scaling multiplies sinogram values 
        scale = 1     #scaling multiplies sinogram values 
        if 1:
            data_corr *= scale
        
            
            
        import odlL2NormSquaredAndSeparableSum_maxnE128 as L2sepSum
        
        g_l2 = getattr(L2sepSum, 'numE'+str(n_energy))(angle_idx,energies,ray_trafo,data_corr,odl)
        
        
        gradient = odl.Gradient(ray_trafo.domain)
        
        if n_energy == 1:
            forward_op = ray_trafo
            pgradient = gradient
            g_reg = odl.solvers.L1Norm(gradient.range)
        else:
            forward_op = odl.DiagonalOperator(ray_trafo, n_energy)
            pgradient = odl.DiagonalOperator(gradient, n_energy)
            g_reg = odl.solvers.NuclearNorm(pgradient.range,
                                              singular_vector_exp=1)
        
        lin_ops = [forward_op, pgradient]
        
        
        f = odl.solvers.IndicatorBox(forward_op.domain, 0)
        #func = f + g_l2 * forward_op + lamb * g_reg * pgradient
        # func = f
        # func = g_l2 * forward_op
        
        #callback = (odl.solvers.CallbackShow(step=10) &
        #            odl.solvers.CallbackPrint(func=func)
        #             odl.solvers.CallbackPrint(func=lamb * g_reg * pgradient)
        #           )
        
        # x = 0.5*forward_op.domain.one()
        #x = forward_op.domain.zero()
        #g = [g_l2, lamb * g_reg]
        
        tau = 4.0/ len(lin_ops)   # maximum value for this parameter:    tau_max = 4.0 / len(lin_ops)
        sigma = [1 / odl.power_method_opnorm(op, rtol=0.01)**2 for op in lin_ops]        #sigma parameter
        niter=200       #number of iterations
        #niter=10       #number of iterations
        
        
        #num_lamb=len(lamb)
        #x= np.zeros([648,648, num_lamb ])
        
        
        #lambs=[0.2,0.3,0.4]
        #lambs=np.arange(0.1, 1, 0.1)
        #x=x[2,:,:]
        #x_tnv_array = np.zeros([height, width, len(lambs)])
        #for i in range(len(lambs)):
         #   lamb = lambs[i]
        g = [g_l2, lamb * g_reg]
        #func = f + g_l2 * forward_op + lamb * g_reg * pgradient
        x = forward_op.domain.zero()
        
        
        odl.solvers.douglas_rachford_pd(x, f, g, lin_ops, tau=tau, sigma=sigma,
                                       niter=niter,
                                       #callback=callback
                                       )
        #x=x[1,:,:]
            #x_tnv_array[:,:,i] = np.array(x) 
        #print (result.shape)
        #print (np.array(x).transpose().shape)
        #raise SystemExit
        result[:,:,slice_ind, :] = np.array(x).transpose() 
        
        return result
        
    
