#!/usr/bin/python
'''
Created on Jun 08, 2018

@author: christian
@mail: chke@dtu.dk

Derived from "GenTrainingSet_split.m" by 
Matteo Busi @ DTU Physics (mbusi@fysik.dtu.dk)
'''
import os
import numpy
from argparse import ArgumentParser
from tomophantom import TomoP2D
from tomophantom import TomoP3D
import matplotlib.pyplot as plt
from scipy import io
import h5py
import string
import itertools
import math
#import random
import AttenuationDB_simple
from numpy import float32

#tomo_models_path = "/media/christian/DATA/git/TomoPhantom/functions/models"


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices // 2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class Envelope():
    geom = 0  # 1:ellipsoid, 2:cuboid, 3: sphube, 4: elliptical cylinder
    hollow = True  # hollow (True) or solid (False) object ?
    mat = 0  # material number
    clipmat = numpy.array([0])

    def __init__(self):
        self.geom = 0
        self.hollow = True
        self.mat = 0
        self.clipmat = numpy.array([0])

    def __str__(self):
        pass

    def __del__(self):
        pass

class SampleParameters():
    objCenterMin = numpy.array([-0.2, -0.2, -0.2], dtype=numpy.float32)  # objects minimum center (x,y)
    objCenterMax = numpy.array([0.2, 0.2, 0.2], dtype=numpy.float32)  # objects maximum center (x,y)
    objDimMin = numpy.array([0.10, 0.10, 0.10], dtype=numpy.float32)  # objects minimum axes (x,y)
    objDimMax = numpy.array([0.25, 0.25, 0.25], dtype=numpy.float32)  # objects maximum axes (x,y)
    objRotMin = 0.0  # objects minimum rotation
    objRotMax = 45.00  # objects maximum rotation
    objStraightness = 1.0

    def __init__(self):
        self.objCenterMin = numpy.array([-0.2, -0.2, -0.2], dtype=numpy.float32)  # objects minimum center (x,y)
        self.objCenterMax = numpy.array([0.2, 0.2, 0.2], dtype=numpy.float32)  # objects maximum center (x,y)
        self.objDimMin = numpy.array([0.10, 0.10, 0.10], dtype=numpy.float32)  # objects minimum axes (x,y)
        self.objDimMax = numpy.array([0.25, 0.25, 0.25], dtype=numpy.float32)  # objects maximum axes (x,y)
        self.objRotMin = 0.0  # objects minimum rotation
        self.objRotMax = 45.00  # objects maximum rotation
        self.objStraightness = 1.0


def generateSphereObject(parameters, materialNumber, N):
    result = numpy.zeros((N, N, N))
    paramObj = dict(Obj = TomoP3D.Objects3D.ELLIPSOID, C0= float(1), x0= 0.0,
                y0= 0.0, z0= 0.0, a= 0.95, b= 0.95, c= 0.95, phi1= 0.0, phi2= 0.0, phi3= 0.0,
                tt= 0, s= 1.0)
    paramObj['C0'] = float(materialNumber)
    # paramObj['C0'] = float(mat[0])
    paramObj['x0'] = float(numpy.random.uniform(low=parameters.objCenterMin[0], high=parameters.objCenterMax[0]))
    paramObj['y0'] = float(numpy.random.uniform(low=parameters.objCenterMin[1], high=parameters.objCenterMax[1]))
    paramObj['z0'] = float(numpy.random.uniform(low=parameters.objCenterMin[2], high=parameters.objCenterMax[2]))
    paramObj['a'] = float(numpy.random.uniform(low=parameters.objDimMin[0], high=parameters.objDimMax[0]))
    paramObj['b'] = float(numpy.random.uniform(low=parameters.objDimMin[1], high=parameters.objDimMax[1]))
    paramObj['c'] = float(numpy.random.uniform(low=parameters.objDimMin[2], high=parameters.objDimMax[2]))
    paramObj['phi1'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi2'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi3'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['s'] = parameters.objStraightness
    result = TomoP3D.Object(N, paramObj)
    return result

def generateCuboidObject(parameters, materialNumber, N):
    result = numpy.zeros((N, N, N))
    paramObj = dict(Obj = TomoP3D.Objects3D.CUBOID, C0= float(1), x0= 0.0,
                y0= 0.0, z0= 0.0, a= 0.95, b= 0.95, c= 0.95, phi1= 0.0, phi2= 0.0, phi3= 0.0,
                tt= 0, s= 1.0)
    paramObj['C0'] = float(materialNumber)
    # paramObj['C0'] = float(mat[0])
    paramObj['x0'] = float(numpy.random.uniform(low=parameters.objCenterMin[0], high=parameters.objCenterMax[0]))
    paramObj['y0'] = float(numpy.random.uniform(low=parameters.objCenterMin[1], high=parameters.objCenterMax[1]))
    paramObj['z0'] = float(numpy.random.uniform(low=parameters.objCenterMin[2], high=parameters.objCenterMax[2]))
    paramObj['a'] = float(numpy.random.uniform(low=parameters.objDimMin[0], high=parameters.objDimMax[0]))
    paramObj['b'] = float(numpy.random.uniform(low=parameters.objDimMin[1], high=parameters.objDimMax[1]))
    paramObj['c'] = float(numpy.random.uniform(low=parameters.objDimMin[2], high=parameters.objDimMax[2]))
    paramObj['phi1'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi2'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi3'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['s'] = parameters.objStraightness
    paramObj['a'] = paramObj['a'] * 2.0
    paramObj['b'] = paramObj['b'] * 2.0
    paramObj['c'] = paramObj['c'] * 2.0
    result = TomoP3D.Object(N, paramObj)
    return result

def generateCylinderObject(parameters, materialNumber, N):
    result = numpy.zeros((N, N, N))
    paramObj = dict(Obj = TomoP3D.Objects3D.ELLIPCYLINDER, C0= float(1), x0= 0.0,
                y0= 0.0, z0= 0.0, a= 0.95, b= 0.95, c= 0.95, phi1= 0.0, phi2= 0.0, phi3= 0.0,
                tt= 0, s= 1.0)
    paramObj['C0'] = float(materialNumber)
    # paramObj['C0'] = float(mat[0])
    paramObj['x0'] = float(numpy.random.uniform(low=parameters.objCenterMin[0], high=parameters.objCenterMax[0]))
    paramObj['y0'] = float(numpy.random.uniform(low=parameters.objCenterMin[1], high=parameters.objCenterMax[1]))
    paramObj['z0'] = float(numpy.random.uniform(low=parameters.objCenterMin[2], high=parameters.objCenterMax[2]))
    paramObj['a'] = float(numpy.random.uniform(low=parameters.objDimMin[0], high=parameters.objDimMax[0]))
    paramObj['b'] = float(numpy.random.uniform(low=parameters.objDimMin[1], high=parameters.objDimMax[1]))
    paramObj['c'] = float(numpy.random.uniform(low=parameters.objDimMin[2], high=parameters.objDimMax[2]))
    paramObj['phi1'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi2'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi3'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['s'] = parameters.objStraightness
    result = TomoP3D.Object(N, paramObj)
    return result

def generateSphericalCubeObject(parameters, materialNumber, N):
    result = numpy.zeros((N, N, N))
    paramObj = dict(Obj = TomoP3D.Objects3D.ELLIPSOID, C0= float(1), x0= 0.0,
                y0= 0.0, z0= 0.0, a= 0.95, b= 0.95, c= 0.95, phi1= 0.0, phi2= 0.0, phi3= 0.0,
                tt= 0, s= 1.0)
    paramObj['C0'] = float(materialNumber)
    # paramObj['C0'] = float(mat[0])
    paramObj['x0'] = float(numpy.random.uniform(low=parameters.objCenterMin[0], high=parameters.objCenterMax[0]))
    paramObj['y0'] = float(numpy.random.uniform(low=parameters.objCenterMin[1], high=parameters.objCenterMax[1]))
    paramObj['z0'] = float(numpy.random.uniform(low=parameters.objCenterMin[2], high=parameters.objCenterMax[2]))
    paramObj['a'] = float(numpy.random.uniform(low=parameters.objDimMin[0], high=parameters.objDimMax[0]))
    paramObj['b'] = float(numpy.random.uniform(low=parameters.objDimMin[1], high=parameters.objDimMax[1]))
    paramObj['c'] = float(numpy.random.uniform(low=parameters.objDimMin[2], high=parameters.objDimMax[2]))
    paramObj['phi1'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi2'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi3'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['s'] = parameters.objStraightness
    result = TomoP3D.Object(N, paramObj)
    return result

def generateFilledGlassTube(parameters, innerMaterialNumber, N, glassIndicator):
    result = numpy.zeros((N, N, N))
    innerSample = numpy.zeros((N, N, N))
    paramObj = dict(Obj = TomoP3D.Objects3D.ELLIPCYLINDER, C0= float(1), x0= 0.0,
                y0= 0.0, z0= 0.0, a= 0.95, b= 0.95, c= 0.95, phi1= 0.0, phi2= 0.0, phi3= 0.0,
                tt= 0, s= 1.0)
    paramObj['C0'] = float(glassIndicator)
    # paramObj['C0'] = float(mat[0])
    paramObj['x0'] = float(numpy.random.uniform(low=parameters.objCenterMin[0], high=parameters.objCenterMax[0]))
    paramObj['y0'] = float(numpy.random.uniform(low=parameters.objCenterMin[1], high=parameters.objCenterMax[1]))
    paramObj['z0'] = float(numpy.random.uniform(low=parameters.objCenterMin[2], high=parameters.objCenterMax[2]))
    paramObj['a'] = float(numpy.random.uniform(low=parameters.objDimMin[0], high=parameters.objDimMax[0]))
    paramObj['b'] = float(numpy.random.uniform(low=parameters.objDimMin[1], high=parameters.objDimMax[1]))
    paramObj['c'] = float(numpy.random.uniform(low=parameters.objDimMin[2], high=parameters.objDimMax[2]))
    paramObj['phi1'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi2'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['phi3'] = float(numpy.random.uniform(low=parameters.objRotMin, high=parameters.objRotMax))
    paramObj['s'] = parameters.objStraightness
    result = TomoP3D.Object(N, paramObj)
    paramObj['a'] = paramObj['a'] * 0.9
    paramObj['b'] = paramObj['b'] * 0.9
    paramObj['c'] = paramObj['c'] * 0.9
    paramObj['C0'] = float(innerMaterialNumber)
    innerSample = TomoP3D.Object(N, paramObj)
    inner_indices = innerSample > 0.0
    result[inner_indices] = innerSample[inner_indices]
    return result



class Generator():
  
  def run(self, Nt=1, N=20, sigma=0.01):
    matDB = AttenuationDB_simple.AttenuationDB()
    # Parameters
    Nt = Nt  # number of training phantoms
    Nobj_min = 3  # minimum number of inner objects - 3
    Nobj_max = 9  # maximum number of inner objects - 9
    #Perc_cuboids = 0.3  # percentage of cuboid objects
    Nmat_max = matDB.getNumberOfMaterials()-1 # maximum number of materials
    N = N  # phantom size
    Ebins = 32  # number of spectral bins

    objCenterMin = numpy.array([-0.2, -0.2, -0.2], dtype=numpy.float32)  # objects minimum center (x,y.z)
    objCenterMax = numpy.array([0.2, 0.2, 0.2], dtype=numpy.float32)  # objects maximum center (x,y,z)
    objCenter_Uncertainty_AccumRate = 0.02
    objDimMin = numpy.array([0.10, 0.10, 0.10], dtype=numpy.float32)  # objects minimum axes (x,y,z)
    objDimMax = numpy.array([0.25, 0.25, 0.25], dtype=numpy.float32)  # objects maximum axes (x,y,z)
    objDimDecay = 0.01
    objRotMin = 0.0  # objects minimum rotation
    objRotMax = 45.00  # objects maximum rotation
    objRot_Uncertainty_AccumRate = 1.0


    envelope = Envelope()
    envelope.geom = 0  # 1:ellipsoid, 2:cuboid, 3: sphube, 4: elliptical cylinder
    envelope.hollow = True  # 1:hollow, 2: full
    envelope.mat = matDB.getMaterialIndex("plexiglass") # envelope material
    envelope.clipmat = numpy.zeros((N,N,N), dtype=numpy.float32)
    envelope_variation = (0.75, 1.0)

    gen_min = [1000.0,]*32
    gen_max = [-1000.0,]*32

    phantoms = numpy.zeros((Nt, N, N, N, Ebins), dtype=numpy.float32)
    phantoms_ref = numpy.zeros((Nt, N, N, N, Ebins), dtype=numpy.float32)

    for i in range(0, Nt):
        envelope.geom = numpy.random.randint(1,4)
        envelope_scale = numpy.random.uniform(envelope_variation[0],envelope_variation[1])

        print("processing img %d" % (i))
        # reinitialise object data #
        objCenterMin = numpy.array([-0.2, -0.2, -0.2], dtype=numpy.float32)
        objCenterMax = numpy.array([0.2, 0.2, 0.2], dtype=numpy.float32)
        objDimMin = numpy.array([0.10, 0.10, 0.10], dtype=numpy.float32)
        objDimMax = numpy.array([0.25, 0.25, 0.25], dtype=numpy.float32)
        objRotMin = 0.0
        objRotMax = 10.0

        G = numpy.zeros((N, N, N), dtype=numpy.float32)
        envelope.clipmat = numpy.ones((N, N, N), dtype=numpy.float32)
        # ---------------------------- #
        # First: generate bounding box #
        # ---------------------------- #
        
        print('envelope.geom:', envelope.geom)
        if envelope.geom == 4:
            obj = {'Obj': TomoP3D.Objects3D.ELLIPCYLINDER,
                   'C0': float(envelope.mat),
                   'x0': 0.0,
                   'y0': 0.0,
                   'z0': 0.0,
                   'a': 0.98*envelope_scale,
                   'b': 0.98*envelope_scale,
                   'c': 0.98*envelope_scale,
                   'phi1': 0.0,
                   'phi2': 0.0,
                   'phi3': 0.0,
                   'tt': 0,
                   's': 1.0}
            #print(obj)
            #envelope.clipmat = TomoP3D.Object(N, obj)  # object lies on its side - bad for training slices.
            envelope.clipmat = numpy.transpose(TomoP3D.Object(N, obj))
            G = G + envelope.clipmat
            if envelope.hollow:
                obj['a'] = 0.90*envelope_scale
                obj['b'] = 0.90*envelope_scale
                obj['c'] = 0.98*envelope_scale
                envelope.clipmat = numpy.transpose(TomoP3D.Object(N, obj))
                G = G - envelope.clipmat
            inIndices = envelope.clipmat>0
            envelope.clipmat[inIndices]=1.0
        elif envelope.geom == 3:
            obj = {'Obj': TomoP3D.Objects3D.ELLIPSOID,
                   'C0': float(envelope.mat),
                   'x0': 0.0,
                   'y0': 0.0,
                   'z0': 0.0,
                   'a': 0.76*envelope_scale,
                   'b': 0.76*envelope_scale,
                   'c': 0.96*envelope_scale,
                   'phi1': 0.0,
                   'phi2': 0.0,
                   'phi3': 0.0,
                   'tt': 0,
                   's': 0.90*envelope_scale}
            #print(obj)
            envelope.clipmat = TomoP3D.Object(N, obj)
            G = G + envelope.clipmat
            if envelope.hollow:
                obj['a'] = 0.72*envelope_scale
                obj['b'] = 0.72*envelope_scale
                obj['c'] = 0.92*envelope_scale
                envelope.clipmat = TomoP3D.Object(N, obj)
                G = G - envelope.clipmat
            inIndices = envelope.clipmat>0
            envelope.clipmat[inIndices]=1.0
            #print(obj)
        elif envelope.geom == 2:
            obj = {'Obj': TomoP3D.Objects3D.CUBOID,
                   'C0': float(envelope.mat),
                   'x0': 0.0,
                   'y0': 0.0,
                   'z0': 0.0,
                   'a': 1.9*envelope_scale,
                   'b': 1.9*envelope_scale,
                   'c': 1.9*envelope_scale,
                   'phi1': 0.0,
                   'phi2': 0.0,
                   'phi3': 0.0,
                   'tt': 0,
                   's': 1.0
                   }
            envelope.clipmat = TomoP3D.Object(N, obj)
            G = G + envelope.clipmat
            if envelope.hollow:
                obj['a'] = 1.8*envelope_scale
                obj['b'] = 1.8*envelope_scale
                obj['c'] = 1.8*envelope_scale
                envelope.clipmat = TomoP3D.Object(N, obj)
                G = G - envelope.clipmat
            inIndices = envelope.clipmat > 0
            envelope.clipmat[inIndices] = 1.0
        elif envelope.geom == 1:
            obj = {'Obj': TomoP3D.Objects3D.ELLIPSOID,
                   'C0': float(envelope.mat),
                   'x0': 0.0,
                   'y0': 0.0,
                   'z0': 0.0,
                   'a': 0.95*envelope_scale,
                   'b': 0.95*envelope_scale,
                   'c': 0.95*envelope_scale,
                   'phi1': 0.0,
                   'phi2': 0.0,
                   'phi3': 0.0,
                   'tt': 0,
                   's': 1.0
                   }
            envelope.clipmat = TomoP3D.Object(N, obj)
            G = G + envelope.clipmat
            if envelope.hollow:
                obj['a'] = 0.9*envelope_scale
                obj['b'] = 0.9*envelope_scale
                obj['c'] = 0.9*envelope_scale
                envelope.clipmat = TomoP3D.Object(N, obj)
                G = G - envelope.clipmat
            inIndices = envelope.clipmat > 0
            envelope.clipmat[inIndices] = 1.0
        if Nobj_max>0:
            # ---------------------------- #
            #   Second: generate objects   #
            # ---------------------------- #
            # Nobj = Nobj_min + (Nobj_max-Nobj_min)        # deterministic
            Nobj = numpy.random.randint(Nobj_min, Nobj_max)
            #mat = [sampleMat for sampleMat in range(0, Nmat_max)]
            #mat = []
            #for k in range(0,Nobj):
            #    mat.append(numpy.random.randint(low=1,high=Nmat_max))
            k = 0
            Gnew = numpy.zeros((N, N, N))
            matIndex = 0
            while k < Nobj:
                params = SampleParameters()
                params.objCenterMin = objCenterMin
                params.objCenterMax = objCenterMax
                params.objDimMin = objDimMin
                params.objDimMax = objDimMax
                params.objRotMin = objRotMin
                params.objRotMax = objRotMax
                params.objStraightness = 1.0
                if(k > (Nobj_max-2)):
                    liquidIndex = numpy.random.randint(low=0,high=matDB.getNumberOfLiquids())
                    liquidName = matDB.getLiquidName(liquidIndex)
                    matIndex = matDB.getMaterialIndex(liquidName)
                    Gnew = generateFilledGlassTube(params, float(matIndex), N, matDB.getMaterialIndex("glass"))
                else:
                    shapeNum = numpy.random.randint(0,4)
                    matIndex = numpy.random.randint(low=1,high=matDB.getNumberOfMaterials())
                    if shapeNum == 0:
                        #generate spheroid/ellipsoid
                        Gnew = generateSphereObject(params, float(matIndex), N)
                    elif shapeNum == 1:
                        #generate cuboid
                        Gnew = generateCuboidObject(params, float(matIndex), N)
                    elif shapeNum == 2:
                        #generate (elliptical) cylinder
                        Gnew = generateCylinderObject(params, float(matIndex), N)
                    elif shapeNum == 3:
                        #generate spherical cube (sphube)
                        params.objStraightness = numpy.random.uniform(low=0.0, high=1.0)
                        Gnew = generateSphericalCubeObject(params, float(matIndex), N)

                Gnew = Gnew * envelope.clipmat
                outIndex = Gnew<0.5
                Gnew[outIndex] = G[outIndex]
                vol_valid = False
                if k > 0:  # this one I don't understand in the original code; I assume it's to prevent empty generations
                    dG = numpy.sum(numpy.fabs(Gnew - G))
                    nG = numpy.sum(Gnew)
                    if (dG > .0) and (nG > .0):
                        k += 1
                        matIndex = min(matIndex + 1, Nmat_max - 1)
                        objCenterMin -= objCenter_Uncertainty_AccumRate
                        objCenterMax += objCenter_Uncertainty_AccumRate
                        objDimMin -= objDimDecay
                        objDimMax -= objDimDecay
                        objRotMax = min(objRotMax + objRot_Uncertainty_AccumRate, 180.0)
                        vol_valid=True
                else:
                    k += 1
                    matIndex = min(matIndex + 1, Nmat_max - 1)
                    objCenterMin -= objCenter_Uncertainty_AccumRate
                    objCenterMax += objCenter_Uncertainty_AccumRate
                    objDimMin -= objDimDecay
                    objDimMax -= objDimDecay
                    objRotMax = min(objRotMax + objRot_Uncertainty_AccumRate, 180.0)
                    vol_valid = True
                if(vol_valid==True):
                    G = Gnew

        # =============================== #
        # Assign attenuation coefficients #
        # =============================== #
        k = 0
        LAC = numpy.zeros((N, N, N, Ebins))
        LAC_ref = numpy.zeros((N, N, N, Ebins))
        #while k < (Nmat_max+1):
        while k < (Nmat_max+1):
            tmp = numpy.zeros((N, N, N), dtype=numpy.float32)
            matName = matDB.getMaterialName(int(k))
            mu_sigma = matDB.getParameters(matName)
            # tmp[(G > (float(k)-0.01)) and (G < (float(k)+0.01))] = 1.0
            tmp[G.astype(numpy.int32) == (k)] = 1.0
            # print(tmp)
            for ii in itertools.islice(itertools.count(), 0, Ebins):
                #gaussSample = numpy.random.normal(mu_sigma[0, ii], mu_sigma[1, ii], (N, N, N))
                gaussSample = numpy.random.normal(mu_sigma[0,ii], sigma, (N,N,N))
                gaussSample = numpy.clip(gaussSample, 0, None)
                gen_min[ii] = min(gen_min[ii], numpy.min(gaussSample))
                gen_max[ii] = max(gen_max[ii], numpy.max(gaussSample))
                gaussSample = gaussSample * tmp
                LAC[:, :, :, ii] += gaussSample
                gaussSample = numpy.random.normal(mu_sigma[0,ii],0.0,(N, N,N))
                gaussSample = numpy.clip(gaussSample, 0, None)
                gaussSample = gaussSample * tmp
                LAC_ref[:, :, :, ii] += gaussSample
            k += 1
            

        saveSwitch = False
        if saveSwitch:
            phantom_path = os.path.join(dirpath, dname + "_phantom" + str(i) + ext)
            #fbp_path = os.path.join(dirpath, dname + "_fbp" + str(i) + ext)

            f = h5py.File(phantom_path, 'w')
            data_group = f.create_group('data')
            # data_hdf5 = phantoms
            # data_group.create_dataset('value', data=data_hdf5)
            data_group.create_dataset('value', data=LAC.transpose())
            f.close()
            
            phantom_path = os.path.join(dirpath, dname + "_phantom_ref" + str(i) + ext)
            f = h5py.File(phantom_path, 'w')
            data_group = f.create_group('data')
            # data_hdf5 = phantoms
            # data_group.create_dataset('value', data=data_hdf5)
            data_group.create_dataset('value', data=LAC_ref.transpose())
            f.close()
            

        phantoms[i, :, :, :, :] = LAC
        phantoms_ref[i, :, :, :, :] = LAC_ref

    print("Generated min-LAC curve: {}".format(gen_min))
    print("Generated max-LAC curve: {}".format(gen_max))
    
    return phantoms, phantoms_ref

if __name__ == '__main__':
    optionParser = ArgumentParser(description="data generation routine for 3D spectral phantoms")
    optionParser.add_argument("-o", "--output", action="store", dest="output", default="",
                              help="output HDF5 file for result storage")
    optionParser.add_argument("-N", "--nSamples", action="store", type=int, dest="nSamples",
                              help="number of samples to generate")
    options = optionParser.parse_args()

    argDict = vars(options)

    dirpath = os.path.dirname(options.output)
    bname = os.path.basename(options.output)
    dotpos = bname.rfind('.')
    dname = bname[0:dotpos]
    ext = bname[dotpos:]
    
    generator = Generator()
    if argDict["nSamples"] != None:
        Nt = argDict["nSamples"]
        LAC, LAC_ref = generator.run(Nt)
    else:
        LAC, LAC_ref = generator.run()





