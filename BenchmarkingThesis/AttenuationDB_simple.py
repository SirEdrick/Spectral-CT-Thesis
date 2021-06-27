import math
import numpy

class AttenuationDB():
    _db = {}
    #
    _keys = ["background","acetone","whiskey","olive_oil","h2o","saltwater","h2o2_30p","h2o2_50p","h2o2_70p","methanol",
"alcohol_40p","benzene","butane","octane","ethylene","nitric_acid","nitroglycerine","nitromethane","magnesium_carbonate","caco3_argonite",
"caco3_calcite","c4_simulant","c4_real","c4_NIST","salt","glass","plexiglass"]
    _solids = ["magnesium_carbonate","caco3_argonite","caco3_calcite","c4_simulant","c4_real","c4_NIST","salt"]
    _liquids = ["acetone","whiskey","olive_oil","h2o","saltwater","h2o2_30p","h2o2_50p","h2o2_70p","methanol","alcohol_40p","benzene",
"butane","octane","ethylene","nitric_acid","nitroglycerine","nitromethane"]
    
    def __init__(self):
        self._db = {}
        #mu_sigma = numpy.zeros((2,128), dtype=numpy.float32)
        #mu_sigma = numpy.zeros((2,16), dtype=numpy.float32)
        #background
        background = numpy.zeros((2,32), dtype=numpy.float32)
        background[0,:] = numpy.array([0]*32, dtype=numpy.float32)
        background[1,:] = numpy.array([0.1,0.09,0.08,0.07,0.06,0.05,0.04,0.03,
                                       0.02,0.01,0.009,0.008,0.007,0.006,0.005,0.004,
                                       0.003,0.002,0.001,0.0009,0.0008,0.0007,0.0006,0.0005,
                                       0.0004,0.0003,0.0002,0.0001,0.00009,0.00008,0.00007,0.00006], dtype=numpy.float32)
        self._db["background"] = background
        #==== PERIODIC TABLE START ====#
        hydrogen  = numpy.zeros((2,32), dtype=numpy.float32)
        hydrogen[0,:] = numpy.array([0.000035644, 0.000035527, 0.000035377, 0.000035195, 0.00003498, 0.000034736, 0.00003446, 0.000034156,
                                     0.000033824, 0.000033467, 0.000033085, 0.00003268, 0.000032255, 0.00003181, 0.000031348, 0.00003087,
                                     0.000030378, 0.000029873, 0.000029359, 0.000028835, 0.000028304, 0.000027768, 0.000027227, 0.000026683,
                                     0.00002614, 0.000025595, 0.000025051, 0.000024511, 0.000023974, 0.000023441, 0.000022914, 0.000022394], dtype=numpy.float32)
        hydrogen[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["hydrogen"] = hydrogen
        helium  = numpy.zeros((2,32), dtype=numpy.float32)
        helium[0,:] = numpy.array([0.000034823, 0.000034576, 0.000034323, 0.00003406, 0.000033786, 0.000033495, 0.000033183, 0.000032852,
                                   0.000032501, 0.000032132, 0.000031737, 0.000031325, 0.000030892, 0.00003044, 0.00002997, 0.000029486,
                                   0.000028985, 0.000028474, 0.00002795, 0.000027416, 0.000026875, 0.000026328, 0.000025777, 0.000025225,
                                   0.00002467, 0.000024116, 0.000023562, 0.000023013, 0.000022471, 0.000021933, 0.000021402, 0.00002088], dtype=numpy.float32)
        helium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["helium"] = helium
        lithium  = numpy.zeros((2,32), dtype=numpy.float32)
        lithium[0,:] = numpy.array([0.097129, 0.094917, 0.093003, 0.091346, 0.08988, 0.088569, 0.08738, 0.086277,
                                    0.085248, 0.084267, 0.083324, 0.082407, 0.08149, 0.080584, 0.079673, 0.078751,
                                    0.077813, 0.076859, 0.075883, 0.074887, 0.073863, 0.072824, 0.071758, 0.070665,
                                    0.069546, 0.068416, 0.067265, 0.066087, 0.064893, 0.063688, 0.062462, 0.061231], dtype=numpy.float32)
        lithium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["lithium"] = lithium
        beryllium  = numpy.zeros((2,32), dtype=numpy.float32)
        beryllium[0,:] = numpy.array([0.40455, 0.38579, 0.37009, 0.35686, 0.34568, 0.33605, 0.32771, 0.32046,
                                      0.31402, 0.30826, 0.30302, 0.29821, 0.29369, 0.28941, 0.28531, 0.28133,
                                      0.27741, 0.27358, 0.26972, 0.26592, 0.26206, 0.25819, 0.25428, 0.25031,
                                      0.24631, 0.24227, 0.23815, 0.234, 0.22981, 0.22557, 0.22129, 0.21697], dtype=numpy.float32)
        beryllium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["beryllium"] = beryllium
        boron  = numpy.zeros((2,32), dtype=numpy.float32)
        boron[0,:] = numpy.array([0.69121, 0.63938, 0.59663, 0.56116, 0.5316, 0.50661, 0.4855, 0.46753,
                                  0.45211, 0.4387, 0.42691, 0.41643, 0.40693, 0.39827, 0.39024, 0.38271,
                                  0.37557, 0.36871, 0.36209, 0.35563, 0.34932, 0.34307, 0.33691, 0.33078,
                                  0.32467, 0.31859, 0.31253, 0.30647, 0.30041, 0.29437, 0.28833, 0.2823], dtype=numpy.float32)
        boron[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["boron"] = boron
        carbon  = numpy.zeros((2,32), dtype=numpy.float32)
        carbon[0,:] = numpy.array([0.97589, 0.87564, 0.79346, 0.72591, 0.67018, 0.62353, 0.58468, 0.55223,
                                   0.52498, 0.50186, 0.48206, 0.46495, 0.44997, 0.43674, 0.42486, 0.41408,
                                   0.40416, 0.39491, 0.38626, 0.37801, 0.3701, 0.36248, 0.35507, 0.34781,
                                   0.34072, 0.33371, 0.3268, 0.31997, 0.31321, 0.3065, 0.29986, 0.2933], dtype=numpy.float32)
        carbon[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["carbon"] = carbon
        nitrogen  = numpy.zeros((2,32), dtype=numpy.float32)
        nitrogen[0,:] = numpy.array([0.00070121, 0.00061362, 0.00054204, 0.00048341, 0.00043528, 0.00039511, 0.00036188, 0.00033444,
                                     0.00031164, 0.0002926, 0.00027657, 0.00026298, 0.00025134, 0.00024127, 0.00023248, 0.00022469,
                                     0.00021773, 0.00021142, 0.00020565, 0.00020029, 0.00019528, 0.00019054, 0.00018602, 0.00018168,
                                     0.00017749, 0.00017342, 0.00016947, 0.0001656, 0.00016182, 0.0001581, 0.00015446, 0.00015086], dtype=numpy.float32)
        nitrogen[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["nitrogen"] = nitrogen # liquid
        oxygen  = numpy.zeros((2,32), dtype=numpy.float32)
        oxygen[0,:] = numpy.array([0.0011208, 0.00096305, 0.00083427, 0.00072899, 0.00064275, 0.00057087, 0.00051164, 0.00046296,
                                   0.00042282, 0.00038954, 0.00036183, 0.00033857, 0.00031893, 0.00030219, 0.00028779, 0.00027529,
                                   0.00026432, 0.00025459, 0.00024585, 0.00023792, 0.00023064, 0.00022389, 0.00021759, 0.00021163,
                                   0.00020599, 0.00020058, 0.0001954, 0.00019039, 0.00018555, 0.00018086, 0.00017629, 0.00017186], dtype=numpy.float32)
        oxygen[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["oxygen"] = oxygen # liquid
        fluorine  = numpy.zeros((2,32), dtype=numpy.float32)
        fluorine[0,:] = numpy.array([1.2215, 1.0351, 0.88316, 0.75908, 0.65761, 0.57325, 0.50393, 0.44719,
                                     0.4006, 0.36223, 0.33047, 0.30407, 0.28201, 0.26344, 0.24772, 0.23428,
                                     0.22269, 0.21259, 0.20373, 0.19584, 0.18876, 0.18233, 0.17644, 0.17099,
                                     0.16589, 0.16111, 0.15657, 0.15225, 0.14811, 0.14414, 0.14031, 0.13661], dtype=numpy.float32)
        fluorine[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["fluorine"] = fluorine
        neon  = numpy.zeros((2,32), dtype=numpy.float32)
        neon[0,:] = numpy.array([0.0013129, 0.0011016, 0.0009293, 0.00078875, 0.00067393, 0.00057869, 0.0005006, 0.00043682,
                                 0.00038461, 0.00034178, 0.00030653, 0.00027742, 0.00025329, 0.00023318, 0.00021633, 0.00020212,
                                 0.00019006, 0.00017974, 0.00017082, 0.00016305, 0.00015621, 0.00015014, 0.00014469, 0.00013974,
                                 0.0001352, 0.00013099, 0.00012706, 0.00012338, 0.00011988, 0.00011656, 0.00011338, 0.00011032], dtype=numpy.float32)
        neon[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["neon"] = neon
        sodium  = numpy.zeros((2,32), dtype=numpy.float32)
        sodium[0,:] = numpy.array([1.9238, 1.6075, 1.3482, 1.1326, 0.9567, 0.81141, 0.69244, 0.59542,
                                   0.51624, 0.45153, 0.39854, 0.35503, 0.31923, 0.28967, 0.26519, 0.24482,
                                   0.22777, 0.21344, 0.2013, 0.19095, 0.18204, 0.1743, 0.1675, 0.16147,
                                   0.15607, 0.15117, 0.14668, 0.14252, 0.13863, 0.13494, 0.13144, 0.12809], dtype=numpy.float32) / 0.968
        sodium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["sodium"] = sodium
        magnesium  = numpy.zeros((2,32), dtype=numpy.float32)
        magnesium[0,:] = numpy.array([4.5707, 3.8017, 3.1723, 2.6568, 2.2345, 1.8821, 1.5892, 1.3506,
                                      1.156, 0.997, 0.86698, 0.7604, 0.67285, 0.6008, 0.54134, 0.4921,
                                      0.45117, 0.41697, 0.38828, 0.36404, 0.34339, 0.32569, 0.31039, 0.29703,
                                      0.28523, 0.27472, 0.26523, 0.25659, 0.24861, 0.24118, 0.23422, 0.22761], dtype=numpy.float32) / 1.738
        magnesium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["magnesium"] = magnesium
        aluminium  = numpy.zeros((2,32), dtype=numpy.float32)
        aluminium[0,:] = numpy.array([8.8504, 7.3452, 6.1126, 5.1024, 4.2742, 3.5821, 3.0131, 2.5482,
                                      2.1654, 1.8501, 1.5927, 1.3823, 1.2101, 1.0688, 0.95261, 0.8568,
                                      0.7776, 0.71186, 0.65706, 0.61118, 0.57252, 0.53974, 0.51174, 0.48763,
                                      0.46665, 0.44819, 0.43181, 0.4171, 0.40376, 0.39151, 0.38014, 0.36952], dtype=numpy.float32) / 2.7
        aluminium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["aluminium"] = aluminium
        silicon  = numpy.zeros((2,32), dtype=numpy.float32)
        silicon[0,:] = numpy.array([9.8679, 8.1768, 6.7902, 5.6527, 4.7193, 3.9368, 3.2925, 2.7666,
                                    2.3369, 1.9853, 1.6975, 1.4576, 1.2614, 1.1009, 0.96932, 0.86116,
                                    0.77205, 0.69839, 0.63733, 0.5865, 0.54397, 0.50815, 0.4778, 0.45189,
                                    0.42962, 0.41027, 0.39329, 0.37825, 0.36477, 0.35259, 0.34143, 0.33111], dtype=numpy.float32) / 2.3290
        silicon[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["silicon"] = silicon
        phosphorus  = numpy.zeros((2,32), dtype=numpy.float32)
        phosphorus[0,:] = numpy.array([9.2824, 7.6828, 6.3704, 5.2931, 4.4084, 3.6633, 3.0494, 2.5485,
                                       2.1396, 1.8053, 1.5319, 1.3082, 1.1248, 0.97272, 0.8475, 0.74476,
                                       0.66028, 0.59063, 0.53302, 0.48523, 0.44539, 0.41205, 0.38397, 0.36018,
                                       0.33985, 0.32236, 0.3072, 0.29388, 0.2821, 0.27156, 0.26204, 0.25334], dtype=numpy.float32) / 2.2
        phosphorus[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["phosphorus"] = phosphorus
        sulfur  = numpy.zeros((2,32), dtype=numpy.float32)
        sulfur[0,:] = numpy.array([13.269, 10.97, 9.0838, 7.5348, 6.2628, 5.1895, 4.3052, 3.584,
                                   2.9949, 2.514, 2.1207, 1.7991, 1.5356, 1.3196, 1.1422, 0.99528,
                                   0.87377, 0.77375, 0.69117, 0.62282, 0.566, 0.51853, 0.47873, 0.44513,
                                   0.41663, 0.39229, 0.37132, 0.3531, 0.33714, 0.323, 0.3104, 0.29899], dtype=numpy.float32) / 1.96
        sulfur[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["sulfur"] = sulfur
        chlorine  = numpy.zeros((2,32), dtype=numpy.float32)
        chlorine[0,:] = numpy.array([11.569, 9.5578, 7.9067, 6.5504, 5.4366, 4.4964, 3.7214, 3.0891,
                                     2.5729, 2.1512, 1.8066, 1.5246, 1.2937, 1.1044, 0.9491, 0.82146,
                                     0.7164, 0.62904, 0.55655, 0.49667, 0.44706, 0.40579, 0.3713, 0.34236,
                                     0.31793, 0.29718, 0.27946, 0.26417, 0.25088, 0.23924, 0.22895, 0.21974], dtype=numpy.float32) / 1.5625
        chlorine[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["chlorine"] = chlorine
        argon  = numpy.zeros((2,32), dtype=numpy.float32)
        argon[0,:] = numpy.array([0.013778, 0.011377, 0.0094046, 0.0077842, 0.0064529, 0.0053291, 0.0044025, 0.0036465,
                                  0.0030292, 0.0025247, 0.0021123, 0.001775, 0.0014989, 0.0012726, 0.001087, 0.00093451,
                                  0.00080913, 0.00070585, 0.00062059, 0.00054918, 0.00048978, 0.00044056, 0.00039956, 0.00036537,
                                  0.0003367, 0.00031249, 0.00029194, 0.00027438, 0.00025924, 0.00024611, 0.00023457, 0.00022438], dtype=numpy.float32) / 1.3954
        argon[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["argon"] = argon
        potassium  = numpy.zeros((2,32), dtype=numpy.float32)
        potassium[0,:] = numpy.array([9.0627, 7.4802, 6.1798, 5.1112, 4.2327, 3.4903, 2.8782, 2.3785,
                                      1.9707, 1.6375, 1.3653, 1.1428, 0.96062, 0.8115, 0.68931, 0.58907,
                                      0.50675, 0.43906, 0.3833, 0.3373, 0.29925, 0.26741, 0.24092, 0.21889,
                                      0.20047, 0.185, 0.17194, 0.16084, 0.15133, 0.14315, 0.13601, 0.12976], dtype=numpy.float32) /0.862
        potassium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["potassium"] = potassium
        calcium  = numpy.zeros((2,32), dtype=numpy.float32)
        calcium[0,:] = numpy.array([19.561, 16.14, 13.331, 11.02, 9.1207, 7.5153, 6.1913, 5.1102,
                                    4.2273, 3.5058, 2.9162, 2.4338, 2.0392, 1.7162, 1.4513, 1.2342,
                                    1.0559, 0.90943, 0.78889, 0.68953, 0.60751, 0.53937, 0.48241, 0.43515,
                                    0.39579, 0.3629, 0.33525, 0.31189, 0.29202, 0.275, 0.26031, 0.24749], dtype=numpy.float32) / 1.55
        calcium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["calcium"] = calcium
        scandium  = numpy.zeros((2,32), dtype=numpy.float32)
        scandium[0,:] = numpy.array([40.719, 33.602, 27.749, 22.934, 18.972, 15.628, 12.868, 10.614,
                                     8.7713, 7.2649, 6.0327, 5.0249, 4.1997, 3.5236, 2.9696, 2.5151,
                                     2.142, 1.8354, 1.5832, 1.3754, 1.2039, 1.0615, 0.94257, 0.84406,
                                     0.76222, 0.69398, 0.63683, 0.58867, 0.5479, 0.51319, 0.48336, 0.45755], dtype=numpy.float32) / 2.985
        scandium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["scandium"] = scandium
        #===========================================================#
        #titanium  = numpy.zeros((2,32), dtype=numpy.float32)
        #titanium[0,:] = numpy.array([], dtype=numpy.float32)
        #titanium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["titanium"] = titanium
        #vanadium  = numpy.zeros((2,32), dtype=numpy.float32)
        #vanadium[0,:] = numpy.array([], dtype=numpy.float32)
        #vanadium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["vanadium"] = vanadium
        #chromium  = numpy.zeros((2,32), dtype=numpy.float32)
        #chromium[0,:] = numpy.array([], dtype=numpy.float32)
        #chromium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["chromium"] = chromium
        #manganese  = numpy.zeros((2,32), dtype=numpy.float32)
        #manganese[0,:] = numpy.array([], dtype=numpy.float32)
        #manganese[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["manganese"] = manganese
        iron = numpy.zeros((2,32), dtype=numpy.float32)
        iron[0,:] = numpy.array([195.11, 161.91, 134.42, 111.65, 92.795, 76.82, 63.554, 52.464,
                                 43.268, 35.73, 29.55, 24.48, 20.319, 16.904, 14.099, 11.796,
                                 9.9028, 8.3457, 7.0648, 6.0095, 5.1396, 4.4164, 3.8102, 3.3113,
                                 2.8999, 2.5598, 2.2781, 2.0438, 1.8485, 1.6849, 1.5472, 1.4307], dtype=numpy.float32)
        iron[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["iron"] = iron
        #cobalt  = numpy.zeros((2,32), dtype=numpy.float32)
        #cobalt[0,:] = numpy.array([], dtype=numpy.float32)
        #cobalt[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["cobalt"] = cobalt
        #nickel  = numpy.zeros((2,32), dtype=numpy.float32)
        #nickel[0,:] = numpy.array([], dtype=numpy.float32)
        #nickel[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["nickel"] = nickel
        #zinc  = numpy.zeros((2,32), dtype=numpy.float32)
        #zinc[0,:] = numpy.array([], dtype=numpy.float32)
        #zinc[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["zinc"] = zinc
        #gallium  = numpy.zeros((2,32), dtype=numpy.float32)
        #gallium[0,:] = numpy.array([], dtype=numpy.float32)
        #gallium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["gallium"] = gallium
        #germanium  = numpy.zeros((2,32), dtype=numpy.float32)
        #germanium[0,:] = numpy.array([], dtype=numpy.float32)
        #germanium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["germanium"] = germanium
        #arsenic  = numpy.zeros((2,32), dtype=numpy.float32)
        #arsenic[0,:] = numpy.array([], dtype=numpy.float32)
        #arsenic[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["arsenic"] = arsenic
        #selenium  = numpy.zeros((2,32), dtype=numpy.float32)
        #selenium[0,:] = numpy.array([], dtype=numpy.float32)
        #selenium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["selenium"] = selenium
        #bromine  = numpy.zeros((2,32), dtype=numpy.float32)
        #bromine[0,:] = numpy.array([], dtype=numpy.float32)
        #bromine[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["bromine"] = bromine
        #krypton  = numpy.zeros((2,32), dtype=numpy.float32)
        #krypton[0,:] = numpy.array([], dtype=numpy.float32)
        #krypton[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["krypton"] = krypton
        #rubidium  = numpy.zeros((2,32), dtype=numpy.float32)
        #rubidium[0,:] = numpy.array([], dtype=numpy.float32)
        #rubidium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["rubidium"] = rubidium
        #strontium  = numpy.zeros((2,32), dtype=numpy.float32)
        #strontium[0,:] = numpy.array([], dtype=numpy.float32)
        #strontium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["strontium"] = strontium
        #zirconium  = numpy.zeros((2,32), dtype=numpy.float32)
        #zirconium[0,:] = numpy.array([], dtype=numpy.float32)
        #zirconium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["zirconium"] = zirconium
        #silver  = numpy.zeros((2,32), dtype=numpy.float32)
        #silver[0,:] = numpy.array([], dtype=numpy.float32)
        #silver[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["silver"] = silver
        #cadmium  = numpy.zeros((2,32), dtype=numpy.float32)
        #cadmium[0,:] = numpy.array([], dtype=numpy.float32)
        #cadmium[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["cadmium"] = cadmium
        #tin  = numpy.zeros((2,32), dtype=numpy.float32)
        #tin[0,:] = numpy.array([], dtype=numpy.float32)
        #tin[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["tin"] = tin
        #xenon  = numpy.zeros((2,32), dtype=numpy.float32)
        #xenon[0,:] = numpy.array([], dtype=numpy.float32)
        #xenon[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["xenon"] = xenon
        #platinum  = numpy.zeros((2,32), dtype=numpy.float32)
        #platinum[0,:] = numpy.array([], dtype=numpy.float32)
        #platinum[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["platinum"] = platinum
        #gold  = numpy.zeros((2,32), dtype=numpy.float32)
        #gold[0,:] = numpy.array([], dtype=numpy.float32)
        #gold[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["gold"] = gold
        #mercury  = numpy.zeros((2,32), dtype=numpy.float32)
        #mercury[0,:] = numpy.array([], dtype=numpy.float32)
        #mercury[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["mercury"] = mercury
        #lead  = numpy.zeros((2,32), dtype=numpy.float32)
        #lead[0,:] = numpy.array([], dtype=numpy.float32)
        #lead[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        #self._db["lead"] = lead
        #==== PERIODIC TABLE END ====#
        #acetone
        acetone  = numpy.zeros((2,32), dtype=numpy.float32)
        acetone[0,:] = numpy.array([0.4280578,0.3803566,0.3412944,0.3092235,0.2828075,0.2606861,0.2423011,0.2270072,
                                    0.2142132,0.2034178,0.1942322,0.1863379,0.1794807,0.1734611,0.1680980,0.1632737,
                                    0.1588725,0.1548097,0.1510274,0.1474574,0.1440611,0.1408117,0.1376756,0.1346258,
                                    0.1316627,0.1287576,0.1259100,0.1231115,0.1203606,0.1176455,0.1149727,0.1123471], dtype=numpy.float32)
        acetone[1,:] = numpy.array([0.3173510,0.3173510,0.2197610,0.1790720,0.1541960,0.1118100,0.0911546,0.0811129,
                                    0.3060280,0.5125620,0.1315450,0.0514360,0.0498023,0.0464912,0.0495933,0.0496523,
                                    0.0440621,0.0394977,0.0369741,0.0383190,0.0344689,0.0341939,0.0313055,0.0319722,
                                    0.0296632,0.0290961,0.0241669,0.0289827,0.0298754,0.0341069,0.2,0.2], dtype=numpy.float32)
        self._db["acetone"] = acetone
        #brandy
        #brandy = numpy.zeros((2,32), dtype=numpy.float32)
        #brandy[0,:] = numpy.array([0.513521, 0.484615, 0.447282, 0.424661, 0.396559, 0.365467, 0.34676, 0.324798,
        #                0.296147, 0.274737, 0.252235, 0.240639, 0.220688, 0.205598, 0.201137, 0.204955], dtype=numpy.float32)
        #brandy[1,:] = numpy.array([0.19207, 0.233871, 0.292077, 0.319927, 0.366121, 0.409953, 0.368271, 0.473703, 
        #                0.543965, 0.507476, 0.476013, 0.538484, 0.406722, 0.295989, 0.391178, 0.241448], dtype=numpy.float32)
        #self._db["brandy_chantre"] = brandy
        #whiskey
        whiskey      = numpy.zeros((2,32), dtype=numpy.float32)
        whiskey[0,:] = numpy.array([0.485780,0.428239,0.381159,0.342547,0.310790,0.284216,0.262178,0.243905,
                                    0.228674,0.215879,0.205052,0.195798,0.187816,0.180853,0.174700,0.169208,
                                    0.164240,0.159693,0.155489,0.151554,0.147836,0.144302,0.140913,0.137635,
                                    0.134467,0.131376,0.128361,0.125406,0.122515,0.119672,0.116881,0.114147], dtype=numpy.float32)
        whiskey[1,:] = numpy.array([0.344884,0.344884,0.283677,0.202893,0.137898,0.123795,0.0959015,0.0835776,
                                    0.067892,0.0531527,0.0605913,0.060274,0.0506022,0.045679,0.0434497,0.0333611,
                                    0.026748,0.035043,0.0283081,0.0278669,0.0244917,0.0328753,0.0297312,0.0245203,
                                    0.0253875,0.0182905,0.0169163,0.0168744,0.0160668,0.0169407,0.2,0.2], dtype=numpy.float32)
        self._db["whiskey"] = whiskey
        #olive oil
        foodoil      = numpy.zeros((2,32), dtype=numpy.float32)
        foodoil[0,:] = numpy.array([0.424244,0.381699,0.346801,0.318085,0.294364,0.274454,0.257833,0.243922,
                                    0.232202,0.222227,0.213654,0.206207,0.199657,0.193836,0.188577,0.183781,
                                    0.179343,0.175190,0.171279,0.167541,0.163947,0.160477,0.157097,0.153787,
                                    0.150549,0.147356,0.144209,0.141105,0.138039,0.135003,0.132005,0.129052], dtype=numpy.float32)
        foodoil[1,:] = numpy.array([0.122013,0.122013,0.102714,0.0979721,0.0798502,0.07638,0.0736009,0.0583113,
                                    0.052545,0.0496192,0.0542111,0.0464007,0.0526606,0.0438939,0.0334396,0.0362969,
                                    0.0292824,0.0330471,0.0323735,0.0369363,0.0424405,0.0476127,0.050306,0.0563072,
                                    0.0495172,0.0610223,0.0878202,0.107877,0.129557,0.234077,0.2,0.2], dtype=numpy.float32)
        self._db["olive_oil"] = foodoil
        #h2o
        water = numpy.zeros((2,32), dtype=numpy.float32)
        water[0,:] = numpy.array([0.792222,0.686820,0.600710,0.530234,0.472427,0.424156,0.384293,0.351437,
                                  0.324239,0.301591,0.282625,0.266598,0.252969,0.241243,0.231057,0.222124,
                                  0.214193,0.207072,0.200597,0.194652,0.189132,0.183963,0.179088,0.174433,
                                  0.169990,0.165706,0.161575,0.157554,0.153661,0.149862,0.146161,0.142555], dtype=numpy.float32)
        water[1,:] = numpy.array([0.12739,0.12739,0.124258,0.106016,0.0931632,0.0853029,0.0786953,0.0719939,
                                  0.055668,0.0582602,0.0574707,0.0593487,0.0485454,0.0442016,0.0456244,0.0411078,
                                  0.0344179,0.0308835,0.03869,0.0364248,0.0491754,0.0525321,0.0490353,0.0512206,
                                  0.0704719,0.0729645,0.0735425,0.100361,0.127124,0.199913,0.2,0.2], dtype=numpy.float32)
        self._db["h2o"] = water
        #water-solved natriumchloride (salt water)
        saltwater = numpy.zeros((2,32), dtype=numpy.float32)
        saltwater[0,:] = numpy.array([1.442402,1.220249,1.038247,0.8888116,0.7662185,0.6633695,0.5785749,0.5091156,
                                      0.4520908,0.4051404,0.3663749,0.334208,0.3074328,0.2849994,0.2660975,0.2500699,
                                      0.2363684,0.2245183,0.2141882,0.2051285,0.1970968,0.1899038,0.1834007,0.1774437,
                                      0.1719597,0.1668495,0.1620633,0.1575293,0.1532324,0.149122,0.1451806,0.1413919], dtype=numpy.float32)
        saltwater[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["saltwater"] = saltwater
        #h2o2 30%
        hydrogenperoxide = numpy.zeros((2,32), dtype=numpy.float32)
        hydrogenperoxide[0,:] = numpy.array([0.884351,0.763981,0.665675,0.585253,0.519326,0.464313,0.418924,0.381559,
                                             0.350676,0.325007,0.303559,0.285482,0.270158,0.257020,0.245653,0.235725,
                                             0.226951,0.219110,0.212012,0.205526,0.199530,0.193937,0.188684,0.183683,
                                             0.178925,0.174350,0.169947,0.165669,0.161536,0.157508,0.153586,0.149771], dtype=numpy.float32)
        hydrogenperoxide[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["h2o2_30p"] = hydrogenperoxide
        #h2o2 50%
        h2o2_50p = numpy.zeros((2,32), dtype=numpy.float32)
        h2o2_50p[0,:] = numpy.array([0.8883058,0.7679259,0.6696058,0.5891654,0.5232155,0.4681766,0.4227582,0.3853604,
                                     0.3544407,0.3287325,0.307243,0.2891214,0.2737506,0.260563,0.2491443,0.2391631,
                                     0.2303343,0.2224374,0.2152828,0.2087379,0.2026834,0.1970306,0.1917176,0.1866558,
                                     0.1818375,0.177201,0.1727379,0.1683999,0.1642066,0.160119,0.1561389,0.1522662], dtype=numpy.float32)
        h2o2_50p[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["h2o2_50p"] = h2o2_50p
        #h2o2 70%
        h2o2_70p = numpy.zeros((2,32), dtype=numpy.float32)
        h2o2_70p[0,:] = numpy.array([0.8924644,0.7709902,0.6717824,0.5906225,0.5240901,0.4685723,0.4227671,0.3850595,
                                     0.3538927,0.3279883,0.306344,0.2881013,0.2726369,0.2593781,0.2479065,0.2378872,
                                     0.2290327,0.2211199,0.2139575,0.2074113,0.2013609,0.1957164,0.1904155,0.1853683,
                                     0.1805669,0.1759491,0.1715061,0.1671891,0.1630177,0.1589525,0.1549951,0.1511453], dtype=numpy.float32)
        h2o2_70p[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["h2o2_70p"] = h2o2_70p
        #methanol
        methanol = numpy.zeros((2,32), dtype=numpy.float32)
        methanol[0,:] = numpy.array([0.500602,0.440448,0.391239,0.350892,0.317719,0.289969,0.266967,0.247910,
                                     0.232039,0.218722,0.207467,0.197861,0.189590,0.182387,0.176035,0.170378,
                                     0.165271,0.160607,0.156303,0.152282,0.148492,0.144893,0.141450,0.138123,
                                     0.134912,0.131783,0.128734,0.125748,0.122830,0.119962,0.117149,0.114395], dtype=numpy.float32)
        methanol[1,:] = numpy.array([0.144582,0.144582,0.125186,0.0979409,0.0895314,0.0716303,0.0707077,0.0583783,
                                     0.0546311,0.0604197,0.0603526,0.0540158,0.0512275,0.04634,0.0423878,0.0400672,
                                     0.0313959,0.0341956,0.0399997,0.0414573,0.0508537,0.056462,0.0623056,0.0552322,
                                     0.0602428,0.0683871,0.0844006,0.0964388,0.152488,0.208154,0.2,0.2], dtype=numpy.float32)
        self._db["methanol"] = methanol
        #alcohol 40%
        alcohol = numpy.zeros((2,32), dtype=numpy.float32)
        alcohol[0,:] = numpy.array([0.48578,0.428239,0.381159,0.342547,0.31079,0.284216,0.262178,0.243905,
                                    0.228674,0.215879,0.205052,0.195798,0.187816,0.180853,0.1747,0.169208,
                                    0.16424,0.159693,0.155489,0.151554,0.147836,0.144302,0.140913,0.137635,
                                    0.134467,0.131376,0.128361,0.125406,0.122515,0.119672,0.116881,0.114147], dtype=numpy.float32)
        alcohol[1,:] = numpy.array([0.12739,0.12739,0.124258,0.106016,0.0931632,0.0853029,0.0786953,0.0719939,
                                  0.055668,0.0582602,0.0574707,0.0593487,0.0485454,0.0442016,0.0456244,0.0411078,
                                  0.0344179,0.0308835,0.03869,0.0364248,0.0491754,0.0525321,0.0490353,0.0512206,
                                  0.0704719,0.0729645,0.0735425,0.100361,0.127124,0.199913,0.2,0.1], dtype=numpy.float32)
        self._db["alcohol_40p"] = alcohol
        #benzene
        benzene = numpy.zeros((2,32), dtype=numpy.float32)
        benzene[0,:] = numpy.array([0.3760934,0.340134,0.3106186,0.2863107,0.2662075,0.2493327,0.2352239,0.2233825,
                                    0.2133799,0.2048374,0.1974654,0.1910386,0.1853563,0.1802899,0.175687,0.1714694,
                                    0.1675476,0.1638589,0.1603736,0.1570265,0.1537952,0.1506656,0.1476047,0.1445984,
                                    0.1416487,0.1387307,0.1358457,0.1329958,0.1301721,0.1273681,0.124593,0.121855], dtype=numpy.float32)
        benzene[1,:] = numpy.array([0.12739,0.12739,0.124258,0.106016,0.0931632,0.0853029,0.0786953,0.0719939,
                                  0.055668,0.0582602,0.0574707,0.0593487,0.0485454,0.0442016,0.0456244,0.0411078,
                                  0.0344179,0.0308835,0.03869,0.0364248,0.0491754,0.0525321,0.0490353,0.0512206,
                                  0.0704719,0.0729645,0.0735425,0.100361,0.127124,0.199913,0.2,0.1], dtype=numpy.float32)
        self._db["benzene"] = benzene
        #butane
        butane = numpy.zeros((2,32), dtype=numpy.float32)
        butane[0,:] = numpy.array([1.055754,0.9642608,0.8890077,0.8268627,0.7752839,0.7318039,0.6952455,0.664355,
                                   0.6380461,0.6153637,0.5955786,0.5781254,0.5625001,0.5483792,0.5353817,0.5233199,
                                   0.5119639,0.5011626,0.4908514,0.4808618,0.4711453,0.4616738,0.4523609,0.4431787,
                                   0.4341399,0.4251789,0.4163041,0.4075286,0.3988274,0.3901887,0.3816417,0.3732067], dtype=numpy.float32)
        butane[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["butane"] = butane
        #octane
        octane = numpy.zeros((2,32), dtype=numpy.float32)
        octane[0,:] = numpy.array([0.2996331,0.2732561,0.2515686,0.2336668,0.2188175,0.2063085,0.1958006,0.1869316,
                                   0.179388,0.1728942,0.1672395,0.1622606,0.1578118,0.1537998,0.1501143,0.1467007,
                                   0.1434928,0.1404467,0.1375432,0.1347338,0.1320042,0.1293459,0.1267341,0.1241603,
                                   0.121628,0.1191182,0.1166331,0.1141762,0.1117404,0.1093219,0.1069291,0.1045677], dtype=numpy.float32)
        octane[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["octane"] = octane
        #ethylene
        ethylene = numpy.zeros((2,32), dtype=numpy.float32)
        ethylene[0,:] = numpy.array([0.3926122,0.3574946,0.3286303,0.3048153,0.2850728,0.2684534,0.2545056,0.2427463,
                                    0.2327578,0.2241726,0.2167097,0.2101513,0.2043029,0.1990399,0.1942153,0.1897554,
                                    0.1855725,0.1816074,0.1778339,0.1741877,0.170649,0.1672061,0.1638262,0.1604975,
                                    0.1572239,0.1539806,0.15077,0.1475963,0.1444501,0.1413263,0.1382354,0.1351853], dtype=numpy.float32)
        ethylene[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["ethylene"] = ethylene
        #nitric acid HNO3
        nitric_acid = numpy.zeros((2,32), dtype=numpy.float32)
        nitric_acid[0,:] = numpy.array([1.180153,1.018563,0.886619,0.7786963,0.6902517,0.6164907,0.5556601,0.5056079,
                                        0.4642602,0.4299247,0.4012564,0.3771252,0.3566924,0.3392034,0.3240997,0.310929,
                                        0.2993118,0.2889523,0.2795925,0.2710525,0.2631749,0.2558333,0.2489452,0.2423976,
                                        0.2361662,0.2301787,0.2244204,0.2188246,0.2134141,0.20814,0.2030027,0.1979955], dtype=numpy.float32)
        nitric_acid[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["nitric_acid"] = nitric_acid
        #nitroglycerine
        nitroglycerine = numpy.zeros((2,32), dtype=numpy.float32)
        nitroglycerine[0,:] = numpy.array([1.114319,0.968004,0.848465,0.750625,0.670366,0.603391,0.548083,0.502479,
                                           0.464722,0.43327,0.406918,0.384647,0.365695,0.349396,0.335226,0.322791,
                                           0.311743,0.301813,0.292783,0.284478,0.276756,0.269517,0.262675,0.256128,
                                           0.249868,0.243814,0.237959,0.232251,0.226705,0.221275,0.215967,0.210787], dtype=numpy.float32)
        nitroglycerine[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["nitroglycerine"] = nitroglycerine
        #nitromethane
        nitromethane = numpy.zeros((2,32), dtype=numpy.float32)
        nitromethane[0,:] = numpy.array([0.7779627,0.6776961,0.5957632,0.5286777,0.4736257,0.4276627,0.3896757,0.3583258,
                                         0.332335,0.310656,0.2924556,0.2770436,0.2638947,0.2525537,0.2426669,0.2339602,
                                         0.2261971,0.2191963,0.2128066,0.2069086,0.2014099,0.196236,0.191331,0.1866309,
                                         0.1821207,0.1777549,0.1735261,0.1693983,0.1653806,0.1614454,0.1575962,0.1538299], dtype=numpy.float32)
        nitromethane[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["nitromethane"] = nitromethane
        #================================ S O L I D S =================================================================#
        #magnesium carbonate
        mgco3 = numpy.zeros((2,32), dtype=numpy.float32)
        mgco3[0,:] = numpy.array([3.846276,3.25009,2.762492,2.363363,2.036325,1.763497,1.53735,1.352446,
                                  1.200931,1.076398,0.9737467,0.8887408,0.8180823,0.759021,0.7093627,0.6673412,
                                  0.6315007,0.6006612,0.5738804,0.550394,0.5295611,0.510906,0.4940382,0.4785758,
                                  0.4643145,0.4510042,0.4385001,0.4266327,0.4153377,0.4044924,0.3940584,0.3839801], dtype=numpy.float32)
        mgco3[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["magnesium_carbonate"] = mgco3
        #calcium carbonate (argonite)
        caco3_argonite = numpy.zeros((2,32), dtype=numpy.float32)
        caco3_argonite[0,:] = numpy.array([15.59111,12.91411,10.71632,8.909707,7.424488,6.170468,5.136212,4.291261,
                                           3.600732,3.0358,2.57348,2.194518,1.883722,1.628492,1.418383,1.24525,
                                           1.102253,0.9838504,0.8855002,0.8035305,0.7349466,0.6771036,0.6279233,0.5862003,
                                           0.5506064,0.5199959,0.4934602,0.4702401,0.4497699,0.4315275,0.4151308,0.4002546], dtype=numpy.float32)
        caco3_argonite[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["caco3_argonite"] = caco3_argonite
        #calcium carbonate (calcite)
        caco3_calcite = numpy.zeros((2,32), dtype=numpy.float32)
        caco3_calcite[0,:] = numpy.array([14.93551,12.37108,10.26571,8.535059,7.112292,5.911003,4.920237,4.110815,
                                          3.449324,2.908146,2.465266,2.102239,1.804512,1.560015,1.358741,1.192888,
                                          1.055903,0.9424801,0.8482654,0.7697425,0.7040425,0.6486317,0.6015195,0.5615509,
                                          0.5274536,0.4981304,0.4727105,0.4504667,0.4308573,0.413382,0.3976748,0.3834241], dtype=numpy.float32)
        caco3_calcite[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["caco3_calcite"] = caco3_calcite
        #calcium chloride (anhydrate)
        cacl2 = numpy.zeros((2,32), dtype=numpy.float32)
        cacl2[0,:] = numpy.array([19.98491,16.50052,13.63939,11.28794,9.355622,7.723626,6.378012,5.279746,
                                  4.382979,3.650276,3.051502,2.561522,2.160556,1.832093,1.562672,1.341509,
                                  1.15971,1.009418,0.8852056,0.782719,0.6979509,0.6274745,0.5685712,0.5194182,
                                  0.4781946,0.4434503,0.4139951,0.3888334,0.3671767,0.3484047,0.3319786,0.3174535], dtype=numpy.float32)
        cacl2[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["calciumchloride"] = cacl2
        #calcium fluoride
        calciumfluoride = numpy.zeros((2,32), dtype=numpy.float32)
        calciumfluoride[0,:] = numpy.array([22.30724,18.44423,15.27304,12.6666,10.52418,8.715642,7.224409,6.006541,
                                            5.011637,4.198151,3.532823,2.987934,2.541506,2.175372,1.874479,1.627005,
                                            1.423088,1.254726,1.115391,0.9997343,0.9034574,0.8227201,0.7544954,0.6971087,
                                            0.6485426,0.6072325,0.571765,0.5411288,0.5144132,0.4909488,0.4701189,0.4514498], dtype=numpy.float32)
        calciumfluoride[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["calciumfluoride"] = calciumfluoride
        #c4 simulant
        c4_sim = numpy.zeros((2,32), dtype=numpy.float32)
        c4_sim[0,:] = numpy.array([4.57441,4.45061,3.98734,3.67612,3.40915,3.14569,2.82519,2.45746,
                                   2.11713,1.79833,1.54839,1.32525,1.14771,1.01042,0.89022,0.78912,
                                   0.66208,0.62869,0.60813,0.55208,0.54241,0.50140,0.45468,0.43012,
                                   0.41229,0.39564,0.37457,0.36354,0.35889,0.35097,0.91521,1.15252], dtype=numpy.float32)
        c4_sim[1,:] = numpy.array([2.31549,2.15692,1.78920,1.45028,1.18470,1.02429,0.84428,0.71676,
                                   0.61670,0.52066,0.44893,0.37872,0.33802,0.29414,0.25367,0.22720,
                                   0.19242,0.18402,0.17751,0.16172,0.14892,0.14513,0.13243,0.13428,
                                   0.13597,0.13539,0.13720,0.14009,0.14271,0.19667,0.89372,1.21496], dtype=numpy.float32)
        self._db["c4_simulant"] = c4_sim
        #c4 real
        c4_real = numpy.zeros((2,32), dtype=numpy.float32)
        c4_real[0,:] = numpy.array([0.52803,0.48827,0.43550,0.39111,0.35546,0.32443,0.29744,0.27284,
                                    0.25223,0.23551,0.22198,0.20936,0.19998,0.19041,0.18499,0.17855,
                                    0.16321,0.16044,0.16455,0.15839,0.16103,0.15499,0.15106,0.14565,
                                    0.14406,0.14034,0.14161,0.13579,0.13678,0.15072,0.47525,0.87010], dtype=numpy.float32)
        c4_real[1,:] = numpy.array([0.083140,0.076524,0.067064,0.060454,0.056145,0.050993,0.046899,0.043476,
                                    0.040009,0.036085,0.033112,0.029280,0.028588,0.027861,0.028098,0.029173,
                                    0.025776,0.025640,0.026726,0.027751,0.029734,0.028688,0.031656,0.031981,
                                    0.031682,0.033601,0.034798,0.038878,0.041917,0.127444,0.536553,0.649601], dtype=numpy.float32)
        self._db["c4_real"] = c4_real
        #c4 NIST
        c4_nist = numpy.zeros((2,32), dtype=numpy.float32)
        c4_nist[0,:] = numpy.array([1.09186,0.9529,0.83934,0.74634,0.670008,0.606278,0.553589,0.510083,
                                    0.473989,0.443862,0.418542,0.397082,0.378747,0.362915,0.349094,0.3369,
                                    0.326009,0.31617,0.307174,0.298854,0.291085,0.283761,0.276803,0.270128,
                                    0.263709,0.257488,0.251454,0.245557,0.239807,0.234169,0.228648,0.223235], dtype=numpy.float32)
        c4_nist[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["c4_NIST"] = c4_nist
        #table salt
        salt = numpy.zeros((2,32), dtype=numpy.float32)
        salt[0,:] = numpy.array([11.45697,9.48127,7.859584,6.525192,5.430373,4.508999,3.750214,3.13122,
                                 2.625881,2.213058,1.875585,1.599243,1.372878,1.187126,1.034496,0.9088466,
                                 0.8051732,0.7188309,0.6469628,0.5873235,0.5376093,0.495966,0.4608684,0.43114,
                                 0.4057623,0.3839437,0.3650288,0.3484662,0.3338191,0.3207552,0.3089855,0.2982653], dtype=numpy.float32)
        salt[1,:] = numpy.array([0.01,]*32, dtype=numpy.float32)
        self._db["salt"] = salt
        #glass
        glass = numpy.zeros((2,32), dtype=numpy.float32)
        glass[0,:] = numpy.array([4.905712,4.099320,3.438588,2.896690,2.452233,2.080064,1.773639,1.523172,
                                  1.318165,1.149983,1.011868,0.896654,0.801876,0.723652,0.658844,0.604934,
                                  0.559858,0.521946,0.489852,0.462493,0.438973,0.418554,0.400685,0.384851,
                                  0.370729,0.357968,0.346336,0.335611,0.325661,0.316339,0.307538,0.299183], dtype=numpy.float32)
        glass[1,:] = numpy.array([0.481135,0.481135,0.462714,0.540329,0.613885,0.836955,0.832325,0.86271,
                                  0.500753,0.435465,0.461401,0.406146,0.335508,0.305731,0.251714,0.217797,
                                  0.187805,0.181421,0.160164,0.152522,0.142129,0.131301,0.120181,0.114494,
                                  0.112182,0.100412,0.103111,0.0923112,0.0962553,0.0910129,0.1,0.1], dtype=numpy.float32)
        self._db["glass"] = glass
        plexiglass = numpy.zeros((2,32), dtype=numpy.float32)
        plexiglass[0,:] = numpy.array([0.6609166, 0.5846956, 0.5223124, 0.471131, 0.4290142, 0.393777, 0.3645351, 0.3402576,
                                  0.319996, 0.3029483, 0.2884918, 0.2761136, 0.2654084, 0.2560527, 0.2477595, 0.2403372,
                                  0.2336014, 0.2274154, 0.221683, 0.2162976, 0.2111956, 0.2063317, 0.201654, 0.1971174,
                                  0.1927207, 0.1884193, 0.1842113, 0.1800807, 0.176027, 0.1720298, 0.1680983, 0.1642395], dtype=numpy.float32)
        plexiglass[1,:] = numpy.array([0.481135,0.481135,0.462714,0.540329,0.613885,0.836955,0.832325,0.86271,
                                  0.500753,0.435465,0.461401,0.406146,0.335508,0.305731,0.251714,0.217797,
                                  0.187805,0.181421,0.160164,0.152522,0.142129,0.131301,0.120181,0.114494,
                                  0.112182,0.100412,0.103111,0.0923112,0.0962553,0.0910129,0.1,0.1], dtype=numpy.float32)
        self._db["plexiglass"] = plexiglass
        #plexiglass - to be done using 3D spectral data statistics
        #plexiglass = numpy.zeros((2,16), dtype=numpy.float32)
        #plexiglass[0,:] = numpy.array([], dtype=numpy.float32)
        #plexiglass[1,:] = numpy.array([], dtype=numpy.float32)
        #self._db["plexiglass"] = plexiglass
        # ================ #

    def getNumberOfMaterials(self):
        return len(self._keys)

    def getMaterialIndex(self, name):
        return self._keys.index(name)

    def getMaterialName(self, index):
        return self._keys[index]

    def getNumberOfSolids(self):
        return len(self._solids)

    def getNumberOfLiquids(self):
        return len(self._liquids)

    def getSolidIndex(self, name):
        return self._solids.index(name)

    def getLiquidIndex(self, name):
        return self._liquids.index(name)

    def getSolidName(self, index):
        return self._solids[index]

    def getLiquidName(self, index):
        return self._liquids[index]

    def getParameters(self, material):
        return self._db[material]
    
    def __del__(self):
        self._db.clear()
    
    def __str__(self):
        print(self._db)