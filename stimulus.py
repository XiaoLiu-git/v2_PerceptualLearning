import pdb

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import numpy as np
import cv2


def grating(X, Y, params):
    """
    GRATING --  Sinusoidal grating (grayscale image).
    %
    %  G = grating(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct w/ fields 'amplit', 'freq', 'orient', and 'phase'.
    %  The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 15*pi/180 ;
    %    G = grating(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """
    A = params["amplit"]
    omega = 2 * np.pi * params["freq"]
    theta = params["orient"]
    phi = params["phase"]
    slant = X * (omega * np.cos(theta)) + Y * (
            omega * np.sin(theta))  # cf. function SLANT
    G = A * np.cos(slant + phi)
    return G


def gabor(X, Y, params):
    """
    %GABOR  --  Sinusoidal grating under a Gaussian envelope.
    %
    %  G = gabor(X,Y,params)
    %
    %  X and Y are matrices produced by MESHGRID (use integers=pixels).
    %  PARAMS is a struct with fields 'amplit', 'freq', 'orient', 'phase',
    %  and 'sigma'. The function GABOR_PARAMS supplies default parameters.
    %  G is a matrix of luminance values.  size(G)==size(X)==size(Y)
    %
    %  Example:
    %    x=[-100:+100] ; y=[-120:+120] ; [X,Y] = meshgrid(x,y) ;
    %    params = gabor_params ; params.orient = 60*pi/180 ;
    %    G = gabor(X,Y,params) ;
    %    imagesc1(x,y,G) ;

    """

    sigmasq = params["sigma"] ** 2
    Gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * sigmasq))
    Grating = grating(X, Y, params)
    G = Gaussian * Grating
    return G


class Gabor:
    def __init__(self, sigma=25, freq=0.2):
        self.params = {
            "amplit": 0.5,  # amplitude [luminance units], min=-A,max=+A
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": 0,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
        }

    def generate(self, size, orient):
        """
        :param size:
        :param orient: 0~360
        :return: G
        """
        x = np.arange(-size[1] // 2, size[1] // 2)
        y = np.arange(-size[0] // 2, size[0] // 2)
        X, Y = np.meshgrid(x, y)
        self.params["orient"] = orient * np.pi / 180
        self.G = gabor(X, Y, self.params)

        return self.G




    def show(self):
        plt.figure()
        plt.imshow(self.G)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()


class Vernier:
    def __init__(self, sigma=25, freq=0.02, var_noise=0.5):
        self.params = {
            "amplit": 1,  # amplitude [luminance units], min=-A,max=+A
            "freq": freq,  # spatial frequency [cycles/pixel]
            "orient": 0,  # orientation [radians]
            "phase": 0,  # phase [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
            "diff_level": [3, 5, 9],  # different of two gabor for vernier
            "var_n": var_noise  # var_noise: variation of diff level --> noise
        }

    def generate(self, size, orient, diff, label, var_noise):
        """
        :param size:
        :param orient: "V, H" or 0-180
        :param diff: 0,1,2 (easy medium hard)
        :param label: +1 or -1
        :param var_noise variation of diff level --> noise:1
        :return: G
        """
        if orient == 'V':
            _orient = 45
        elif orient == 'H':
            _orient = 135
        else:
            _orient = orient
        # generate Gabor
        x = np.arange(-size[1] // 4, size[1] // 4)
        y = np.arange(-size[0] // 4, size[0] // 4)
        X, Y = np.meshgrid(x, y)
        self.params["orient"] = 0 * np.pi / 180  # ori of gabor is 0
        self.params["var_n"] = var_noise
        G = gabor(X, Y, self.params)

        # arrange Gabor into Vernier
        diff_noi = np.abs(np.around(np.random.normal(0, self.params["var_n"])))
        jitter = int((self.params["diff_level"][diff] + diff_noi) * label)

        self.V = np.zeros(size)
        self.V[0:size[0] // 2,
        size[1] // 4 - jitter:size[1] * 3 // 4 - jitter] = G
        self.V[size[0] // 2:,
        size[1] // 4 + jitter:size[1] * 3 // 4 + jitter] = G
        # pdb.set_trace()
        ## rotating V
        self.V = rotation_matrix(self.V,_orient)
        return self.V

    def show(self):
        plt.figure()
        plt.imshow(self.V)
        plt.show(block=False)
        plt.pause(0.5)
        plt.close()


class NoiseGrating:
    def __init__(self, sigma=5, num_grat=20):
        self.params = {
            "amplit": 1,  # amplitude [luminance units], min=-A,max=+A
            "num_grat": num_grat,  # number of grating
            "orient": 0,  # orientation [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
        }

    def generate(self, size, orient):
        """
        :param size:
        :param orient: 0~360
        :return: G
        """
        x = np.arange(-size[1] // 2, size[1] // 2)
        y = np.arange(-size[0] // 2, size[0] // 2)
        X, Y = np.meshgrid(x, y)
        num_grat = self.params["num_grat"]
        Gaussian = np.exp(-(X ** 2 + Y ** 2) / (2 * self.params["sigma"]))
        gartingList = np.sort(np.random.choice(size[0], num_grat*2, replace=False))

        self.NG = np.zeros(size)
        for i in range(num_grat):
            self.NG[gartingList[2*i]:gartingList[2*i+1],:]=1
        self.NG = self.NG * Gaussian
        self.params["orient"] = orient
        self.NG = np.sign(self.NG)*self.params["amplit"]* np.random.normal(0,1,size)
        self.NG = rotation_matrix(self.NG, self.params["orient"])
        return self.NG

    def show(self):
        plt.figure()
        plt.imshow(self.NG)
        plt.show(block=False)
        plt.pause(1)
        plt.close()


class SymmetricDots:
    def __init__(self, sigma=2, var_noise=0.5,num_dotspair=18):
        self.params = {
            "amplit": 0.5,  # amplitude [luminance units], min=-A,max=+A
            "orient": 0,  # orientation [radians]
            "sigma": sigma,  # std.dev. of Gaussian envelope [pixels]
            "diff_level": [3, 5, 9],  # different of two gabor for vernier
            "var_n": var_noise,  # var_noise: variation of diff level --> noise
            "num_dp": num_dotspair # num_dotspair: the number of dot pairs also
            # is the size for index matrix of the symmetric dots
        }

    def generate(self,size, orient):
        num_dp=self.params['num_dp']
        num_half = num_dp//2
        index_mtr = np.zeros([num_dp, num_dp])
        columns=np.concatenate((np.arange(num_half-1),np.arange(num_half-1),np.random.choice(num_half-1, 2, replace=False)),axis=0)
        rows=np.random.choice(num_dp, num_dp, replace=False)
        index_mtr[rows,columns]=1
        index_mtr[rows, num_dp-columns-1] = 1 # symmetric dots in the other side

        dot_x = np.floor(size[1] / (num_dp*2))
        dot_y = np.floor(size[0] / (num_dp * 2))
        x = np.arange(-dot_x,dot_x) # the size of each dots
        y = np.arange(-dot_y, dot_y)

        index_mask = np.repeat(np.repeat(index_mtr,len(x),axis=1),len(y),axis=0)
        X, Y = np.meshgrid(x, y)

        sigmasq = self.params["sigma"] ** 2
        self.g = np.exp(-(X ** 2 + Y ** 2) / (2 * sigmasq))

        G = np.tile(self.g, (num_dp, num_dp))*index_mask

        ## Padding G into the given size
        if G.shape != size:
            y0=(size[0]-G.shape[0])//2
            x0 = (size[1] - G.shape[1]) // 2
            self.G = np.zeros(size)
            self.G[y0:y0+G.shape[0], x0:x0+G.shape[1]]= G

        ## Rotating G
        self.params['orient'] = orient
        self.G = rotation_matrix(self.G, orient)

        return self.G

    def show(self):
        plt.figure()
        plt.imshow(self.G)
        plt.show(block=False)
        plt.pause(5)
        plt.close()


def rotation_matrix(matrix, orientation):
    size = matrix.shape
    M = cv2.getRotationMatrix2D((size[0] // 2, size[1] // 2), orientation, 1)
    matrix = cv2.warpAffine(matrix, M, (size[0], size[1]))
    return matrix


## Test!!
# Sdot=SymmetricDots()
# Sdot.generate(size=(300,300))
# Sdot.show()
# G= Gabor(sigma=50, freq=0.02)
# G.generate(size=(200,200),orient=35)
# G.show()
# NG=NoiseGrating(sigma=5, num_grat=20)
# NG.generate(size=(200,200),orient=90)
# NG.show()