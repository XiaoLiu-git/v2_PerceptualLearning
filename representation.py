import pdb

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
import numpy as np
from stimulus import Gabor, NoiseGrating

plt.style.use("fivethirtyeight")

class GenImg:
    def __init__(self, Sti, size=None, orient=0, loc="L", noise_cutout=0.2, diff=1):
        """

        :param size: [400, 200]
        :param orient: theta: 36 or 126
        :param loc: "L" or "R"
        :param noise_cutout: 0 ~ 1
        :param diff: 0-2 Hard-->Easy
        :param var_noise: variation of diff level --> noise: 1
        """

        if size is None:
            size = [400, 200]
        self.w, self.h = size
        self.orient = orient
        self.loc = loc
        self.noise_cutout = noise_cutout
        self.diff = diff
        self.Sti = Sti

    def gen_reference(self):

        self.label = 0 ## label +1, 0, -1 clockwise,reference,coounterclockwise

        sti = self.Sti.generate([self.w // 2, self.h], self.orient)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg[:self.w // 2, :] += sti
        else:
            self.tg[self.w // 2:, :] += sti
        return self.tg

    def gen_train(self, diff=None):
        if diff is not None:
            self.diff = diff

        self.label = np.sign(np.random.rand(1) - 0.5)
        example_orient = self.label*diff_level(self.diff)+self.orient

        sti = self.Sti.generate([self.w // 2, self.h], example_orient)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout

        if self.loc == "L":
            self.tg[:self.w // 2, :] += sti
        else:
            self.tg[self.w // 2:, :] += sti
        return self.label, self.tg

    def gen_test(self, diff=None):
        if diff is not None:
            self.diff = diff

        self.label = 1
        example_orient = self.label * diff_level(self.diff)+self.orient
        sti_p = self.Sti.generate([self.w // 2, self.h], example_orient)
        self.tg_p = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout

        self.label = -1
        example_orient = self.label * diff_level(self.diff)+self.orient
        sti_n = self.Sti.generate([self.w // 2, self.h], example_orient)
        self.tg_n = (np.random.rand(self.w,
                                    self.h) - 0.5) * 2 * self.noise_cutout
        if self.loc == "L":
            self.tg_p[:self.w // 2, :] += sti_p
            self.tg_n[:self.w // 2, :] += sti_n
        else:
            self.tg_p[self.w // 2:, :] += sti_p
            self.tg_n[self.w // 2:, :] += sti_n
        return self.tg_p, self.tg_n

    def loc_gen_train(self, diff=None):
        # diff = 5,10,20 limit49, transfer in different location
        if diff is not None:
            self.diff = diff

        self.label = np.sign(np.random.rand(1) - 0.5)
        example_loc = np.int(self.label * diff_level(self.diff))

        sti = self.Sti.generate([self.w // 2, self.h], self.orient)
        self.tg = (np.random.rand(self.w,
                                  self.h) - 0.5) * 2 * self.noise_cutout

        sti = np.roll(sti, example_loc, axis=0)##TODO

        if self.loc == "L":
            self.tg[:self.w // 2, :] += sti
        else:
            self.tg[self.w // 2:, :] += sti
        # pdb.set_trace()

        return self.label, self.tg

    def loc_gen_test(self, diff=None):
        # diff = 5,10,20 limit49, transfer in different location
        if diff is not None:
            self.diff = diff

        sti = self.Sti.generate([self.w // 2, self.h], self.orient)
        self.tg_p = (np.random.rand(self.w,
                                    self.h) - 0.5) * 2 * self.noise_cutout
        self.tg_n = (np.random.rand(self.w,
                                    self.h) - 0.5) * 2 * self.noise_cutout

        self.label = 1
        loc_p = np.int(self.label * diff_level(self.diff))
        sti_p = np.roll(sti,loc_p,axis=0)

        self.label = -1
        loc_n = np.int(self.label * diff_level(self.diff))
        sti_n = np.roll(sti,loc_n,axis=0)

        if self.loc == "L":
            self.tg_p[:self.w // 2, :] += sti_p
            self.tg_n[:self.w // 2, :] += sti_n
        else:
            self.tg_p[self.w // 2:, :] += sti_p
            self.tg_n[self.w // 2:, :] += sti_n
        # pdb.set_trace()

        return self.tg_p, self.tg_n

def show_img(img):
    plt.figure()
    plt.imshow(img)
    plt.show()
    plt.show(block=False)
    plt.pause(0.5)
    plt.close()

def diff_level(diff):

    jitter = (diff+1)*2+np.random.normal(diff+1)
    return jitter


def representation(img, num_x=40, num_theta=18):
    [w, h] = img.shape
    activity = np.zeros([num_x, num_theta])
    func_size = [w // 4, h, num_theta]
    basis_gabor = np.zeros(func_size)
    Sti = Gabor(sigma=30, freq=.01)
    for theta in range(num_theta):
        # pdb.set_trace()
        basis_gabor[:, :, theta] = Sti.generate(func_size[:-1],
                                               theta * 180 / num_theta)
        # show_img(basis_gabor[:,:,theta])
    for x in range(num_x):
        center = w * (x+5) // num_x
        # pdb.set_trace()
        img_cut = np.roll(img, center, axis=0)[:func_size[0], :]
        # show_img(img_cut)
        for theta in range(num_theta):
            # pdb.set_trace()
            activity[x, theta] = (img_cut * basis_gabor[:, :, theta]).sum()
    return activity


def df_representation(reference,img):
    ref_act = representation(reference)
    img_act = representation(img)
    df_act = ref_act-img_act
    return df_act


def merger_representation(reference,img):
    ref_act = representation(reference)
    img_act = representation(img)
    return np.concatenate((ref_act, img_act), axis=1)

for i in range(18):
    sti = Gabor(sigma=20, freq=0.02)
    # sti = NoiseGrating(sigma=5, num_grat=15)
    genimg = GenImg(sti, orient=i*10, loc="R", diff=0)

    img_tg_p, img_tg_n = genimg.gen_test()
    ref = genimg.gen_reference()
    # print(label)
    show_img(img_tg_p)
    show_img(ref)
    activity_tg = representation(img_tg_p)
    show_img(activity_tg)
    activity_tg = activity_tg-representation(ref)
    show_img(activity_tg)
    activity_tg = representation(img_tg_n)
    show_img(activity_tg)
    activity_tg = activity_tg-representation(ref)
    show_img(activity_tg)

# genimg = GenImg(orient=0, loc="L", diff=0)
#
# label, img_tg = genimg.gen_train()
#
# show_img(img_tg.T)
#
# activity_tg = representation(img_tg)
# show_img(activity_tg)
# print(label)
#
