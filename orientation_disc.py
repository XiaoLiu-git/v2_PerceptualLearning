import pdb
from datetime import datetime
import copy
import numpy as np
import torch
import torch.optim as optim

import Network as nw
import representation as rp
import stimulus
import tools_PL


def GenTestData(_ori, num=10, loc="L", diff=0):
    num_ori = int(180 / _ori)
    dataset = np.zeros([num * 2 * num_ori, 1, 40, 18])
    label = np.zeros([num * 2 * num_ori])
    for i in range(num_ori):
        ori_i = i * _ori
        Sti = stimulus.Gabor(sigma=30, freq=0.01)
        Img = rp.GenImg(Sti, orient=ori_i, loc=loc, diff=diff)
        for ii in range(num):
            label[i * num * 2 + ii] = 1
            label[i * num * 2 + ii + num] = -1
            img = Img.gen_test()
            dataset[i * num * 2 + ii, :, :, :] = rp.representation(img[0])
            dataset[i * num * 2 + ii + num, :, :, :] = rp.representation(
                img[1])
    return dataset, label


def feedforward(x, W):
    """
    :param x:size [num * 2 * num_ori, 1, 40, 18]
    :param W: size[40, 18]
    :return: Y: same size as x, Y=x*W
    """
    return x * W


def ff_train(show_epc, net_name, slow_learning=True, bg_epoch=0):
    # input
    num_batch = 16
    # Sti = stimulus.NoiseGrating(sigma=3, num_grat=15)
    Sti = stimulus.Gabor(sigma=30, freq=0.01)
    Img_L = rp.GenImg(Sti, orient=45, loc="L", diff=0)
    Img_R = rp.GenImg(Sti, orient=45, loc="R", diff=0)
    Img_DT = rp.GenImg(Sti, orient=135, loc="R", diff=0)
    Img_th = rp.GenImg(Sti, orient=135, loc="L", diff=0)
    inputs = np.zeros([num_batch, 1, 40, 18])
    labels = np.zeros([num_batch])

    # network
    if net_name is "s_cc3":
        net = nw.Net_sCC()
    elif net_name is "s_cc5":
        net = nw.Net_sCC5()
    elif net_name is "s_fc3":
        net = nw.Net_sFC()
    else:
        # pdb.set_trace()
        net = nw.Net_sCC7()

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = optim.Adam([
        {'params': net.layer2.parameters()},
        {'params': net.readout.parameters()},
        {'params': net.layer1.parameters(), 'lr': 1e-3}
    ], lr=1e-3)
    running_loss = 0.0
    # generate testing data, select 40 < L,R,TH < 60
    acc_list = np.zeros([200 // show_epc + 1, 5])
    loss_list = np.zeros([200 // show_epc, 1])
    num_test = 20

    t_inputs = np.zeros([num_test * 4, 1, 40, 18])
    t_labels = np.zeros([num_test * 4])
    for i in range(num_test // 2):
        t_labels[2 * i] = 1
        t_labels[2 * i + 1] = -1
        # pdb.set_trace()
        img_tg = Img_L.gen_test()
        t_inputs[2 * i, :, :, :] = rp.representation(img_tg[0])
        t_inputs[2 * i + 1, :, :, :] = rp.representation(img_tg[1])

        t_labels[1 * num_test + 2 * i] = 1
        t_labels[1 * num_test + 2 * i + 1] = -1
        img_tg = Img_DT.gen_test()
        t_inputs[1 * num_test + 2 * i, :, :, :] = rp.representation(img_tg[0])
        t_inputs[1 * num_test + 2 * i + 1, :, :, :] = rp.representation(
            img_tg[1])

        t_labels[2 * num_test + 2 * i] = 1
        t_labels[2 * num_test + 2 * i + 1] = -1
        img_tg = Img_R.gen_test()
        t_inputs[2 * num_test + 2 * i, :, :, :] = rp.representation(img_tg[0])
        t_inputs[2 * num_test + 2 * i + 1, :, :, :] = rp.representation(
            img_tg[1])

        t_labels[3 * num_test + 2 * i] = 1
        t_labels[3 * num_test + 2 * i + 1] = -1
        img_tg = Img_th.gen_test()
        t_inputs[3 * num_test + 2 * i, :, :, :] = rp.representation(img_tg[0])
        t_inputs[3 * num_test + 2 * i + 1, :, :, :] = rp.representation(
            img_tg[1])

    _epc = 1
    size_input = t_inputs.shape
    # pdb.set_trace()
    t_inputs = np.reshape(
        tools_PL.normalize(np.reshape(t_inputs, [size_input[0], -1])),
        size_input)
    acc_list[0, 1:] = test(net, t_inputs, num_test, t_labels)[:-1]
    acc_list[0, 0] = acc_list[0, 1]
    best_acc = np.zeros(np.size(acc_list[0, :]))

    ## generate reference

    t_ref = np.zeros([15, 1, 40, 18])
    for i in range(5):
        img_tg = Img_L.gen_test()
        img_ref = Img_L.gen_reference()
        t_ref[i,:,:,:] = rp.representation(img_tg[0])
        t_ref[5+i,:,:,:] = rp.representation(img_tg[1])
        t_ref[10+i,:,:,:] = rp.representation(img_ref)
    size_input = t_ref.shape
    t_ref = np.reshape(tools_PL.normalize(np.reshape(t_ref,[size_input[0],-1])),size_input)

    # Training
    weight = np.ones([40, 18])
    for epoch in range(200):  # loop over the dataset multiple times

        for i in range(num_batch):
            labels[i], img_tg = Img_L.gen_train()
            inputs[i, :, :, :] = rp.representation(img_tg)

        size_input = inputs.shape
        inputs = np.reshape(
            tools_PL.normalize(np.reshape(inputs, [size_input[0], -1])),
            size_input)
        inputs = feedforward(inputs, weight)
        inputs = np.reshape(
            tools_PL.normalize(np.reshape(inputs, [num_batch, -1])), size_input)
        # if epoch > bg_epoch:
            # weight[25:35, :] = weight[25:35, :] + 0.05 * np.ones([10, 18])

        optimizer.zero_grad()
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)

        loss = criterion(outputs, torch.FloatTensor((labels + 1) // 2))
        # pdb.set_trace()
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if epoch % show_epc == show_epc - 1:  # print every show epoch
            print_test = False
            acc = tools_PL.np_acc(outputs, labels)
            acc_list[_epc, 0] = acc
            loss_list[_epc - 1, 0] = running_loss / show_epc
            if epoch % (show_epc * 10) == show_epc * 10 - 1:
                print('[%d] loss: %.6f' %
                      (epoch + 1, running_loss / show_epc))
                print('train acc: %.2f %%' % (acc * 100))
                print_test = True
                reference(net, t_ref)
            PATH = './net.pth'
            torch.save(net.state_dict(), PATH)
            running_loss = 0.0
            t_input_ff = feedforward(t_inputs, weight)
            size_input = t_inputs.shape
            t_input_ff = np.reshape(
                tools_PL.normalize(np.reshape(t_input_ff, [size_input[0], -1])),
                size_input)
            acc_test = test(net, t_input_ff, num_test, t_labels, print_test)
            acc_list[_epc, 1:] = acc_test[:-1]
            if acc_test[-1] > best_acc[-1]:
                best_net = copy.deepcopy(net)
                best_acc = acc_test
            _epc += 1


    print("Best Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
        best_acc[0] * 100,
        best_acc[1] * 100,
        best_acc[2] * 100,
        best_acc[3] * 100,
        best_acc[4] * 100))
    return acc_list, loss_list, net, weight, best_net


def test(net, inputs, num_batch, labels, prt=False):
    with torch.no_grad():
        b_x = torch.tensor(inputs, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)
        acc = ACC(outputs, labels, num_batch)
        if prt:
            print("Accuracy: {:.1f}% {:.1f}% {:.1f}% {:.1f}% {:.1f}%".format(
                acc[0] * 100,
                acc[1] * 100,
                acc[2] * 100,
                acc[3] * 100,
                acc[4] * 100
            ))
    return acc


def ACC(outputs, labels, num_batch):
    # pdb.set_trace()
    num_test = len(outputs) // num_batch
    acc = np.zeros(num_test + 1)

    total = (torch.sign(outputs * labels) + 1) / 2
    for i in range(num_test):
        acc[i] = torch.sum(total[num_batch * i:num_batch * (i + 1)]) / num_batch
    acc[-1] = torch.sum(total) / len(outputs)
    return acc


def reference(net, input):
    with torch.no_grad():
        b_x = torch.tensor(input, dtype=torch.float32)
        outputs = net(b_x)
        outputs = outputs.squeeze(1)
        print('###################')
        print(outputs)
        print(torch.sign(outputs))
        print('###################')

# plt.style.available
# plt.style.use('seaborn')
day = datetime.now().strftime('%Y-%m-%d')
foldername = './' + day
tools_PL.mkdir(foldername)
num_section = 50
num_test = 10
ori = 22.5
loc = 5
AccALL = np.zeros([num_section, 101, 5])
LossALL = np.zeros([num_section, 100, 1])
Acc_test = np.zeros([num_section, int(40 / loc), int(180 / ori)])
Acc_test_best = np.zeros([num_section, int(40 / loc), int(180 / ori)])
# Acc_test_db = np.zeros([num_section, 200, int(40 / loc), int(180 / ori)])
print_test = False
net_list = ['s_cc3']
t_data, t_label = GenTestData(_ori=ori)
size_input = t_data.shape
t_data = np.reshape(tools_PL.normalize(np.reshape(t_data, [size_input[0], -1])),
                    size_input)
# pdb.set_trace()

for net_name in net_list:
    for begin_epoch in [199]:
        for s in range(num_section):
            ### Training
            AccALL[s, :, :], LossALL[s, :, :], net, weight, bestNet = ff_train(
                show_epc=2,
                net_name=net_name,
                slow_learning=False,
                bg_epoch=begin_epoch)

            ### Testing for different location

            for i in range(40 // loc):
                y = feedforward(t_data, weight)
                size_input = t_data.shape
                y = np.reshape(
                    tools_PL.normalize(np.reshape(y, [size_input[0], -1])),
                    size_input)
                # plt.subplot(2, 4, i+1)
                # plt.imshow(y[1].squeeze())
                Acc_test[s, i, :] = test(net, y, num_test * 2, t_label,
                                            print_test)[:-1]
                Acc_test_best[s, i, :] = test(bestNet, t_data, num_test * 2,
                                               t_label,
                                               print_test)[:-1]
                t_data = np.roll(t_data, -5, axis=2)
            print(Acc_test_best[s])
            print(Acc_test[s])
            # pdb.set_trace()

            # ### Dobule training
            #
            # for db_i in range(200):
            #
            #     y = feedforward(t_data, weight)
            #     size_input = t_data.shape
            #     y = np.reshape(
            #         tools_PL.normalize(np.reshape(y, [size_input[0], -1])),
            #         size_input)
            #
            #     for i in range(40 // loc):
            #         y = feedforward(t_data, weight)
            #         size_input = t_data.shape
            #         y = np.reshape(
            #             tools_PL.normalize(np.reshape(y, [size_input[0], -1])),
            #             size_input)
            #         # plt.subplot(2, 4, i+1)
            #         # plt.imshow(y[1].squeeze())
            #         Acc_test_db[s, db_i, i, :] = test(net, y, num_test * 2,
            #                                              t_label,
            #                                              print_test)[:-1]
            #         t_data = np.roll(t_data, -5, axis=2)

                # weight[5:15, :] = weight[5:15, :] + 0.05 * np.ones([10, 18])

            # pdb.set_trace()
            # print(Acc_test_db[s,db_i])

        name_head = foldername + '/New/ori_disc_' + net_name
        np.save(name_head + 'AccALL.npy', AccALL)
        np.save(name_head + 'LossALL.npy', LossALL)
        np.save(name_head + 'Acc_test.npy', Acc_test)
        np.save(name_head + 'Acc_test_best.npy', Acc_test_best)
        # np.save(name_head + 'Acc_testdb.npy', Acc_test_db)
###TODO!!!把0.05改到模型里面去，有两个地方要同时改！！！
