import argparse
import pickle
import os

import numpy as np
from tqdm import tqdm
import csv


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu120/xsub', 'ntu120/xset'},
                        help='the work folder for storing results')

    parser.add_argument('--acc_dir',
                        help='Directory containing "epoch?_test_each_class.csv" ')

    parser.add_argument('--best_epoch',
                        help='Number of best epoch')

    arg = parser.parse_args()

    dataset = arg.dataset
    if 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]

    else:
        raise NotImplementedError

    with open(os.path.join(arg.acc_dir, 'epoch1_test_score.pkl'), 'rb') as r1:
        r = list(pickle.load(r1).items())

    counter = 0
    hand_joints = "(A1) drink water, (A2) eat meal/snack, (A3) brushing teeth, (A4) brushing hair, (A10) clapping, (A11) reading,(A12) writing, (A13) tear up paper, (A28) make a phone call/answer phone, (A29) playing with phone/tablet, (A30) typing on a keyboard, (A31) pointing to something with finger, (A32) taking a selfie, (A33) check time (from watch), (A34) rub two hands together, (A37) wipe face, (A39) put the palms together, (A49) use a fan (with hand or paper)/feeling warm, (A54) point finger at the other person, (A58) handshaking, (A67) hush (quite), (A68) flick hair, (A69) thumb up, (A70) thumb down, (A71) make ok sign, (A72) make victory sign, (A73) staple book, (A74) counting money, (A75) cutting nails, (A76) cutting paper (using scissors), (A77) snapping fingers, (A78) open bottle, (A81) toss a coin, (A82) fold paper, (A83) ball up paper, (A84) play magic cube, (A85) apply cream on face, (A86) apply cream on hand back, (A89) put something into a bag, (A90) take something out of a bag, (A91) open a box, (A93) shake fist, (A105) blow nose, (A110) shoot at other person with a gun, (A113) cheers and drink, (A115) take a photo of other person, (A112) high-five, (A120) finger-guessing game (playing rock-paper-scissors)"
    hand_joints = hand_joints.split(",")
    for hand_class in hand_joints:
        hand_class = int("".join([s for s in hand_class if s.isdigit()]))-1
        hand_joints[counter] = hand_class
        counter += 1
    #print(hand_joints)

    counter = 0
    with open('{}/epoch{}_test_each_class_acc.csv'.format(arg.acc_dir, arg.best_epoch), newline='') as f:
    #    writer = csv.writer(f)
    #    writer.writerow(each_acc)
    #    writer.writerows(confusion)
        acc_reader = csv.reader(f, delimiter=' ', quotechar='|')
        for row in acc_reader:
            if counter == 0:
                class_acc = row
            else:
                break
            counter += 1
            #print(', '.join(row))
    # confusion = confusion_matrix(label_list, pred_list) #true label, pred
    
    list_class_acc = []
    for x in class_acc:
        list_class_acc = x.split(",")

    class_acc_hand = []
    for index, acc in enumerate(list_class_acc):
        if index in hand_joints:
            acc = float(acc)
            class_acc_hand.append(acc)

    #print(class_acc_hand)
    #print(len(class_acc_hand), len(hand_joints))
    
    conv_matrix = np.loadtxt(open('{}/epoch{}_test_each_class_acc.csv'.format(arg.acc_dir, arg.best_epoch), "rb"), delimiter=",", skiprows=1)
    for index in range(120):
        if index not in hand_joints:
            conv_matrix[index,:] = np.zeros(120)
    conv_total = conv_matrix.sum()
    conv_corr = conv_matrix.diagonal().sum()

    # load accucary from best file
    # right_num = total_num = right_num_5 = 0
    # right_num_hand = total_num_hand = right_num_5_hand = 0

    # for i in tqdm(range(len(label))):
    #     l = label[i]
    #     _, acc = r[i]
    #     rank_5 = acc.argsort()[-5:]
    #     right_num_5 += int(int(l) in rank_5)
    #     acc = np.argmax(acc)
    #     right_num += int(acc == int(l))
    #     total_num += 1
    #     if l in hand_joints:
    #         right_num_hand += int(acc == int(l))
    #         total_num_hand += 1
    #         right_num_5_hand += int(int(l) in rank_5)

    # acc = right_num / total_num
    # acc5 = right_num_5 / total_num

    # acc_hand = right_num_hand / total_num_hand
    # acc5_hand = right_num_5_hand / total_num_hand
    
    # print("Accuracy over all classes")
    # print('Top1 Acc: {:.4f}%'.format(acc * 100))
    # print('Top5 Acc: {:.4f}%'.format(acc5 * 100))

    print("Accuracy over all hand classes")
    acc_sum = sum(class_acc_hand)
    #print('Top1 Acc: {:.4f}%'.format(acc_sum/len(class_acc_hand) * 100))
    print('Top1 Acc: {:.4f}%'.format(conv_corr/conv_total * 100))

    #print('Top1 Acc: {:.4f}%'.format(acc_hand * 100))
    #print('Top5 Acc: {:.4f}%'.format(acc5_hand * 100))
