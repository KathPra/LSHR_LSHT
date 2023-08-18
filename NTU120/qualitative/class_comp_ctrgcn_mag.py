from cmath import exp
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt

# Plot Confusion matrix
def plot_CM(confusion, save_name):
    fig, ax = plt.subplots(figsize=(7.5,7.5))
    fig = plt.imshow(confusion, cmap='hot', interpolation='nearest', origin="upper")
    ax.xaxis.tick_top()
    ax.set_xlabel('Predictions', fontsize=15)
    ax.set_ylabel('Actuals', fontsize=15)
    # ax.set_xlim(0,120)
    # ax.set_ylim(120,0)
    plt.title('Confusion Matrix', fontsize=18)
    cbar = plt.colorbar(fig)
    # cbar.solids.set_edgecolor("face")
    # plt.draw()
    plt.savefig(f"ConfusionMatrix_{save_name}.png")
    plt.close()

# load class dict
ntu120_class = np.loadtxt("/home/prasse/Documents/ActionRecognition/SphericalHarm/CTR-GCN/qualitative/ntu120_classes.txt", delimiter='\t',dtype=str)
#class_ind = np.arange(0,120,1)
#class_dict = dict(zip(class_ind, ntu120_class))
#print(class_dict)

# load data
def load_data(file_name):
    folder ="/home/prasse/Documents/ActionRecognition/SphericalHarm/CTR-GCN/work_dir/"
    file=file_name+"_test_each_class_acc.csv"
    experiments = genfromtxt(folder+file, delimiter=',')
    #print(experiments.shape)
    accuracy = experiments[0] # for each class
    confusion = experiments[1:] # for each class i: true label (row), j: prediction (column)
    return accuracy, confusion

def top_flop(accuracy, class_dict):
    accuracy = np.around(accuracy, decimals = 3)
    acc_list = zip(class_dict, accuracy)
    acc = sorted(acc_list, key = lambda acc_list: acc_list[1])
    sorted_dict = dict(acc)

 
    max = np.flip(accuracy[-5:])
    amax = np.flip(np.argsort(accuracy)[-5:]) 
    max_label = class_dict[amax]
    best_dict = dict(zip(max_label, max))

    min = np.sort(accuracy)[:5]
    amin = np.argsort(accuracy)[:5]
    print(amin)
    min_label = class_dict[amin]
    worst_dict = dict(zip(min_label, min))
    print("Best: ", best_dict, "Worst: ", worst_dict)
    return best_dict, worst_dict, sorted_dict

def plot_acc(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        epoch = np.arange(0,120,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    plt.xlim(left=-0.25, right=120)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/home/prasse/Documents/ActionRecognition/SphericalHarm/CTR-GCN/qualitative/{save_name}_class_accuracy.png")
    plt.close()

def plot_acc_extr(acc_list, acc_label, save_name):
    # Plot accuracy
    acc_count = 0
    for i in acc_list:
        i = i[20:41]
        epoch = np.arange(20,41,1)
        plt.bar(epoch + 0.25*acc_count, i, label =acc_label[acc_count], width=0.25)
        acc_count+=1 
    plt.title(f'Class accuracy comparison')
    plt.xlabel('Class')
    #plt.xlim(left=60, right=81)
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f"/home/prasse/Documents/ActionRecognition/SphericalHarm/CTR-GCN/qualitative/{save_name}_class_accuracy_extract2.png")
    plt.close()


def model_comp(acc1, acc2, class_dict,save_name):
    diff1 = acc2-acc1
    diff1 = np.around(diff1, decimals = 3)
    diff1 = zip(class_dict, diff1, acc1, acc2)
    diff1 = sorted(diff1, key = lambda diff1: diff1[1])

    f = open("qualitative/class_comp_LSHT_XSUB_joint_mag.txt", "a")
    f.write(save_name+": first is better"+"\n")
    f.writelines(str(item)+"\n" for item in diff1 )
    f.write(save_name+": second is better"+"\n")
    f.write("\n")
    #f.writelines(str(item) for item in diff2 )
    f.close()

# ## Baseline (CSET)
# print("Baseline CSET")
# acc1, conf1 = load_data("cset/baseline_imp/epoch64")
# plot_CM(conf1, "baseline_CSET")
# BaselineIMP_t5, BaselineIMP_l5,  = top_flop(acc1, ntu120_class)

## Baseline
print("CTRGCN JOINT LSHR (M)")
acc2, conf2 = load_data("csub/ctrgcn_local_SHT_mag/epoch60")
plot_CM(conf2, "ctrgcn_LSHT")
Baseline_t5, Baseline_l5, LSHR_M = top_flop(acc2, ntu120_class)

## Baseline_imp
print("CTRGCN")
acc3, conf3 = load_data("csub/ctrgcn/epoch58")
plot_CM(conf3, "ctrgcn_LSHT_BL")
BaselineIMP_t5, BaselineIMP_l5,CTRGCN = top_flop(acc3, ntu120_class)



results = [LSHR_M, CTRGCN]

## Plot accuracies of models
label_list =  ["CTRGCN LSHR X-Sub Joint","CTRGCN X-Sub Joint" ]


model_comp(acc2, acc3, ntu120_class, "NTU120_xsub_joint_mag")



plot_acc([acc2, acc3], label_list, "NTU120_xsub_joint_mag")
# plot_acc_extr([acc2, acc3, acc4, acc5], label_list, "Local_Spher_Coord")
# plot_acc([acc4, acc5], label_list1, "Local_Spher_Coord_excl.")
# plot_acc_extr([acc4, acc5], label_list1, "Local_Spher_Coord_excl.")
# plot_acc([acc3, acc6, acc5], label_list2, "Comp._Spher_Coord_excl.")
# plot_acc_extr([acc3, acc6, acc5], label_list2, "Comp._Local_Spher_Coord_excl.")


# for i in ['A65. tennis bat swing.', "A66. juggling table tennis balls.",'A67. hush (quite).', 'A68. flick hair.', "A69. thumb up.",'A70. thumb down.','A71. make ok sign.']:
#     print(i)
#     for j in results:
#         print(j[i])

# for i in ['A72. make victory sign.','A73. staple book.', 'A74. counting money.' ,'A75. cutting nails.','A76. cutting paper (using scissors).', 'A77. snapping fingers.']:
#     print(i)
#     for j in results:
#         print(j[i])


# # for i in ['A78. open bottle.','A79. sniff (smell).']:
# #     print(i)
# #     for j in results:
# #         print(j[i])



# # for i in ["A36. shake head.","A37. wipe face.","A38. salute.","A39. put the palms together.","A40. cross hands in front (say stop).","A41. sneeze/cough.","A42. staggering.","A43. falling."]:
# #     print(i)
# #     for j in results:
# #         print(j[i])