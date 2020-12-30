import glob
import os


def gen_txt(dirsrc):
    imglist = glob.glob(f'{dirsrc}/annotations/*.xml')
    imgnameList = [os.path.split(imgpath)[-1][: -4] for imgpath in imglist]
    imgnameList.sort()

    splits_dir = dirsrc
    traintest_split_f = os.path.join(splits_dir, "img_list" + '.txt')
    traintest_split = imgnameList

    with open(traintest_split_f, 'w') as fp:
        fp.write('\n'.join(traintest_split) + '\n')
    return 

def gen_txt_shore(dirsrc, dirdest, split):
    imglist = glob.glob(f'{dirsrc}/*.jpg')
    imgnameList = [os.path.split(imgpath)[-1][: -4] for imgpath in imglist]
    imgnameList.sort()

    splits_dir = dirdest
    traintest_split_f = os.path.join(splits_dir, "img_list_" + split + '.txt')
    traintest_split = imgnameList

    with open(traintest_split_f, 'w') as fp:
        fp.write('\n'.join(traintest_split) + '\n')
    return 


if __name__ == "__main__":
    # generate_dir()

    # dirsrc_list = ["/home/sun/projects/sar/SSDD/SSDD_train", "/home/sun/projects/sar/SSDD/SSDD_test"] 
    # for dirsrc in dirsrc_list:
    #     gen_txt(dirsrc)
    
    dirsrc_list = {'inshore': '/home/sun/projects/sar/SSDD/SSDD_r/JPEGImages_test_inshore', 'offshore': '/home/sun/projects/sar/SSDD/SSDD_r/JPEGImages_test_offshore'}
    dirdest = '/home/sun/projects/sar/SSDD/SSDD_test'
    for k, v in dirsrc_list.items():
        spilt = k
        dirsrc = v
        gen_txt_shore(dirsrc, dirdest, spilt)
    