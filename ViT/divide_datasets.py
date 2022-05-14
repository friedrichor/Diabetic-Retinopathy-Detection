import os
import shutil
import pandas as pd


def data_process(file, path_ori, path_pro):  # csv文件，处理前文件路径，处理后文件路径
    df = pd.read_csv(file)
    df = df[['image', 'Retinopathy_grade']]
    # 按类创建处理后的图片文件夹
    for name in ['0-1', '2-3']:
        folder = path_pro + str(name)
        if os.path.exists(folder):
            shutil.rmtree(folder)  # 文件夹存在则删除(清空文件的目的)
        os.makedirs(folder)
    # 把grade=0-1的分到一个文件夹，grade=2-3的分到一个文件夹
    for index, row in df.iterrows():
        img, grade = row[0] + '.png', row[1]
        if grade is 0 or grade is 1:
            shutil.copy(path_ori + img, path_pro + '0-1/')
        else:
            shutil.copy(path_ori + img, path_pro + '2-3/')
    print('图片总量 =', len(os.listdir(path_ori)))
    print('grade=0-1图片数量 =', len(os.listdir(path_pro + '0-1/')))
    print('grade=2-3图片数量 =', len(os.listdir(path_pro + '2-3/')))


if __name__ == '__main__':
    file = 'data/Mess1_annotation_train.csv'
    path_ori = 'data/train/'
    path_pro = 'data/train_pro/'
    data_process(file, path_ori, path_pro)


