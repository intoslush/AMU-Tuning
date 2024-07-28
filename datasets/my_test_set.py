import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader,listdir_nohidden
import pickle

template = ['a photo of a {}.']


class MyDataSet(DatasetBase):

    dataset_dir = 'TrainSet'

    def __init__(self, root, num_shots,if_load):
        """if_load,1是保存,2是加载,3是加载最好的"""
        self.root=root
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, 'images')
        # self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        # self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')
        class_name_dic_name=self.ret_class_name_dic(root)

        self.template = template
        train, val, test=self.read_and_split_data(p_trn=0.5,p_val=0.1,class_name_dic_name=class_name_dic_name)

        # train, val, test = self.read_split(self.split_path, self.image_dir)
        if if_load==1:
            dic={'train':train,'val':val,'test':test}
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            with open('datasets/pic_slect', 'wb') as file:
                pickle.dump(dic, file)
        elif if_load==2:
            dic={'train':train,'val':val,'test':test}
            train = self.generate_fewshot_dataset(train, num_shots=num_shots)
            dic={}
            with open('datasets/pic_slect', 'wb') as file:
                dic = pickle.load(file)
            if if_load==3:
                with open('datasets/pic_slect_best', 'wb') as file:
                    dic = pickle.load(file)
            train=dic['train']
            val=dic['val']
            test=dic['test']
        super().__init__(train_x=train, val=val, test=test)
    
    @staticmethod
    def save_split(train, val, test, filepath, path_prefix):
        def _extract(items):
            out = []
            for item in items:
                impath = item.impath
                label = item.label
                classname = item.classname
                impath = impath.replace(path_prefix, '')
                if impath.startswith('/'):
                    impath = impath[1:]
                out.append((impath, label, classname))
            return out
        
        train = _extract(train)
        val = _extract(val)
        test = _extract(test)

        split = {
            'train': train,
            'val': val,
            'test': test
        }

        write_json(split, filepath)
        print(f'Saved split to {filepath}')
    

    def read_and_split_data(
        self,
        p_trn=0.5,
        p_val=0.2,
        class_name_dic_name={}
    ):
        image_dir=self.dataset_dir
        categories=[]
        for i in os.listdir(image_dir):#第一层为数据集
            for j in os.listdir(os.path.join(image_dir, i)):#第二层为类
                categories.append(i+'/'+j)
        p_tst = 1 - p_trn - p_val
        print(f'Splitting into {p_trn:.0%} train, {p_val:.0%} val, and {p_tst:.0%} test')

        def _collate(ims, y, c):
            items = []
            for im in ims:
                item = Datum(
                    impath=im,
                    label=y, # is already 0-based
                    classname=c
                )
                items.append(item)
            return items

        train, val, test = [], [], []
        for category in categories:
            category_dir = os.path.join(image_dir, category)
            images = listdir_nohidden(category_dir)
            images = [os.path.join(category_dir, im) for im in images]
            random.shuffle(images)
            n_total = len(images)
            n_train = round(n_total * p_trn)
            n_val = round(n_total * p_val)
            n_test = n_total - n_train - n_val
            assert n_train > 0 and n_val > 0 and n_test > 0

            # if new_cnames is not None and category in new_cnames:
            #     category = new_cnames[category]
            # print(category,len(categories),categories[0].split('/')[1],class_name_dic_name[category.split('/')[1]])
        
            # return 1, 1 ,1
            label=int(class_name_dic_name[category.split('/')[1]])
            train.extend(_collate(images[:n_train], label, category))
            val.extend(_collate(images[n_train:n_train+n_val], label, category))
            test.extend(_collate(images[n_train+n_val:], label, category))
        
        return train, val, test
    
    def ret_class_name_dic(self,root)->dict:
        """返回动物名字到数字和数字映射到动物名的字典"""
        classes = open(os.path.join(root,'classname.txt')).read().splitlines()#这是一个包含所有类的列表
        # class_name_dic_num={}
        class_name_dic_name={}
        for i in classes:
            name,idx = i.split(' ')
            c = name
            if c.startswith('Animal'):
                c = c[7:]
            if c.startswith('Thu-dog'):
                c = c[8:]
            if c.startswith('Caltech-101'):
                c = c[12:]
            if c.startswith('Food-101'):
                c = c[9:]
            if c not in class_name_dic_name:
                class_name_dic_name[c]=idx
            else:
                print(name,"already exist!!")
        return class_name_dic_name