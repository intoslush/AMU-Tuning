import os
import math
import random
from collections import defaultdict

import torchvision.transforms as transforms

from .utils import Datum, DatasetBase, read_json, write_json, build_data_loader,listdir_nohidden


template = ['a photo of a {}.']


class MyDataSet2(DatasetBase):

    dataset_dir = 'TestSetA'

    def __init__(self, root, num_shots):
        self.root=root
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, 'images')
        # self.anno_dir = os.path.join(self.dataset_dir, 'annotations')
        # self.split_path = os.path.join(self.dataset_dir, 'split_zhou_OxfordPets.json')
        class_name_dic_name=self.ret_class_name_dic(root)

        self.template = template
        train, val, test=self.read_and_split_data(p_trn=0.5,p_val=0.2,class_name_dic_name=class_name_dic_name)
        # train, val, test = self.read_split(self.split_path, self.image_dir)
        print("!!!!!")
        train = self.generate_fewshot_dataset(train, num_shots=num_shots)

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
        images=[]
        for i in os.listdir(image_dir):#第一层为数据集
            images.append(os.path.join(image_dir,i))
        

        def _collate(ims, y, c):
            items = []
            # print("ims[0]:",ims[0],ims[0].split('_')[1][:-5])
            for im in ims:
                item = Datum(
                    impath=im,
                    label=int(im.split('_')[1].split('.')[0]), 
                    classname=c
                )
                items.append(item)
            
            return items

        train, val, test = [], [], []

        test.extend(_collate(images, 0, "!forword"))
        return test[0:2], test[0:2], test
    
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