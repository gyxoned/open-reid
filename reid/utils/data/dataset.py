from __future__ import print_function
import os.path as osp

import re
import numpy as np

from ..serialization import read_json


def _pluck(img_path, identities, indices, relabel=False, start_idx=0):
    ret = []
    for index, pid in enumerate(indices):
        pid_images = identities[pid]
        for camid, cam_images in enumerate(pid_images):
            for fname in cam_images:
                name = osp.splitext(fname)[0]
                x, y, _ = map(int, name.split('_'))
                assert pid == x and camid == y
                if relabel:
                    ret.append((osp.join(img_path,fname), start_idx+index, camid))
                else:
                    ret.append((osp.join(img_path,fname), pid, camid))
    return ret


class Dataset(object):
    def __init__(self, root, split_id=0):
        self.root = root
        self.split_id = split_id
        self.meta = None
        self.split = None
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        # return osp.join(self.root, 'images')
        return self.root

    def load(self, name, num_val=0.3, verbose=True, start_idx=0):
        splits = read_json(osp.join(self.root, name, 'splits.json'))
        if self.split_id >= len(splits):
            raise ValueError("split_id exceeds total splits {}"
                             .format(len(splits)))
        self.split = splits[self.split_id]

        # Randomly split train / val
        trainval_pids = sorted(np.asarray(self.split['trainval']))
        #np.random.shuffle(trainval_pids)
        num = len(trainval_pids)
        if isinstance(num_val, float):
            num_val = int(round(num * num_val))
        if num_val >= num or num_val < 0:
            raise ValueError("num_val exceeds total identities {}"
                             .format(num))
        train_pids = sorted(trainval_pids[:-num_val])
        val_pids = sorted(trainval_pids[-num_val:])

        img_path = osp.join(name, 'images')

        self.meta = read_json(osp.join(self.root, name, 'meta.json'))
        identities = self.meta['identities']
        self.train = _pluck(img_path, identities, train_pids, relabel=True, start_idx=start_idx)
        self.val = _pluck(img_path, identities, val_pids, relabel=True, start_idx=start_idx)
        self.trainval = _pluck(img_path, identities, trainval_pids, relabel=True, start_idx=start_idx)
        self.query = _pluck(img_path, identities, self.split['query'])
        self.gallery = _pluck(img_path, identities, self.split['gallery'])
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(trainval_pids)

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(self.split['query']), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(self.split['gallery']), len(self.gallery)))

    def _check_integrity(self, name):
        return osp.isdir(osp.join(self.root, name, 'images')) and \
               osp.isfile(osp.join(self.root, name, 'meta.json')) and \
               osp.isfile(osp.join(self.root, name, 'splits.json'))



def _pluck_msmt(list_file, subdir, pattern=re.compile(r'([-\d]+)_([-\d]+)_([-\d]+)')):
    with open(list_file, 'r') as f:
        lines = f.readlines()
    ret = []
    pids = []
    for line in lines:
        line = line.strip()
        fname = line.split(' ')[0]
        pid, _, cam = map(int, pattern.search(osp.basename(fname)).groups())
        if pid not in pids:
            pids.append(pid)
        ret.append((osp.join(subdir,fname), pid, cam))
    return ret, pids

class Dataset_MSMT(object):
    def __init__(self, root):
        self.root = root
        self.train, self.val, self.trainval = [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0

    @property
    def images_dir(self):
        return osp.join(self.root, 'raw', 'MSMT17_V1')

    def load(self, verbose=True):
        exdir = osp.join(self.root, 'raw', 'MSMT17_V1')
        self.train, train_pids = _pluck_msmt(osp.join(exdir, 'list_train.txt'), 'train')
        self.val, val_pids = _pluck_msmt(osp.join(exdir, 'list_val.txt'), 'train')
        self.trainval = self.train + self.val
        self.query, query_pids = _pluck_msmt(osp.join(exdir, 'list_query.txt'), 'test')
        self.gallery, gallery_pids = _pluck_msmt(osp.join(exdir, 'list_gallery.txt'), 'test')
        self.num_train_ids = len(train_pids)
        self.num_val_ids = len(val_pids)
        self.num_trainval_ids = len(list(set(train_pids).union(set(val_pids))))

        if verbose:
            print(self.__class__.__name__, "dataset loaded")
            print("  subset   | # ids | # images")
            print("  ---------------------------")
            print("  train    | {:5d} | {:8d}"
                  .format(self.num_train_ids, len(self.train)))
            print("  val      | {:5d} | {:8d}"
                  .format(self.num_val_ids, len(self.val)))
            print("  trainval | {:5d} | {:8d}"
                  .format(self.num_trainval_ids, len(self.trainval)))
            print("  query    | {:5d} | {:8d}"
                  .format(len(query_pids), len(self.query)))
            print("  gallery  | {:5d} | {:8d}"
                  .format(len(gallery_pids), len(self.gallery)))
