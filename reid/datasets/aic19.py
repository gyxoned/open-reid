from __future__ import print_function, absolute_import
import os.path as osp
import xml.etree.ElementTree as ET

from ..utils.data import Dataset
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json

xmlp = ET.XMLParser(encoding="utf-8")

class AIC19(Dataset):
    url = 'https://www.aicitychallenge.org/track2-download/'
    md5 = '5e5d04de9dc978f273962748e72e396a'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(AIC19, self).__init__(root, split_id=split_id)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. " +
                               "You can use download=True to download it.")

        self.load(num_val)

    def download(self):
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'aic19-track2-reid.zip')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'AIC19-ReID')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)

        # Format
        images_dir = osp.join(self.root, 'images')
        mkdir_if_missing(images_dir)

        # 666 identities (+1 for background) with 40 camera views each
        identities = [[[] for _ in range(40)] for _ in range(666+2)]

        def register(subdir, name_txt, label_xml=None, unique_pid=None):
            with open(osp.join(exdir, name_txt), 'rb') as f:
                fnames = f.readlines()
            fpaths = [osp.join(exdir, subdir, n.strip().decode()) for n in fnames]
            if(label_xml is not None):
                raw_label = ET.parse(osp.join(exdir, label_xml),parser=xmlp).getroot()
                label_dir = {}
                for elem in raw_label:
                    for subelem in elem:
                        item = subelem.attrib
                        label_dir[item['imageName']] = {'pid':int(item['vehicleID']), 
                                                        'cam':int(item['cameraID'][1:])}
            pids = set()
            for fpath in fpaths:
                fname = osp.basename(fpath)
                if(label_xml is not None):
                    pid, cam = label_dir[fname]['pid'], label_dir[fname]['cam']
                # if pid == -1: continue  # junk images are just ignored
                    assert 1 <= pid <= 666  # pid == 0 means background
                    assert 1 <= cam <= 40
                else:
                    pid = unique_pid
                    cam = 1
                pid -= 1
                cam -= 1
                pids.add(pid)
                fname = ('{:08d}_{:02d}_{:04d}.jpg'
                         .format(pid, cam, len(identities[pid][cam])))
                identities[pid][cam].append(fname)
                shutil.copy(fpath, osp.join(images_dir, fname))
		     
            return pids

        trainval_pids = register('image_train', 'name_train.txt', label_xml='train_label.xml')
        gallery_pids = register('image_test', 'name_test.txt', unique_pid=667) 
        query_pids = register('image_query', 'name_query.txt', unique_pid=668)
        # assert query_pids <= gallery_pids
        # assert trainval_pids.isdisjoint(gallery_pids)

        # Save meta information into a json file
        meta = {'name': 'AIC19', 'shot': 'multiple', 'num_cameras': 40,
                'identities': identities}
        write_json(meta, osp.join(self.root, 'meta.json'))

        # Save the only training / test split
        splits = [{
            'trainval': sorted(list(trainval_pids)),
            'query': sorted(list(query_pids)),
            'gallery': sorted(list(gallery_pids))}]
        write_json(splits, osp.join(self.root, 'splits.json'))
