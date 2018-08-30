from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.data import Dataset_MSMT
from ..utils.osutils import mkdir_if_missing
from ..utils.serialization import write_json


class MSMT17(Dataset_MSMT):
    url = 'https://drive.google.com/file/d/1iGgqiV5kOt3xjjyg3rCAeA3_zULYJTNq/view'
    md5 = 'ea5502ae9dd06c596ad866bd1db0280d'

    def __init__(self, root, split_id=0, num_val=100, download=True):
        super(MSMT17, self).__init__(root)

        if download:
            self.download()

        self.load(num_val)

    def download(self):

        import re
        import hashlib
        import shutil
        from glob import glob
        from zipfile import ZipFile

        raw_dir = osp.join(self.root, 'raw')
        mkdir_if_missing(raw_dir)

        # Download the raw zip file
        fpath = osp.join(raw_dir, 'MSMT17_V1.tar.gz')
        if osp.isfile(fpath) and \
          hashlib.md5(open(fpath, 'rb').read()).hexdigest() == self.md5:
            print("Using downloaded file: " + fpath)
        else:
            raise RuntimeError("Please download the dataset manually from {} "
                               "to {}".format(self.url, fpath))

        # Extract the file
        exdir = osp.join(raw_dir, 'MSMT17_V1')
        if not osp.isdir(exdir):
            print("Extracting zip file")
            with ZipFile(fpath) as z:
                z.extractall(path=raw_dir)