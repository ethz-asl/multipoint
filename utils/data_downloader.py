from __future__ import print_function
import os
from tqdm import tqdm
import requests
# import zipfile

class MrDataGrabber(object):  # Does not work with bananas

    def __init__(self, url, target_path, chunk_size=1024):
        self.url = url
        self.path = target_path
        self.chunk_size = chunk_size
        self.target_basename = os.path.basename(self.url)
        self.target_file = os.path.join(self.path, self.target_basename)

    def download(self):
        if os.path.exists(self.target_file):
            print('Target {0} already exists, please remove if you wish to update.'.format(self.target_file))
        else:
            # Make target directory if required
            os.makedirs(self.path, exist_ok=True)

            filesize = int(requests.head(self.url).headers["Content-Length"])

            with requests.get(self.url, stream=True) as r, \
                    open(self.target_file, "wb") as f, \
                    tqdm(unit="B", unit_scale=True, unit_divisor=self.chunk_size, total=filesize,
                         desc=self.target_basename) as progress:
                for chunk in r.iter_content(chunk_size=self.chunk_size):
                    datasize = f.write(chunk)
                    progress.update(datasize)

            # if os.path.splitext(self.target_basename)[1].lower() == '.zip':
            #     self.unzip_data()
        return self.target_file

    # def unzip_data(self, target_dir=None, force=False):
    #     if target_dir is None:
    #         target_dir = os.path.join(self.path, os.path.splitext(self.target_basename)[0])
    #     if (not force) and os.path.isdir(target_dir):
    #         print('Target directory already exists, not unzipping.')
    #     else:
    #         zip_ref = zipfile.ZipFile(self.target_file, 'r')
    #         zip_ref.extractall(target_dir)
