# -*- coding: utf-8 -*-

import shutil
import os 

src = os.path.abspath(os.getcwd())
dst = "/home/qc/PowerFolders/Quantenchaos/Non_Weyl"

shutil.copytree(src, dst)