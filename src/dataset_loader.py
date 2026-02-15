import os
import numpy as np
import OpenEXR
import Imath


def read_exr(path):
    exr = OpenEXR.InputFile(path)
    dw = exr.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

    R = np.frombuffer(exr.channel('R', FLOAT), dtype=np.float32)
    G = np.frombuffer(exr.channel('G', FLOAT), dtype=np.float32)
    B = np.frombuffer(exr.channel('B', FLOAT), dtype=np.float32)

    R = R.reshape((height, width))
    G = G.reshape((height, width))
    B = B.reshape((height, width))

    return np.stack([R, G, B], axis=-1)


def list_scenes(folder):
    return sorted([
        f.replace(".exr", "")
        for f in os.listdir(folder)
        if f.endswith(".exr")
    ])