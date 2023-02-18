from PIL import Image
import numpy as np


p = './result_gray/101.jpg'


img2 = np.asarray(Image.open(p).convert('L'))

print(set(img2.flatten()))
print('ok')