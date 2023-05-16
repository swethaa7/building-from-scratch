from conv import Conv3x3
from maxpool import MaxPool
from softmax import SoftMax
import numpy as np
from PIL import Image

import mnist

train_images = mnist.train_images()
train_labels = mnist.train_labels()

test_images = mnist.test_images()[:1000]
test_labels = mnist.test_labels()[:1000]

conv = Conv3x3(8)   # 28x28x1 -> 26x26x8
pool = MaxPool()    # 26x26x8 -> 13x13x8
soft = SoftMax(13*13*8, 10)    # 13x13x8 -> 10

# output = conv.forward(train_images[13])
# output = pool.forward(output)

# print(output.shape)

# image_array_uint8 = (output * 255).astype(np.uint8)
# image = Image.fromarray(image_array_uint8)
# image.save("Convolutional Neural Networks\image.png")
# image.show()

def forward(image, label):
    out =  conv.forward((image / 255) - 0.5)
    out = pool.forward(out)
    out = soft.forward(out)

    loss = -np.log(out[label])
    acc = 1 if np.argmax(out) == label else 0

    return out, loss, acc

print('CNN initialised on MNIST!!')

loss = 0
num_correct = 0

for i, (im, im_label) in enumerate(zip(train_images, train_labels)):
    _, l, acc = forward(im, im_label)
    loss += l
    num_correct += acc

    if i % 100 == 99:
        print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
        loss = 0
        num_correct = 0
