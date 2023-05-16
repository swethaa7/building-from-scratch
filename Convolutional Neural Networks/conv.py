import numpy as np

class Conv3x3:
    def __init__(self, num_filters):
        self.num_filters = num_filters
        self.filters = np.random.randn(num_filters, 3, 3)/9
        #divide by 9 to reduce the variance
    
    def iterate_regions(self, image):
        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                im_region = image[i:(i+3), j:(j+3)]
                yield im_region, i, j

    def forward(self, input):
        # im_region is the relevant 3x3 array from image
        # output is the final (h-2, w-2, no. of filters) array that we get after conv
        h, w = input.shape
        output = np.zeros((h-2, w-2, self.num_filters))
        for im_region, i, j in self.iterate_regions(input):
            # we do element-wise multiplication and add them all and assign it to output[i, j]
            output[i, j] = np.sum(im_region * self.filters, axis = (1, 2))

        return output

