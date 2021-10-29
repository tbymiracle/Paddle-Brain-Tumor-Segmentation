import numpy as np

def model_gen(input_dim, x, y, slice_no):
    X1 = []
    X2 = []
    Y = []
    u = (y[:, slice_no, :])
    for i in range(int((input_dim) / 2), y.shape[0] - int((input_dim) / 2)):
        for j in range(int((input_dim) / 2), y.shape[2] - int((input_dim) / 2)):
            # Filtering all 0 patches
            if (x[i - 16:i + 17, j - 16:j + 17, :].any != 0):
                X2.append(x[i - 16:i + 17, j - 16:j + 17, :])
                X1.append(x[i - int((input_dim) / 2):i + int((input_dim) / 2) + 1,
                          j - int((input_dim) / 2):j + int((input_dim) / 2) + 1, :])
                Y.append(y[i, slice_no, j])


    X1 = np.asarray(X1)
    X2 = np.asarray(X2)
    Y = np.asarray(Y)
    d = [X1, X2, Y]
    return d


def data_gen(data, y, slice_no, model_no):
    d = []
    x = data[slice_no]
    # filtering all 0 slices and non-tumor slices
    if (x.any() != 0 and y.any() != 0):
        if (model_no == 0):
            X1 = []
            for i in range(16, 138):
                for j in range(16, 223):
                    if (x[i - 16:i + 17, j - 16:j + 17, :].all != 0):
                        X1.append(x[i - 16:i + 17, j - 16:j + 17, :])
            Y1 = []
            for i in range(16, 138):
                for j in range(16, 223):
                    if (x[i - 16:i + 17, j - 16:j + 17, :].all != 0):
                        Y1.append(y[i, slice_no, j])
            X1 = np.asarray(X1)
            Y1 = np.asarray(Y1)
            d = [X1, Y1]
        elif (model_no == 1):
            d = model_gen(65, x, y, slice_no)
        elif (model_no == 2):
            d = model_gen(56, x, y, slice_no)
        elif (model_no == 3):
            d = model_gen(53, x, y, slice_no)

    return d