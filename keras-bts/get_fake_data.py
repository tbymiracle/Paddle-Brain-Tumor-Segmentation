import numpy as np


def gen_fake_data():
    fake_data1 = np.random.rand(1, 65, 65, 4).astype(np.float32) - 0.5
    fake_data2 = np.random.rand(1, 33, 33, 4).astype(np.float32) - 0.5
    fake_label = np.ones([1, 1, 5])
    np.save("fake_data.npy", fake_data1)
    np.save("fake_data2.npy", fake_data2)
    np.save("fake_label.npy", fake_label)


if __name__ == "__main__":
    gen_fake_data()
