from reprod_log import ReprodDiffHelper

if __name__ == "__main__":
    diff_helper = ReprodDiffHelper()
    keras_info = diff_helper.load_info("forward_keras.npy")
    paddle_info = diff_helper.load_info("../paddle-bts/forward_paddle.npy")

    diff_helper.compare_info(keras_info, paddle_info)
    diff_helper.report(path="forward_diff.log")