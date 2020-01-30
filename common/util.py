import numpy as np

def im2col(input_data, out_h, out_w, FH, FW, stride, pad):
    N, C, H, W = input_data.shape

    # input_dataに対して0を埋め込む
    # 第二引数を配列にすることで各ランクの幅量を変える
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, FH, FW, out_h, out_w))

    # 入力データに対してファイルターを適用する場所の領域を横方向に一列に展開する
    # フィルターを適用する全ての場所で行う
    for y in range(FH):
        y_max = y + stride * out_h
        for x in range(FW):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3)
    col = col.reshape(N * out_h * out_w, -1)
    return col

def col2im(col, input_shape, FH, FW, stride, pad):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - FH)//stride + 1
    out_w = (W + 2*pad - FW)//stride + 1
    col = col.reshape(N, out_h, out_w, C, FH, FW)
    col = col.transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2 * pad + stride - 1, W + 2 * pad + stride - 1))
    for y in range(FH):
        y_max = y + stride*out_h
        for x in range(FW):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]