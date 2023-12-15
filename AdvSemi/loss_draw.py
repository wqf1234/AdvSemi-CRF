import numpy as np
import matplotlib.pyplot as plt;


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


if __name__ == "__main__":
    train_loss_seg_path = "./train_loss_seg.txt"  # 存储文件路径
    train_loss_D_path = "./train_loss_D.txt"  # 存储文件路径

    y1_train_loss = data_read(train_loss_seg_path)  # loss值，即y轴
    x1_train_loss = range(len(y1_train_loss))  # loss的数量，即x轴

    y2_train_loss = data_read(train_loss_D_path)  # loss值，即y轴
    x2_train_loss = range(len(y2_train_loss))  # loss的数量，即x轴


    plt.figure(figsize=(15, 5))

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')  # x轴标签
    plt.ylabel('loss')  # y轴标签

    # 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x1_train_loss, y1_train_loss, linewidth=1, linestyle="solid", label="train_loss_seg", color='blue')
    plt.plot(x2_train_loss, y2_train_loss, linewidth=1, linestyle="solid", label="train_loss_D", color='red')
    plt.legend()
    plt.title('Loss curve')
    # plt.show()
    plt.savefig('loss.png')
