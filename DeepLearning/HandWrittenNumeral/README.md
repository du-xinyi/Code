输入层：数据输入



卷积层：进行特征提取和特征映射

torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')

- in_channels - 输入的通道数量

- out_channels - 卷积产生的通道数量

- kernel_size - 卷积核的大小

- stride - 卷积步长，默认为1

- padding - 输入两边填充0的个数，默认为0

- padding_mode - 'zeros'

- dilation - 卷积点空隙，默认为1

- groups - 从输入通道到输出通道阻塞连接的数量，默认为1

- bias - 如果设置为True，会在输出上增加可学习的偏置。默认为True

$$
L\_{in} = \frac{L\_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} + 1
$$



池化层：进行下采样，对特征图稀疏处理，去除噪声

torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)

- kernel_size - 最大化操作的窗口尺寸
- stride - 窗口的步长。默认stride=kernel_size
- padding - 在两边隐形施加零填充的数量
- dilation - 控制窗口元素间隔的参数
- return_indices - 如果设置为True，将返回最大索引和输出
- ceil_mode - 当该参数为True时，使用ceil(向上取整)而不是floor(向下取整)来计算输出尺寸

$$
L\_{out} = \frac{L\_{in} + 2 \times padding - dilation \times (kernel\_size - 1) - 1}{stride} + 1
$$



全连接层：torch.nn.Linear(in_features, out_features, bias=True)
in_features 表示输入的二维张量的大小，out_features 表示输出的二维张量的大小.





输出层：输出结果
