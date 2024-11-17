我自己的代码是MSNET
具体的运行路径需要再修改options
消融实验
去掉通道和空间是MSNET1
去掉位置是
MSNET2
都去掉是cat
我自己的训练代码是MSNet_train_me.py
原先的是MSNet_train.py
conda activate dformer
python MSNet_train_me.py

00没有PA
11是全有
111（NJU2K）
2没有HF
3没有DFEM和RFEM
4没有afem
5修改CCD（TPAMI）
55(NJU2K)
555(单个TPAMI解码器)
6(改为SwinNet的解码器)
7(6的基础上去掉afem)
77(NJU2K+论文损失函数)
8(损失函数由BCE改为论文提出)
888(8的基础上NJU2K)
9（888的基础上改为SwinNet解码器）