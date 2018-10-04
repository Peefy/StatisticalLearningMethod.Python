
class _Symbol:
    """
    符号表
    """
    def __init__(self):
        pass

    @property
    def symbol_dict(self):
        d = {}
        d['R'] = '实数集'
        d['R^n'] = 'n维实数向量空间,n维欧式空间'
        d['H'] = '希尔伯特空间'
        d['X'] = '输入空间'
        d['Y'] = '输出空间'
        d['x∈X'] = '输入,实例'
        d['y∈Y'] = '输出,标记'
        d['X'] = '输入随机变量'
        d['Y'] = '输出随机变量'
        d['T={(x1,y1),(x2,y2),...,(xn,yn)}'] = '训练输出集'
        d['N'] = '样本容量'
        d['(xi,yi)'] = '第i个训练数据点'
        d['x=(x(1),x(2),...,x(n))'] = '输入向量,n维实数向量'
        d['xi(j)'] = '输入向量xi的第j分量'
        d['P(X),P(Y)'] = '概率分布'
        d['P(X,Y)'] = '联合概率分布'
        d['F'] = '假设空间'
        d['f∈F'] = '模型参数'
        d['θ, w'] = '权值向量'
        d['w=(w1,w2,...,wn)^T'] = '偏置'
        d['J(f)'] = '模型的复杂度'
        d['R_emp'] = '经验风险或者经验损失'
        d['R_exp'] = '风险函数或期望损失'
        d['L'] = '损失函数，拉格朗日函数'
        d['η'] = '学习率'
        d['||·||1'] = 'L1范数'
        d['||·||2,||·||'] = 'L2范数'
        d['(x·x\')'] = '向量x和x\'的内积'
        d['H(X),H(p)'] = '熵'
        d['H(Y|X)'] = '条件熵'
        d['S'] = '分离超平面'
        d['α=(α1,α2,...,αn)^T'] = '拉格朗日乘子,对偶问题变量'
        d['αi'] = '对偶问题的第i个变量'
        d['K(x,z)'] = '核函数'
        d['sign(x)'] = '符号函数'
        d['I(x)'] = '指示函数'
        d['Z(x)'] = '规范化因子'
        return d

class Chapter1:
    def __init__(self):
        pass

    def note(self):
        print('第1章 ')

_sym = _Symbol()
chapter1 = Chapter1()

def main():
    print(_sym.symbol_dict)
    chapter1.note()

if __name__ == '__main__':
    main()