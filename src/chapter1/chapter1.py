
class _Symbol:
    """
    统计方法 符号表
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
    """
    第1章 统计学习方法概论
    """
    def __init__(self):
        """
        第1章 统计学习方法概论
        """
        pass

    def note(self):
        """
        chapter1 note
        """
        print('第1章 统计学习方法概论')
        # !统计学习方法三要素:模型,策略和算法
        print('统计学习方法三要素:模型,策略和算法')
        print('模型选择,包括正则化,交叉验证,学习的泛化能力;')
        print('生成模型和判别模型')
        print('监督学习方法的应用:分类问题,标注问题,回归问题')
        print('1.1 统计学习')
        print('1.统计学习的特点')
        print('统计学习是关于计算机基于数据构建概率统计模型并运用模型对数据进行预测与分析的一门学科,统计学习也称为统计机器学习')
        print('统计学习的特点:')
        print(' (1) 统计学习以计算机及网络为平台,是建立在计算机及网络之上的')
        print(' (2) 统计学习以数据为研究对象,是数据驱动的学科')
        print(' (3) 统计学习的目的是对数据进行预测与分析')
        print(' (4) 统计学习以方法为中心,统计学习方法构建模型并应用模型进行预测与分析')
        print(' (5) 统计学习是概率论、统计学、信息论、计算理论、最优化理论及计算机科学等多个领域的交叉学科')
        print('“学习”的定义：如果一个系统能够通过执行某个过程改进它的性能,这就是学习')
        # !统计学习就是计算机系统同构运用数据及统计方法提高系统性能的机器学习
        print('统计学习就是计算机系统同构运用数据及统计方法提高系统性能的机器学习')
        print('2.统计学习的对象')
        # !统计学习的对象是数据,它从数据出发,提取数据的特征,抽象出数据的模型,发现数据中的知识,又回到对数据的分析与预测中去.
        print('统计学习的对象是数据,它从数据出发,提取数据的特征,抽象出数据的模型,发现数据中的知识,又回到对数据的分析与预测中去.',
            '作为统计学习的对象,数据是多样的.(数字,文字,图像,视频,音频数据以及它们的组合)')
        # !统计学习的前提:关于数据的基本假设:同类数据具有一定的统计规律性
        print('统计学习的前提:关于数据的基本假设:同类数据具有一定的统计规律性')
        print('统计学习过程以变量或者变量组表述数据.数据分为由联系变量和离散变量表示的类型')
        print('3.统计学习的目的')
        print('统计学习用于对数据进行预测与分析,特别是对未知新数据进行预测和分析')
        print('对数据的预测与分析是通过构建概率统计模型实现的.统计学习的目标就是考虑学习什么样的模型和如何学习模型,',
            '以使模型能对数据进行准确的预测与分析,同时也要考虑尽可能地提高学习效率')
        print('统计学习的方法')
        print('统计学习的方法是基于数据构建统计模型从而对数据进行预测与分析.统计学习由监督学习(supervised learning),',
            '非监督学习(unsupervised learding),半监督学习(semi-supervised-learning)和强化学习(reinforcement learning)等组成')
        print('监督学习情况下的统计学习方法概括:从给定的、有限的、用于学习的训练数据(training data)集合出发,',
            '假设数据是独立分布产生的;并且假设要学习的模型属于某个函数的集合,称为假设空间,应用某个评价准则,从假设空间中选取一个最优的模型,',
            '使它对已知训练数据及未知测试数据(test data)在给定的评价准则下有最优的预测:最优模型的选取由算法实现.',
            '这样,统计学习方法包括模型的假设空间、模型选择的准则以及模型学习的算法,称为统计学习的三要素,简称为模型(model)、策略(strategy)和算法(algorithm)')
        print('实现统计学习方法的步骤：')
        print('  (1) 得到一个有限的训练数据集合;')
        print('  (2) 确定包含所有可能的模型的假设空间,即学习模型的集合;')
        print('  (3) 确定模型选择的准则,即学习的策略')
        print('  (4) 实现求解最优模型的算法,即学习的算法')
        print('  (5) 通过学习方法选择最优模型')
        print('  (6) 利用学习的最优模型对新数据进行预测或分析')
        print('5.统计学习的研究')
        print('  统计学习方法、理论、应用')
        print('6.统计学习的重要性')
        print('  人工智能,模式识别,数据挖掘,自然语言处理,语音识别,图像识别,信息检索,生物信息,')
        print('  (1) 统计学习是处理海量数据的有效方法')
        print('  (2) 统计学习是计算机智能化的有效手段')
        print('  (3) 统计学习是计算机科学发展的一个重要组成部分')
        print('1.2 监督学习')
        print('监督学习的任务是学习一个模型,使模型能够对任意给定的输入,对其相应的输出做出一个好的预测',
            '(这里的输入、输出是指某个系统的输入与输出,与学习的输入与输出不同)')
        print('1.2.1 基本概念')
        print('1.输入空间、特征空间、输出空间')
        print('在监督学习中,将输入与输出所有可能取值的集合分别称为输入空间与输出空间.',
            '输入空间与输出空间可以是有限元素的集合,也可以是整个欧式空间',
            '输入空间与输出空间可以是同一个空间,也可以是不同的空间,通常输出空间远远小于输入空间')
        print('将输入与输出看作是定义在输入(特征)空间与输出空间上的随机变量的取值.')
        print('监督学习从训练数据集合中学习模型,对测试数据进行预测.训练数据由输入(或特征向量)与输出组成',
            '测试数据也由相应的输入与输出对组成.输入与输出对又称为样本或样本点')
        print('输入变量X和输出变量Y有不同的类型,可以是连续的,也可以是离散的.')
        print('根据输入、输出变量的不同类型,对预测任务给与不同的名称：')
        print('  输入变量与输出变量均为连续变量的预测问题称为回归问题;')
        print('  输出变量为有限个离散变量的预测问题称为分类问题;')
        print('  输入变量与输出变量均为变量序列的预测问题称为标注问题')
        print('2.联合概率分布')
        print('监督学习假设输入与输出的随机变量X和Y遵循联合概率分布P(X,Y).P(X,Y)表示分布函数,或分布密度函数')
        print('学习过程中假定这一联合概率分布存在,但是对于学习系统来说,联合概率分布的具体定义是未知的')
        print('训练数据与测试数据被看作是依联合概率缝补P(X,Y)独立同分布产生的.')
        print('统计学习假设数据存在一定的统计规律,X和Y具有联合概率分布的假设就是监督学习关于数据的基本假设')
        print('3.假设空间')
        print('监督学习的目的在于学习一个由输入到输出的映射,这一映射由模型来表示.')
        print('学习的目的就在于找到最好的这样的模型.模型属于由输入空间到输出空间的映射的集合,',
            '这个集合就是假设空间.假设空间的确定意味着学习范围的确定')
        print('监督学习的模型可以是概率模型也可以是非概率模型,由条件概率分布P(X|Y)或决策函数Y=f(X)表示,随具体学习方法而定.',
            '对具体的输入进行相应的输出预测时,写作P(y|x)或y=f(x)')
        print('1.2.2 问题的形式化')
        print('监督学习利用训练数据集学习一个模型,再用模型对测试样本集进行预测,由于训练数据集合往往是人工给出的,所以称为监督学习.',
            '监督学习分为学习和预测两个过程,由学习系统与预测系统完成.')
        print('监督学习中,假设训练数据与测试数据是依联合概率分布P(X,Y)独立同分布产生')
        print('在学习过程中,学习系统利用给定的训练数据集,通过学习(或训练)得到一个模型,表示为条件概率分布P\'(Y|X)或者决策函数Y=f\'(X)',
            '条件概率分布或者决策函数描述输入与输出随机变量之间的映射关系')
        print('在预测过程中,预测系统对于给定的测试样本集中的输入xN+1,由模型yN+1=argmaxP\'(yN+1|xN+1)给出相应的输出yn+1')
        print('在学习过程中,学习系统(也就是学习算法)试图通过训练数据集合中的样本(xi,yi)带来的信息学习模型.',
            '对输入xi,一个具体的模型y=f(x)可以产生一个输出f(xi),而训练数据集中对应的输出是yi,',
            '如果这个模型有很好的预测能力,训练样本输出yi和模型输出f(xi)之间的差就应该足够小.',
            '学习系统通过不断的尝试,选取最好的模型,以便对训练数据集有足够好的预测,同时对未知的测试数据集合也有尽可能好的推广')
        print('1.3 统计学习三要素')
        print('方法=模型+策略+算法')
        print('监督学习,非监督学习,强化学习也同样拥有三要素')
        print('1.3.1 模型')
        print('统计学习首先考虑的问题是学习什么样的模型.')
        print('在监督学习过程中,模型就是所要学习的条件概率分布或决策函数.模型的假设空间(hypothesis space)包含所有可能条件概率分布或决策函数',
            '例如,假设决策函数是输入变量的线性函数,那么模型的假设空间就是所有这些线性函数构成的函数集合.假设空间中的模型一般有无穷多个.')
        print('假设空间用F表示.假设空间可以定义为决策函数的集合F={f|Y=f(X)}')
        print('其中X和Y是定义在输入空间X和输出空间Y上的变量.这时F通常是由一个参数向量决定的函数族:F={f|Y=fθ(X),θ∈R^n}')
        print('参数向量θ取决于n维欧式空间R^n,称为参数空间(parameter space)')
        print('假设空间也可以定义为条件概率的集合F={P|P(Y|X)}')
        print('其中,X和Y是定义在输入空间X和输出空间Y上的随机变量.这时F通常是由一个参数向量决定的条件概率分布族:')
        print('F={P|Pθ(Y|X),θ∈R^n)}')
        print('参数向量θ取决于n维欧式空间R^n,也称为参数空间')
        print('1.3.2 策略')
        print('统计学习的目标在于从假设空间中选取最优模型')
        print('1.损失函数和风险函数')
        # !输出的预测值f(X)与真实值Y可能一致也可能一致,用一个损失函数(loss function)或者代价函数(cost function)来度量预测错误的程度
        print('监督学习问题是在假设空间F中选取模型f作为决策函数,对于给定的输入X,由f(X)给出相应的输出Y.',
            '这个输出的预测值f(X)与真实值Y可能一致也可能一致,用一个损失函数(loss function)或者代价函数(cost function)来度量预测错误的程度',
            '损失函数是f(X)和Y的非负实数值函数,记做L(Y,f(X))')
        print('统计学习常用的损失函数')
        print('  (1) 0-1损失函数')
        print('    L(Y,f(X))=1,Y!=f(X); L(Y,f(X))=0,Y==f(X);')
        print('  (2) 平方损失函数')
        print('    L(Y,f(X))=(Y-f(X))^2')
        print('  (3) 绝对损失函数')
        print('    L(Y,f(X))=|Y-f(X)|')
        print('  (4) 对数损失函数或者对数似然损失函数')
        print('    L(Y,P(Y|X))=-logP(Y|X)')
        print('损失函数数值越小,模型就越好,由于模型的输入、输出(X,Y)是随机变量,遵循联合分布P(X,Y),所以损失函数的期望是')
        print('    Rexp(f)=Ep[L(Y,f(X))]=int(L(y,f(x))P(x,y))')
        print('这是理论上模型f(X)关于联合分布P(X,Y)的平均意义下的损失,称为风险函数或期望损失')
        print('学习的目标就是选择期望风险最小的模型,由于联合分布P(X,Y)是未知的,',
            'Rexp(f)不能直接计算.实际上,如果知道联合分布P(X,Y),可以直接从联合分布直接求出条件概率分布P(Y|x),也就不需要学习了.',
            '正因为不知道联合概率分布,所以才需要进行学习')
        print('给定一个训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},模型f(X)关于训练数据集的平均损失称为经验风险或经验损失,记做Remp:')
        print('   Remp(f)=1/N∑L(yi,f(xi))')
        print('期望风险Rexp(f)是模型关于联合分布的期望损失,经验风险Remp(f)是模型关于训练样本集的平均损失.')
        print('根据大叔定律,当样本容量N趋于无穷时,经验风险Remp(f)趋于期望风险Rexp(f).')
        print('所以一个很自然的想法是用经验风险估计期望风险.但是,由于现实中训练样本数目有限,甚至很小',
            '所以用经验风险估计期望风险常常不理想,要对经验风险进行一定的矫正.这就关系到监督学习的两个基本策略:',
            '经验风险最小化和结构风险最小化')
        print('2.经验风险最小化与结构风险最小化')
        print('在假设空间、损失函数以及训练数据集确定的情况下,经验风险函数就可以确定')
        print('经验风险最小化(ERM)的策略认为,经验风险最小的模型是最优的模型,根据这一策略,ERM就是求解最优化问题:')
        print('     min 1/N∑L(yi,f(xi)).   其中F是假设空间')
        print('当样本容量足够大时,经验风险最小化能保证有很好的学习效果,在现实中被广泛使用')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')

_symbols = _Symbol().symbol_dict
chapter1 = Chapter1()

def main():
    print(_symbols)
    chapter1.note()

if __name__ == '__main__':
    main()