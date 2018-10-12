
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
        print('比如:极大似然估计就是经验风险最小化的一个例子.当模型是条件概率分布,',
            '损失函数是对数损失函数时,经验风险最小化就等价于极大似然估计')
        print('但是,当样本容量很小时,经验风险最小化学习的效果就未必很好,如会产生“过拟合”现象')
        print('结构风险最小化(SRM)是为了防止过拟合而提出来的策略.结构风险最小化等价于正则化.',
            '结构风险在经验风险上加上表示模型复杂度的正则化项或者罚项,在假设空间、损失函数以及训练数据集确定的情况下,结构风险的定义为')
        print('   Rsrm(f)=1/N∑L(yi,f(xi))+λJ(f)')
        print('其中J(f)为模型的复杂度,是定义在假空间F上泛函.模型f越复杂,复杂度J(f)就越大;反之,模型越简单,复杂度J(f)就越小.',
            '复杂度表示了对复杂模型的惩罚.λ>=0是系数,用以权衡经验风险和模型复杂度.结构风险小需要经验风险与复杂度同时小.',
            '结构风险小的模型往往对训练数据以及未知的测试数据都有较好的预测')
        print('如:贝叶斯估计中的最大后验概率估计(MAP)就是结构风险最小化的一个例子.当模型是条件概率分布、损失函数是对数损失函数、',
            '模型复杂度由模型的先验概率表示时,结构风险最小化就等价于最大后验概率估计')
        print('结构风险最小化的策略认为结构风险最小的模型是最优的模型.所以求最优模型,就是求解最优化问题：')
        print('   min Rsrm(f)=1/N∑L(yi,f(xi))+λJ(f)')
        print('1.3.3 算法')
        print('算法是指学习模型的具体计算方法.统计学习基于训练数据集,根据学习策略,从假设空间中选择最优模型,最后需要考虑用什么样的计算方法求解最优模型')
        print('这时,统计学习问题归结为最优化问题,统计学习的算法成为求解最优化问题的算法,如果最优化问题有显式的解析解,这个最优化问题就较为简单')
        print('但通常解析解不存在,这就需要数值计算的方法求解.如何保证找到全局最优解,并使求解的过程非常高效,就成为一个重要问题')
        print('统计学习可以利用已有的最优化算法,有时也需要开发独自的最优化算法')
        print('统计学习方法之间的不同,主要来自其模型、策略、算法的不同.确定了模型、策略、算法，统计学习的方法也就确定了.')
        print('1.4 模型评估与模型选择')
        print('1.4.1 训练误差与测试误差')
        print('统计学习的的目的是使学习到的模型不仅对已知的数据而且对未知数据都能有很好的预测能力.')
        print('不同的学习方法会给出不同的模型.当损失函数给定时,基于损失函数的模型的训练误差和模型的测试误差就自然成为学习方法评估的标准')
        print('注意：统计学习方法具体采用的损失函数未必是评估时使用的损失函数.当然,让两者一致是比较理想的')
        print('假设学习到的模型是Y=f‘(X),训练误差是模型Y=f’(X)关于训练数据集的平均损失:')
        print('  Remp(f‘)=1/N∑L(yi,f‘(xi))  其中N是训练样本的容量')
        print('测试误差是模型Y=f‘(X)关于测试数据集的平均损失:etest=1/N’∑L(yi,f‘(xi))  其中N’是测试样本容量')
        print('训练误差的大小,对判断给定的问题不是一个容易学习的问题是有意义的,但本质上不重要')
        print('测试误差反映了学习方法对未知的测试数据集的预测能力,是学习中的重要概念.')
        print('显然,给定两种学习方法,测试误差小的方法具有更好的预测能力,是更有效的方法,通常将学习方法对未知数据的预测能力称为泛化能力')
        print('1.4.2 过拟合与模型选择')
        print('当假设空间含有不同复杂度(例如,不同的参数个数)的模型时,就要面临模型选择的问题')
        print('希望选择或学习一个合适的模型.如果假设空间中存在“真”模型,那么所选择的模型应该逼近真模型.')
        print('具体地,所选择的模型要与真模型的参数个数相同,所选择的模型的参数向量与真模型的参数向量相近')
        print('过拟合现象:如果所选择的模型的复杂度往往比真模型更高.这种现象称为过拟合(over-fitting).学习时所选择的模型所包含的参数过多,',
            '以致于出现这一模型对已知数据预测得很好,但对未知数据预测得很差的现象.可以说模型选择旨在避免过拟合并提高模型的预测能力')
        print('以多项式函数拟合问题为例,说明过拟合与模型选择.这是一个回归问题')
        print('例1.1 假定一个给定的训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)}')
        print('其中,xi∈R是输入x的观测值,yi∈R是相应的输出y的观测值,i=1,2,...,N.多项式函数拟合的任务是假设给定数据由M次多项式函数生成,',
            '选择最有可能产生这些数据的M次多项式函数,即在M次多项式函数中选择一个对已知数据以及未知数据都有很好预测能力的函数')
        print('设M次多项式fM(x,w)=w0+w1x+w2x^2+...+wMx^M=∑wjx^j')
        print('x是单变量输入,w0,w1,...,wM是M+1个参数')
        print('解决这一问题的方法步骤:首先确定模型的复杂度,即确定多项式的次数;然后在给定的模型复杂度下,按照经验风险最小化的策略,求解参数,',
            '即多项式的系数,具体地,求以下经验风险最小化L(w)=0.5∑(f(xi,w)-yi)^2')
        print('取系数0.5是为了计算方便')
        print('将模型与训练数据带入损失函数')
        print('L(w)=0.5∑(∑wjx^j-yi)^2   对wj求偏导并令其为0,可得')
        print('wj=∑xiyi/∑xi  于是求得拟合多项式系数w0*,w1*,w2*,...,wM*')
        print('当选取不同模型时M=0,M=1,M=3及M=9时多项式函数拟合的情况.如果M=0,多项式曲线是一个常数,数据拟合效果很差.',
            '如果M=1,多项式曲线是一个常数,数据拟合效果很差.相反,如果M=9,多项式曲线通过每个数据点,训练误差为0.从对给定训练数据拟合的角度来说,效果是最好的')
        print('但是,因为训练数据本身存在噪声,这种拟合曲线对未知数据的预测能力往往并不是最好的,在实际学习中并不可取.',
            '这时过拟合现象就会发生.这就是说,模型选择时,不仅考虑对已知数据的预测能力,而且还要考虑对未知数据的预测能力')
        print('当M=3时,多项式曲线对训练数据拟合效果足够好,模型也比较简单,是一个较好的选择')
        print('在多项式函数拟合中可以看到,随着多项式次数(模型复杂度)的增加,训练误差会减小,直至趋向于0,',
            '但是测试误差却不如此,会随着多项式次数的增加先减小而后增大')
        print('而最终的目的是使测试误差达到最小.这样,在多项式函数拟合中,就要选择合适的多项式次数,',
            '以达到这一目的.这一结论对一般的模型选择也是成立的')
        print('学习时要防止过拟合,进行最优的模型选择,即选择复杂度适当的模型,以达到使测试误差最小的学习目的:正则化与交叉验证')
        print('1.5 正则化与交叉验证')
        print('1.5.1 正则化')
        print('模型选择的典型方法是正则化.正则化是结构风险最小化策略的实现,是在经验风险上加上一个正则化项或罚项.',
            '正则化项一般是模型复杂度的单调递增函数,模型越复杂,正则化值就越大.比如,正则化项可以是模型参数向量的范数')
        print('正则化一般形式: min 1/N∑L(yi,f(xi)) + λJ(f)')
        print('正则化项可以取不同的形式,回归问题中,损失函数是平方损失,正则化项可以是参数向量的L2范数||w||,也可以是参数向量的L1范数')
        print('正则化符合奥卡姆剃刀原理,奥卡姆剃刀原理应用于模型选择时变为一下想法:在所有可能选择的模型中,',
            '能够很好地解释已知数据并且十分简单才是最好的模型,也就是应该选择的模型.从贝叶斯估计的角度来看,',
            '正则化对应于模型的先验概率.可以假设复杂的模型有较大的先验概率,简单的模型有较小的先验概率.')
        print('L1正则化和L2正则化可以看做是损失函数的惩罚项.对于线性回归模型,使用L1正则化的模型叫做Lasso回归,',
            '使用L2正则化的模型叫做Ridge回归')
        print('L1正则化有助于生成一个稀疏权值矩阵,进而可以用于特征选择.')
        print('稀疏矩阵指的是很多元素为0,只有少数元素是非零值的矩阵,即得到的线性回归模型的大部分系数都是0.',
            '通常机器学习中特征数量很多,例如文本处理时一个词组对应于上万个特征数量')
        print('但是加入正则化项以后,损失函数不完全可微,可以通过梯度下降法求出损失函数的最小值,加入正则化项后相当于加入了约束')
        print('L1正则化会比L2正则化更具有稀疏性')
        print('监督机器学习问题就是在规则化参数的同时最小化误差.最小化误差是为了让我们的模型拟合训练数据,',
            '而规则化参数是防止模型过分拟合训练数据')
        print('1.5.2 交叉验证')
        print('如果给定的样本数据充足,进行模型选择的一种简单方法是随机地将数据集切分成三部分,分别为训练集、验证集、测试集')
        print('训练集用来训练模型、验证集用于模型的选择,而测试集用于最终对学习方法的评估.在学习到不同的复杂度的模型中,',
            '选择对验证集合有最小预测误差的模型,由于验证集有足够多的数据,用它对模型进行选择也是有效的')
        print('但是,许多应用中数据是不足的,为了选择更好的模型,可以采用交叉验证方法.交叉验证的基本想法是重复地使用数据;',
            '把给定的数据进行切分,切分的数据集组合为训练集合与测试集,在此基础上反复地进行训练、测试及模型选择')
        print('1.简单交叉验证')
        print('简单交叉验证方法是:首先随机地将已给数据分为两部分,一部分作为训练集,另一部分作为测试集',
            '(例如,70%的数据为训练集,30%的数据为测试集合):然后用训练集合在各种条件下(例如：不同的参数个数)训练模型,从而得到不同的模型',
            '在测试集上评价各个模型的测试误差,选出测试误差最小的模型')
        print('2.S折交叉验证')
        print('应用最多的是S折交叉验证(S-fold cross validation),首先随机地将已给数据切分为S个互不相交的大小相同的子集;',
            '然后利用S-1个子集的数据训练模型,利用余下的子集测试模型;将这一过程对可能的S种选择重复进行;',
            '最后选出S次评测中平均测试误差最小的模型')
        print('3.留一交叉验证')
        print('S折交叉验证的特殊情形是S=N,称为留一交叉验证,往往在数据缺乏的情况下使用.这里,N是给定数据集的容量')
        print('1.6 泛化能力')
        print('1.6.1 泛化误差')
        print('学习方法的泛化能力是指由该方法学习到的模型对未知数据的预测能力,是学习方法本质上重要的性质,',
            '现实中采用最多的方法是通过测试误差来评价学习方法的泛化能力.但这种评价是依赖于测试数据集的.',
            '因为测试数据集是有限的,很有可能由此得到的评价结果是不可靠的.统计学习理论视图从理论上对学习方法的泛化能力进行分析')
        print('泛化误差的定义.如果学到的模型是f‘,那么用这个模型对未知数据预测的误差即为泛化误差(generalization error)')
        print('  Rexp(f’)=Ep[L(Y,f‘(X)))]=int(L(y,f‘(x))P(x,y)dxdy)')
        print('泛化误差反应了学习方法的泛化能力,如果一种方法学习的模型比另一种方法的模型具有更小的泛化误差,那么这种方法就更有效.',
            '事实上,泛化误差就是所学习到的模型的期望风险.')
        print('1.6.2 泛化误差上界')
        print('学习方法的泛化能力分析往往是通过研究泛化误差的概率上界进行的,简称为泛化误差上界(generalization error bound)')
        print('具体来说,就是通过比较两种学习方法的泛化误差上界的大小来比较它们的优劣.')
        print('泛化误差上界通常具有以下性质:它是样本容量的函数,当样本容量增加时,泛化上界趋于0;它是假设空间容量的函数,',
            '假设空间容量越大,模型就越难学习,泛化误差上界就越大')
        print('一个简单的泛化误差上界的例子')
        print('二类分类问题,已知训练数据集T={(x1,y1),(x2,y2),...,(xn,yn),它是从联合概率分布P(X,Y)独立同分布产生的,X∈R^n,Y∈{-1,1}.',
            '假设空间是函数的有限集合F={f1,f2,...,fd},d是函数个数.设f是从F中选取的函数.',
            '损失函数是0-1损失.关于f的期望风险和经验风险分别是')
        print('   R(f)=E[L(Y,f(X)))]     R‘(f)=1/N∑L(yi,f(xi))')
        print('经验风险最小化函数是fN=argminR‘(f)  更关心的是fN的泛化能力R(fN)=E[L(Y,fN(X))]')
        print('下面讨论从有限集合F={f1,f2,...,fd}中任意选出的函数f的泛化误差上界')
        print('定理1.1 (泛化误差上界) 对二类分类问题,当假设空间是有限个函数的集合F={f1,f2,...,fd},对任意的一个函数f∈F,至少以概率1-d,以下不等式成立:')
        print('  R(f)<=R‘(f)+e(c,N,d)  其中e(c,N,d)=sqrt(1/2N(logc+log(1/d)))')
        print('不等式左端R(f)是泛化误差,右端即为泛化误差上界.在泛化误差上界中,第1项是训练误差,训练误差越小,泛化误差也越小.',
            '第2项e(c,N,d)是N的单调递减函数,当N趋于无穷时趋于0;同时它也是sqrt(logd)阶的函数,假设空间F包含的函数越多,其值越大')
        print('Hoeffding不等式')
        print('设Sn=∑Xi是独立随机变量X1,X2,...,Xn之和,Xi∈[ai,bi],则对任意t>0,一下不等式成立:')
        print('  P(Sn-ESn>=t)<=exp(-2t^2/∑(bi-ai)^2)  ')
        print('  P(ESn-Sn>=t)<=exp(-2t^2/∑(bi-ai)^2)  ')
        print('对任意函数f∈F,R‘(f)是N个独立的随机变量L(Y,f(X))的样本均值,R(f)是随机变量L(Y,f(X))的期望值.如果损失函数取值于区间[0,1],',
            '即对所有i,[ai,bi]=[0,1],那么由Hoeffding不等式一步一步推理可得定理1.1')
        print('就是说,训练误差小的模型,其泛化误差也会小.')
        print('1.7 生成模型与判别模型')
        print('监督学习的任务就是学习一个模型,应用这一模型,对给定的输入预测相应的输出.这个模型的一般形式为决策函数Y=f(X)或者为条件概率分布P(Y|X)')
        print('监督学习方法又可以分为生成方法(generative approach)和判别方法(discriminative approach).',
            '所学到的模型分别为生成模型和判别模型')
        print('生成方法由数据学习联合概率分布P(X,Y),然后求出条件概率分布P(Y|X)作为预测的模型,即生成模型P(Y|X)=P(X,Y)/P(X)')
        print('模型表示了给定输入X产生输出Y的生成关系.典型的生成模型有：朴素贝叶斯法和隐马尔可夫模型')
        print('判别方法由数据直接学习决策函数f(X)或者条件概率分布P(X|Y)作为预测的模型,即判别模型.',
            '判别模型包括：k近邻法、感知机、决策树、Logistic回归模型、最大熵模型、支持向量机、提升方法和条件随机场')
        print('在监督学习中，生成方法和判别方法各有优缺点，适合于不同条件下的学习问题')
        print('生成方法的特点:生成方法可以还原出联合概率分布P(X,Y),而判别方法则不能;生成方法的学习收敛速度更快,',
            '即当样本容量增加的时候,学到的模型可以更快地收敛于真实模型;当存在隐变量时,仍可以用生成方法学习,此时方法就不能用.')
        print('判别方法的特点:判别方法直接学习的是条件概率P(Y|X)或f(X),可以对数据进行各种程度上的抽象、定义特征并实用特征,',
            '因此可以简化学习问题')
        print('1.8 分类问题')
        print('分类是监督学习的一个核心问题.在监督学习中,当输出变量Y取有限个离散值时,预测问题便成为分类问题.这时,输入变量X可以是离散的,',
            '也可以是连续的.监督学习从数据中学习一个分类模型或分类决策函数,称为分类器(classfier).分类器对新的输入进行输出的预测(prediction),',
            '称为分类(classification).可能的输出称为类(class).分类的类别为多个时,称为多类分类问题.')
        print('分类问题包括学习和分类两个过程.在学习过程中,根据已知的训练数据集利用有效的学习方法学习一个分类器;',
            '在分类过程中,利用学习的分类器对新的输入实例进行分类')
        print('学习系统由训练数据学习一个分类器P(Y|X)或Y=f(X);分类系统通过学到的分类器P(Y|X)或Y=f(X)对新的输入实例xN+1进行分类',
            '即预测其输出的类标记yN+1')
        print('评价分类器性能的指标一般是分类准确率(accuracy),其定义是:对于给定的测试数据集,分类器正确分类的样本数与总样本数之比.',
            '也就是损失函数是0-1损失时测试数据集上的准确率')
        print('对于二类分类问题常用的评价指标是精确率(precision)与召回率(recall).通常以关注的类为正类,',
            '其他类为负类,分类器在测试数据集上的预测或正确或不正确,4种情况出现的总数分别记做')
        print('  TP 将正类预测为正类数;')
        print('  FN 将正类预测为负整数')
        print('  FP 将负类预测为正类数')
        print('  TN 将负类预测为负类数')
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