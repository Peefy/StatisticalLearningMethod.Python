
class Chapter6:
    """
    第6章 逻辑斯谛回归与最大熵模型
    """
    def __init__(self):
        """
        第6章 逻辑斯谛回归与最大熵模型
        """
        pass

    def note(self):
        """
        chapter6 note
        """
        print('第6章 逻辑斯谛回归与最大熵模型')
        print('逻辑斯谛回归是统计学习中的经典分类方法.最大熵概率模型学习的一个准则,',
            '将其推广到分类问题得到最大熵模型.逻辑斯谛回归模型与最大熵模型都属于对数线性模型.')
        print('6.1 逻辑斯谛回归模型')
        print('6.1.1 逻辑斯谛分布')
        print('定义6.1（逻辑斯谛分布）设X是连续随机变量,X服从逻辑斯谛分布是指X具有下列分布函数和密度函数')
        print('  F(x)=P(X<=x)=1/(1+e^(-(x-u)/y))')
        print('  f(x)=F\'(x)=e^(-(x-u)/y)/(y(1+e^(-(x-u)/y)))')
        print('式中,u为位置参数,y>0为形状参数')
        print('逻辑斯谛分布的密度函数f(x)和分布函数F(x)的图形如图6.1所示.分布函数属于逻辑斯谛函数,',
            '其图形是一条S形曲线(sigmoid curve).该曲线以点(u,0.5)为中心对称,即满足:')
        print('  F(-x+u)-0.5=-F(x-u)+0.5')
        print('曲线在中心附近增长速度较快,在两端增长速度较慢.形状参数y的值越小,曲线在中心附近增长得越快.')
        print('6.1.2 二项逻辑斯谛回归模型')
        print('二项逻辑斯谛回归模型是一种分类模型,由条件概率分布P(Y|X)表示,形式为参数化的逻辑斯谛分布.',
            '这里,随机变量X取值为实数,随机变量Y取值为1或0.通过监督学习的方法来估计模型参数.')
        print('定义6.2(逻辑斯谛回归模型)二项逻辑斯谛回归模型是如下的条件概率分布:')
        print('   P(Y=1|x)=exp(wx+b)/(1+exp(wx+b))')
        print('   P(Y=0|x)=1/(1+exp(wx+b))')
        print('这里,x∈R^n,Y∈{0,1},w∈R^n和b∈R是参数,w称权值向量,b称为偏置,w·x为w和x的内积')
        print('现在考察逻辑斯谛回归模型的特点.一个事件的几率(odds)是指该事件发生的概率与该事件不发生的概率的比值.',
            '如果事件发生的概率的比值.如果事件发生的概率是p,那么该事件的几率是p/(1-p),',
            '该事件的对数几率(log odds)或logit函数是:logit(p)=logp/(1-p)')
        print('对逻辑斯谛回归而言:logP(Y=1|x)/(1-P(Y=1|x))=w·x')
        print('在逻辑斯谛回归模型中,输出Y=1的对数几率输入x的线性函数.或者说,',
            '输出Y=1的对数几率是由输入x的线性函数表示的模型,即逻辑斯谛回归模型')
        print('换一个角度看,考虑对输入x进行分类的线性函数w·x,其值域为实数域.',
            '注意：这里x∈R^(n+1),w∈R^(n+1).通过逻辑斯谛回归模型定义可以将线性函数w·x转换为概率：',
            'P(Y=1|x)=exp(wx)/(1+exp(wx))')
        print('这时,线性函数的值越接近正无穷,概率值就越接近于1;线性函数的值越接近负无穷,',
            '概率值就越接近0.这样的模型就是逻辑斯谛回归模型')
        print('6.1.3 模型参数估计')
        print('逻辑斯谛回归模型学习时,对于给定的训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},',
            '其中,xi∈R^n,yi∈{0,1},可以应用极大似然估计法估计模型参数,从而得到逻辑斯谛回归模型.')
        print('设:P(Y=1|x)=pi(x),P(Y=1|x)=1-pi(x)')
        print('对数似然函数为：L(w)=∑[yi(w·xi)-log(1+exp(w·xi))]')
        print('对L(w)求极大值,得到w的估计值')
        print('问题就变成了以对数似然函数为目标函数的最优化问题.逻辑斯谛回归学习中通常采用的方法是梯度下降法及拟牛顿法')
        print('假设w的极大似然估计值是w,那么学到的逻辑斯谛回归模型为:')
        print('  P(Y=1|x)=exp(w·x)/(1+exp(w·x))')
        print('  P(Y=0|x)=1/(1+exp(w·x))')
        print('6.1.4 多项式逻辑斯谛回归')
        print('之前介绍的逻辑斯谛回归模型是二项分类模型,用于二类分类.可以将推广为多项逻辑斯谛回归模型,',
            '用于多类分类.假设离散型随机变量Y的取值集合是{1,2,...,K},那么多项式逻辑斯谛回归模型是',
            'P(Y=k|x)=exp(wk·x)/(1+∑exp(wk·x)),k=1,2,...,K-1.',
            'P(Y=K|x)=1/(1+∑exp(wk·x));   这里x∈R^(n+1),wk∈R^(n+1)')
        print('二项逻辑斯谛回归的参数估计法也可以推广到多项逻辑斯谛回归.')
        print('6.2 最大熵模型')
        print('最大熵模型(maximum entropy model)由最大熵原理推导实现.')
        print('6.2.1 最大熵原理')
        print('最大熵原理是概率模型学习的一个准则.最大熵原理认为,学习概率模型时,',
            '在所有可能的概率模型(分布)中,熵最大的模型是最好的模型.通常用约束条件来确定概率模型的集合',
            '所以,最大熵原理也可以表述为在满足约束条件的模型集合中选取熵最大的模型')
        print('假设离散随机变量X的概率分布式P(X),则其熵是H(P)=-∑P(x)logP(X),',
            '熵满足下列不等式:0<=H(P)<=log|X|')
        print('式中,|X|是X的取值个数,当且仅当X的分布是均匀分布时右边的等号成立,这就是说,当X服从均匀分布时,熵最大.')
        print('最大熵原理认为要选择的概率模型首先必须满足已有的事实,即约束条件.在没有更多信息的情况下,',
            '那些不确定的部分都是“等可能的”.最大熵原理通过熵的最大化表示等可能性.',
            '“等可能性”不容易操作,而熵则是一个可优化的数值指标.')
        print('例6.1 假设随机变量X有5个取值{A,B,C,D,E},要估计各个值的概率P(A),P(B),P(C),P(D),P(E).')
        print('解：这些值满足以下约束条件:P(A)+P(B)+P(C)+P(D)+P(E)=1')
        print('满足这个约束条件的概率分布有无穷多个.如果没有任何其他信息,仍要对概率分布进行估计,',
            '一个办法就是认为这个分布中取各个值的概率是相等的:',
            'P(A)=P(B)=P(C)=P(D)=P(E)=0.2')
        print('等概率表示了对事实的无知.因为没有更多的信息,这种判断是合理的.')
        print('有时,能从一些先验知识中得到一些对概率值的约束条件,例如:')
        print('P(A)+P(B)=3/10; P(A)+P(B)+P(C)+P(D)+P(E)=1')
        print('但是满足这两个约束条件的概率分布仍然有无穷多个.在缺少其他信息的情况下,',
            '可以认为A与B是等概率的,C,D与E是等概率的,于是:',
            'P(A)=P(B)=3/20;  P(C)=P(D)=P(E)=7/30;')
        print('如果还有第3个约束条件:P(A)+P(C)=0.5; P(A)+P(B)=3/10; ',
            'P(A)+P(B)+P(C)+P(D)+P(E)=1')
        print('可以按照满足约束条件下求等概率的方法估计概率分布.这里不再继续讨论.',
            '以上概率模型学习的方法正是遵循了最大熵原理.')
        print('提供了用最大熵原理进行概率模型选择的几何解释.概率模型集合P可由欧式空间中的单纯形表示,',
            '如左图的三角形(2-单纯形).一个点代表一个模型,整个单纯形代表模型集合.',
            '右图上的一条直线对应于一个约束条件,直线的交集对应于满足所有约束条件的模型集合.',
            '一般地,这样的模型仍然有无穷多个.学习的目的是在可能的模型集合中选择最优模型,',
            '而最大熵原理则给出最优模型选择的一个准则')
        print('6.2.2 最大熵模型的定义')
        print('最大熵原理是统计学习的一般过程,将它应用到分类得到最大熵模型.')
        print('假设分类模型是一个条件概率分布P(Y|X),X∈x∈R^n表示输入,Y∈y表示输出,',
            'X和Y分别是输入和输出的集合.这个模型表示的是对于给定的输入X,以条件概率P(Y|X)输出Y')
        print('给定一个训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)}')
        print('学习的目标是用最大熵原理选择最好的分类模型')
        print('首先考虑模型应该满足的条件.给定训练数据集,可以确定联合分布P(X,Y)的经验分布和边缘分布P(X)的经验分布,',
            '分别以P(X,Y)和P(X)表示.这里,P(X=x,Y=y)=v(X=x,Y=y)/N; P(X=x)=v(X=x)/N')
        print('其中,v(X=x,Y=y)表示训练数据中样本(x,y)出现的频数,v(X=x)表示训练数据中输入x出现的频数,',
            'N表示训练样本容量')
        print('用特征函数(feature function) f(x,y)描述输入x和输出y之间的某一个事实.其定义是:',
            'f(x,y)=1, x与y满足某一事实; f(x,y)=0, otherwise.')
        print('它是一个二值函数,当x和y满足这个事实时取值为1,否则取值为0.')
        print('特征函数f(x,y)关于经验分布P(X,Y)的期望值,用Ep(f)表示：Ep(f)=∑P(x,y)f(x,y)')
        print('特征函数f(x,y)关于模型P(Y|X)与经验分布P(X)的期望值,用Ep(f)表示.')
        print(' Ep(f)=∑P(x)P(y|x)f(x,y)=∑P(x,y)f(x,y)')
        print('作为模型学习的约束条件. 假如有n个特征函数fi(x,y),i=1,2,...,n,那么就有n个约束条件')
        print('定义在条件概率分布P(Y|X)上的条件熵为：H(P)=-∑P(x)P(y|x)logP(y|x)')
        print('则模型集合C中条件熵H(P)最大的模型称为最大熵模型.式中的对数为自然对数')
        print('6.2.3 最大熵模型的学习')
        print('最大熵模型的学习过程就是求解最大熵模型的过程.最大熵模型的学习可以形式化约束最优化问题')
        print('对于给定的训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)}以及特征函数fi(x,y),i=1,2,...,n',
            '最大熵模型的学习等价约束最优化问题:',
            'maxH(P)=-∑P(x)P(y|x)logP(y|x) s.t. Ep(fi)=Ep(fi),i=1,2,...,n; ∑P(y|x)=1')
        print('按照最优化的习惯,将求最大值问题改写为等价的求最小值问题：',
            'min-H(P)=∑P(x)P(y|x)logP(y|x) s.t. Ep(fi)-Ep(fi)=0, i=1,2,...,n ∑P(y|x)=1')
        print('求解约束最优化问题,所得出的解,就是最大熵模型学习的解.')
        print('具体推导:将约束最优化问题转换为无约束最优化的对偶问题.通过切结对偶问题求解原始问题.',
            '首先,引进拉格朗日乘子w0,w1,w2,...,wn,定义拉格朗日函数L(P,w)')
        print('Zw(x)=∑exp(∑wifi(x,y))')
        print('Zw(x)称为规范化因子;fi(x,y)是特征函数;wi是特征的权值.',
            'Pw=Pw(y|x)就是最大熵模型.这里,w是最大熵模型中的参数向量.')
        print('例6.2 学习例6.1中的最大熵模型.')
        print('解: 为了方便,分别以y1,y2,y3,y4,y5表示A,B,C,D,E,于是最大熵模型学习的最优化问题是：')
        print('min- H(P)=∑P(yi)logP(yi) s.t. P(y1)+P(y2)=P(y1)+P(y2)=3/10')
        print('引进拉格朗日乘子w0,w1,定义拉格朗日函数')
        print('L(P,w)=∑P(yi)logP(yi)+w1(P(y1)+P(y2)-3/10)+w0(∑P(yi)-1)')
        print('根据拉格朗日对偶性,可以通过求解对偶优化问题得到原始最优化问题的解,所以求解:')
        print('  maxminL(P,w)')
        print('首先求解L(P,w)关于P的极小化问题.为此,固定w0,w1,求偏导数：')
        print('  dL(P,w)/dP(y1)=1+logP(y1)+w1+w0')
        print('  dL(P,w)/dP(y2)=1+logP(y2)+w1+w0')
        print('  dL(P,w)/dP(y3)=1+logP(y3)+w0')
        print('  dL(P,w)/dP(y4)=1+logP(y4)+w0')
        print('  dL(P,w)/dP(y5)=1+logP(y5)+w0')
        print('令各偏导数等于0,解得:P(y1)=P(y2)=e^(-w1-w0-1); P(y3)=P(y4)=P(y5)=e^(-w0-1)')
        print('于是,minL(P,w)=L(Pw,w)=-2e^(-w1-w0-1)-3e^(-w0-1)-3/10w1-w0')
        print('再求解L(Pw,w)关于w的极大化问题: maxL(Pw,w)=-2e^(-w1-w0-1)-3e^(-w0-1)-3/10w1-w0')
        print('分别求L(Pw,w)对w0,w1的偏导数并令其为0,得到:e^(-w1-w0-1)=3/20; e^(-w0-1)=7/30')
        print('于是得到所要求的概率分布为P(y1)=P(y2)=3/20  P(y3)=P(y4)=P(y5)=7/30')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')

def main():
    chapter6.note()

if __name__ == '__main__':
    main()