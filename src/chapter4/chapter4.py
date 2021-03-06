
class Chapter4:
    """
    第4章 朴素贝叶斯法
    """
    def __init__(self):
        """
        第4章 朴素贝叶斯法
        """
        pass

    def note(self):
        """
        chapter4 note
        """
        print('第4章 朴素贝叶斯法')
        print('朴素贝叶斯法是基于贝叶斯定理与特征条件独立假设的分类方法.对于给定的训练数据集,首先基于特征条件独立假设学习输入/输出的联合概率分布,',
            '然后基于此模型,对给定的输入x,利用贝叶斯定理求出后验概率最大的输出y,朴素贝叶斯法实现简单,学习与预测的效率都很高,是一种常用的方法')
        print('4.1 朴素贝叶斯法的学习与分类')
        print('4.1.1 基本方法')
        print('设输入空间X∈R为n维向量的集合,输出空间为类标记集合Y={c1,c2,...,cK}.输入为特征向量x∈X,输出为类标记(class label)y∈Y',
            'X是定义在输入空间X上的随机向量,Y是定义在输出空间Y上的随机变量.',
            'P(X,Y)是X和Y的联合概率分布,训练数据集:T={(x1,y1),(x2,y2),...,(xn,yn)}由P(X,Y)独立同分布产生.')
        print('朴素贝叶斯法通过训练数据集学习联合概率分布P(X,Y).具体地,学习下先验概率分布及条件概率分布.',
            '先验概率分布P(Y=ck),k=1,2,...,K')
        print('条件概率分布:P(X=x,Y=ck)=P(X(1)=x(1),...,X(n)=x(n)|Y=ck),k=1,2,...,K,于是学习到联合概率分布P(X,Y)')
        print('条件概率分布P(X=x|Y=ck)有指数级数量的参数,其估计实际是不可行的.事实上,假设x(j)可取值有Sj个,j=1,2,...,n,',
            'Y可取值有K个,那么参数个数为K∏Sj')
        print('注意:朴素贝叶斯法与贝叶斯估计是不同的概念.')
        print('朴素贝叶斯法对条件概率分布作了条件独立性的假设.由于这是一个较强的假设,朴素贝叶斯法也由此得名.',
            '具体地,条件独立性假设是:P(X=x|Y=ck)=P(X(1)=x(1),...,X(n)=x(n)|Y=ck)=∏P(X(j)=x(j)|Y=ck)')
        print('朴素贝叶斯法实际上学习到生成数据的机制,所以属于生成模型.条件独立假设等于是说用于分类的特征类确定的条件下都是条件独立的',
            '这一假设使朴素贝叶斯法变得简单,但有时会牺牲一定的分类准确率')
        print('朴素贝叶斯法分类时,对给定的输入x,通过学习到的模型计算后验概率分布P(Y=ck|X=x),',
            '将后验概率最大的类作为x的类输出.后验概率计算根据贝叶斯定理进行:')
        print('P(Y=ck|X=x)=P(X=x|Y=ck)P(y=CK)/∑P(X=x|Y=ck)P(Y=ck)')
        print('朴素贝叶斯分类器的数学表示：')
        print('y=argmaxP(Y=ck)∏P(X(j)=x(j)|Y=ck)')
        print('4.1.2 后验概率最大化的含义')
        print('朴素贝叶斯法将实例分到后验概率最大的类中.这等价于期望风险最小化,假设选择0-1损失函数:')
        print('   L(Y,f(X))=1, Y!=f(X); L(Y,f(X))=0, Y=f(X);')
        print('式中f(X)是分类决策函数.这时,期望风险函数为Rexp(f)=E[L(Y,f(X))]')
        print('期望是对联合分布P(X,Y)取的.由此取条件期望:Rexp(f)=Ex∑[L(ck,f(X))]P(ck|X)')
        print('这样,根据期望风险最小化准则就得到了后验概率最大化准则:f(x)=argmaxP(ck|X=x),',
            '即朴素贝叶斯法所采用的原理')
        print('4.2 朴素贝叶斯法的参数估计')
        print('4.2.1 极大似然估计')
        print('在朴素贝叶斯法中,学习意味着估计P(Y=ck)和P(X(j)=x(j)|Y=ck).',
            '可以应用极大似然估计相应的概率.先验概率P(Y=ck)的极大似然估计是:',
            '   P(Y=ck)=∑I(yi=ck)/N,k=1,2,...,K')
        print('设第j个特征x(j)可能取值的集合为{aj1,aj2,...,ajSj},条件概率P(X(j)==ajl|Y=ck)',
            '的极大似然估计是',
            '   P(X(j)=ajl|Y=ck)=∑I(xi(l)=aji,yi=ck)/∑I(yi=ck),j=1,2,...,n;l=1,2,...,Sj;k=1,2,...,K')
        print('4.2.2 学习与分类算法')
        print('算法4.1 朴素贝叶斯算法(native Bayes algotithm).')
        print('输入:训练数据T={(x1,y1),(x2,y2),...,(xN,yN)},其中xi=(xi(1),xi(2),...,xi(n))^T,',
            'xi(j)是第i个样本的第j个特征,xi(j)∈{aj1,aj2,...,ajSj},ajl是第j个特征可能取的第l个值,',
            'j=1,2,...,n,l=1,2,...,Sj,yi∈{c1,c2,...,cK};实例x;')
        print('输出:实例x的分类')
        print('(1) 计算先验概率及条件概率')
        print('  P(Y=ck)=∑I(yi=ck)/N')
        print('  P(X(j)=ajl|Y=ck)=∑I(xi(l)=aji,yi=ck)/∑I(yi=ck)')
        print('  j=1,2,...,n; l=1,2,...,Sj;k=1,2,...,K')
        print('(2) 对于给定的实例x=(x(1),x(2),...,x(n))^T,计算')
        print('  P(Y=ck)∏P(X(j)=x(j)|Y=ck), k=1,2,...,K')
        print('(3) 确定实例x的类')
        print('  y=argmaxP(Y=ck)∏P(X(j)=x(j)|Y=ck)')
        print('例4.1 试由表4.1的训练数据学习一个朴素贝叶斯分类器并确定x=(2,S)^T的类标记',
            '表中X(1),X(2)为特征,取值的集合分别为A1={1,2,3},A2={S,M,L},Y为类标记,Y∈C={-1,1}')
        print('根据算法4.1,由随机变量表,容易计算下列概率:',
            'P(Y=1)=9/15,P(Y=-1)=6/15')
        print('P(X(1)=1|Y=1)=2/9, P(X(1)=2|Y=1)=3/9, P(X(1)=3|Y=1)=4/9')
        print('P(X(2)=S|Y=1)=1/9, P(X(2)=M|Y=1)=4/9, P(X(2)=L|Y=1)=4/9')
        print('P(X(2)=S|Y=-1)=3/6, P(X(2)=2|Y=-1)=2/6, P(X(2)=3|Y=-1)=1/6')
        print('P(X(S)=S|Y=-1)=3/6, P(X(2)=M|Y=-1)=2/6, P(X(2)=L|Y=-1)=1/6')
        print('对于给定的x=(2,S)^T计算：')
        print('P(Y=1)P(x(1)=2|Y=1)P(X(2)=S|Y=1)=9/15*3/9*1/9=1/45')
        print('P(Y=-1)P(x(1)=2|Y=-1)P(X(2)=S|Y=-1)=6/15*2/6*3/6=1/15')
        print('因为P(Y=-1)P(x(1)=2|Y=-1)最大,所以y=-1')
        print('4.2.3 贝叶斯估计')
        print('用极大似然估计可能会出现所要估计的概率值为0的情况.这时会影响到后验概率的计算结果,',
            '使分类产生偏差.解决这一问题的方法是采用贝叶斯估计.具体地,条件概率的贝叶斯估计是:')
        print('   Pla(X(j)=ajl|Y=ck)=(∑I(xi(j)=ajl,yi=ck)+la)/(∑I(yi=ck)+Sjla)')
        print('式中la>=0.等价于在随机变量各个取值的频数上赋予一个正数la>0.当la=0时,',
            '就是极大似然估计.常取la=1,这时称为拉普拉斯平滑.显然,对任何l=1,2,...,Sj',
            'k=1,2,..,K,有Pla(X(j)=ajl|Y=ck)>0, ∑P(X(j)=ajl|Y=ck)=1')
        print('表明式确为一种概率分布.同样,先验概率的贝叶斯估计是:')
        print('   Pla(Y=ck)=Pla(Y=ck)=(∑I(yi=ck)+la)/(N+Kla)')
        print('例4.2 问题同例4.1,按照拉普拉斯平滑估计概率,即取la=1')
        print('解:A1={1,2,3},A2={S,M,L},C={1,-1}.按照计算下列概率:')
        print('  P(Y=1)=10/17, P(Y=-1)=7/17')
        print('  P(X(1)=1|Y=1)=3/12, P(X(1)=2|Y=1)=4/12, P(X(1)=3|Y=1)=5/12')
        print('  P(X(2)=S|Y=1)=2/12, P(X(2)=M|Y=1)=5/12, P(X(2)=L|Y=1)=5/12')
        print('  P(X(2)=S|Y=-1)=4/9, P(X(2)=2|Y=-1)=3/9, P(X(2)=3|Y=-1)=2/9')
        print('  P(X(S)=S|Y=-1)=4/9, P(X(2)=M|Y=-1)=3/9, P(X(2)=L|Y=-1)=2/9')
        print('对于给定的x=(2,S)^T计算:')
        print('  P(Y=1)P(X(1)=2|Y=1)P(X(2)=S|Y=1)=10/17*4/12*2/12=5/153=0.0327')
        print('  P(Y=-1)P(X(1)=2|Y=-1)P(X(2)=S|Y=-1)=7/17*3/9*4/9=28/459=0.0610')
        print('由于P(Y=-1)P(X(1)=2|Y=-1)P(X(2)=S|Y=1)最大,所以y=-1')
        print('本章概要')
        print('1.朴素贝叶斯法是典型的生成学习方法.生成方法由训练数据学习联合概率分布P(X,Y)',
            '然后求得后验概率分布P(Y|X).具体来说,利用训练数据学习P(X|Y)和P(Y)的估计,',
            '得到联合概率分布:P(X,Y)=P(Y)P(X|Y)',
            '概率估计方法可以是极大似然估计或贝叶斯估计')
        print('2.朴素贝叶斯法的基本假设是条件独立性,')
        print('  P(X=x|Y=ck)=P(X(1)=x(1),...,X(n)=x(n)|Y=ck)=∏P(X(j)))')
        print(' 这是一个较强的假设.由于这一假设,模型包含的条件概率的数量大为减少,',
            '朴素贝叶斯法的学习与预测大为简化.因而朴素贝叶斯法高效,且易于实现',
            '其缺点是分类的性能不一定很高')
        print('3.朴素贝叶斯法利用贝叶斯定理与学到的联合概率模型进行分类预测.')
        print('  P(Y|X)=P(X,Y)/P(X)=P(Y)P(X|Y)/∑P(Y)P(X|Y)')
        print('将输入x分到后验概率最大的类y.')
        print('  y=argmaxP(Y=ck)∏P(Xj=x(j)|Y=ck)')
        print('后验概率最大等价于0-1损失函数时的期望风险最小化.')
        
chapter4 = Chapter4()

def main():
    chapter4.note()

if __name__ == '__main__':
    main()