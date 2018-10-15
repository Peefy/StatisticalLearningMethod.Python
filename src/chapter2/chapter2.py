
class Chapter2:
    """
    第2章 感知机
    """
    def __init__(self):
        """
        第2章 感知机
        """
        pass

    def note(self):
        """
        chapter2 note
        """
        print('第2章 感知机')
        print('感知机(perception)是二类分类的线性分类模型,其输入为实例的特征向量,输出为实例的类别,取+1和-1二值')
        print('感知机对应于输入空间(特征空间)中将实例划分为正负两类的分离超平面,属于判别模型.感知机学习旨在求出将训练数据进行线性划分的分离超平面')
        print('为此,导入基于误分类的损失函数,利用梯度下降法对损失函数进行极小化,求得感知机模型.')
        print('感知机学习算法具有简单而易于实现的优点,分为原始形式和对偶形式.感知机预测是用学习得到的感知机模型对新的输入实例进行分类.')
        print('感知机1957年由Rosenblatt提出,是神经网络与支持向量机的基础.')
        print('本章介绍感知机模型、学习策略(损失函数)、学习算法(原始形式，对偶形式),并证明算法的收敛性')
        print('2.1 感知机模型')
        print('定义2.1 (感知机) 假设输入空间(特征空间)是X∈R^n,输出空间是Y={+1,-1}.',
            '输入x∈X表示实例的特征向量,对应于输入空间(特征空间)的点；输出y∈Y表示实例的类别.',
            '由输入空间到输出空间的如下函数:f(x)=sign(w·x+b)称为感知机',
            '其中,w和b为感知机模型参数,w∈R^n叫做权值(weight)或权值向量,b∈R叫作偏置,w·x表示w和x的内积.sign是符号函数')
        print('sign(x)=1,x>=0; sign(x)=-1,x<0')
        print('感知机是一种线性分类模型,属于判别模型.感知机模型的假设空间是定义在特征空间中的所有线性分类模型或者线性分类器,',
            '即函数集合{f|f(x)=w·x+b}')
        print('感知机有如下几何解释:线性方程w·x+b=0对应于特征空间R^n中的一个超平面S,其中w是超平面的法向量,b是超平面的截距.')
        print('这个超平面将特征空间划分为两个部分.位于两部分的点(特征向量)分别被正、负两类.因此,超平面S称为分离超平面')
        print('感知机学习,由训练数据集(实例的特征向量及类别).其中,xi∈X=R^n,yi∈Y={+1,-1},i=1,2,...,N,求得感知机模型,',
            '即求得模型参数w,b.感知机预测,通过学习得到的感知机模型,对于信的输入实例给出其对应的输出类别')
        print('2.2 感知机学习策略')
        print('2.2.1 数据集的线性可分性')
        print('定义2.2 (数据集的线性可分性) 给定一个数据集T={(x1,y1),(x2,y2),...,(xn,yn)}')
        print('其中,xi∈X=R^n,yi∈Y={+1,-1},i=1,2,...,N,如果存在某个超平面S,w·x+b=0能够将数据集的正实例点和负实例点完全正确地划分到超平面的两侧,',
            '即对所有yi=+1的实例i,有w·xi+b>0,对所有yi=-1的实例i,有w·xi+b<0,则称数据集T为线性可分数据集;否则,称数据集T线性不可分')
        print('2.2.2 感知机学习策略')
        print('假设训练数据集是线性可分的,感知机学习的目标是求得一个能够将训练集正实例点和负实例点完全正确分开的分离超平面.为了找出这样的超平面,',
            '即确定感知机模型参数w,b,需要确定一个学习策略,即定义(经验)损失函数并将损失函数极小化')
        print('损失函数的一个自然选择是误分类点的总数.但是,这样的损失函数不是参数w,b的连续可导函数,不易优化.',
            '损失函数的另一个选择是误分类点到超平面S的总距离,这是感知机所采用的.为此,首先写出输入空间R^n中任一点x0到超平面S的距离:',
            '1/||w|||w·x0+b|')
        print('这里,||w||是w的L2范数')
        print('其次,对于误分类的数据(xi,yi)来说:-yi(w·xi+b)>0')
        print('成立.因此当w·xi+b>0时,yi=-1,而当w·xi+b<0时,yi=+1.因此,误分类点xi到超平面S的距离-1/||w||yi(w·xi+b)')
        print('这样,假设超平面S的误差分类点集合为M,那么所有误分类点到超平面S的总距离为：-1/||w||∑yi(w·xi+b)')
        print('不考虑1/||w||,就得到感知机学习的损失函数')
        print('给定训练数据集:T={(x1,y1),(x2,y2),...,(xN,yN)}')
        print('其中,xi∈X=R^n,yi∈Y={+1,-1},i=1,2,...,N.感知机sign(w·x+b)学习的损失函数定义为L(w,b)=-∑yi(w·xi+b)',
            '其中M为误分类点的集合.这个损失函数就是感知机学习的经验风险函数')
        print('显然,损失函数L(w,b)是非负的.如果没有误分类点,损失函数值是0.而且,误分类点越少,误分类点离超平面越近,损失函数值就越小.',
            '一个特定的样本点的损失函数:在误分类时是参数w,b的线性函数,在正确分类时是0.因此,给定训练数据集T,损失函数L(w,b)是w,b的连续可导函数')
        print('感知机学习的策略是在假设空间中选取使损失函数最小的模型参数w,b,即感知机模型.')
        print('2.3 感知机学习算法')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')
        print('')

chapter2 = Chapter2()

def main():
    chapter2.note()

if __name__ == '__main__':
    main()