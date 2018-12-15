
class Chapter8:
    """
    第8章 提升方法
    """
    def __init__(self):
        """
        第8章 提升方法
        """
        pass

    def note(self):
        """
        chapter8 note
        """
        print('第8章 提升方法')
        print('提升(boosting)方法是一种常用的统计学习方法,应用广泛且有效.',
            '在分类问题中,通过改变训练样本的权重,学习多个分类器,',
            '并将这些分类器线性组合,提高分类的性能')
        print('8.1 提升方法AdaBoost算法')
        print('8.1.1 提升方法的基本思路')
        print('提升方法基于这样一种思想:对于一个复杂任务来说,',
            '将多个专家的判断进行适当的综合所得出的判断.',
            '要比其中任何一个专家单独的判断好.',
            '类似“三个臭皮匠顶个诸葛亮”')
        print('“强可学习”和“弱可学习”.',
            '在概率近似正确(PAC)学习框架中,一个概念(一个类),如果存在一个多项式的学习算法能够学习它',
            '并且正确率很高,那么就称这个概念是强可学习的;一个概念,',
            '如果存在一个多项式的学习算法能够学习它,学习的正确率仅比随机猜测略好,',
            '那么就称这个概念是弱可学习的.')
        print('在PAC学习的框架下,一个概念是强可学习的充分必要条件是这个概念是弱可学习的.')
        print('对于分类问题而言,给定一个训练样本集,求比较粗糙的分类规则(弱分类器)要比求精确的分类规则(强分类器)',
            '容易的多,提升方法就是从弱学习算法出发,反复学习,得到一系列弱分类器(又称为基本分类器),',
            '然后组合这些弱分类器,构成一个强分类器.大多数提升方法都是改变训练数据的概率分布(训练数据的权值分布)',
            '针对不同的训练数据分布调用弱学习算法学习一系列弱分类器')
        print('AdaBoost算法做法是:提高那些被前一轮弱分类器错误分类样本的权值,',
            '而降低那些被正确分类样本的权值,这样一来,那些没有得到正确分类的数据,',
            '由于其权值的加大而受到后一轮的弱分类器的更大关注.')
        print('于是,分类问题被一系列的弱分类器“分而治之”.弱分类器的组合,',
            'AdaBoost采取加权多数表决的方法.具体地,加大分类误差率小的弱分类器的权值,',
            '使其在表决中起较大的作用,减小分类误差率大的弱分类器的权值,使其在表决中起较小的作用.',
            'AdaBoost的巧妙之处就在于它将这些想法自然且有效地实现在一种算法里')
        print('8.1.2 AdaBoost算法')
        print('假设给定一个二类分类的训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)}.',
            '其中,每个样本点由实例与标记组成.实例xi∈X∈R^n,标记yi∈Y={-1,+1},',
            'X是实例空间,Y是标记集合.AdaBoost利用一下算法,从训练数据中学习一系列弱分类器或基本分类器',
            '并将这些弱分类器线性组合成为一个强分类器')
        print('算法8.1 (AdaBoost)')
        print('输入:训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},其中xi∈X∈R^n,yi∈Y={-1,+1};弱学习算法;')
        print('输出:最终分类器G(x)')
        print('(1) 初始化训练数据的权值分布D1=(w11,...,w1i,...,w1N),w1i=1/N,w1i=1/N,i=1,2,...,N')
        print('(2) 对m=1,2,...,M')
        print('  (a) 使用具有权值分布Dm的训练数据集学习,得到基本分类器Gm(x)=X->{-1,+1}')
        print('  (b) 计算Gm(x)在训练数据集上的分类误差率')
        print('  (c) 计算Gm(x)的系数am=0.5log(1-em)/em')
        print('  (d) 更新训练数据集的权值分布')
        print('(3) 构建基本分类器的线性组合f(x)=∑amGm(x)')
        print('最终得到分类器:G(x)=sign(f(x))=sign(∑amGm(x))')
        print('步骤(1) 假设训练数据集具有均匀的权值分布,即每个训练样本在基本分类器的学习中作用相同,',
            '这一假设保证第1步能够在原始数据上学习基本分类器G1(x).')
        print('步骤(2) AdaBoost反复学习基本分类器,在每一轮m=1,2,...,M顺次地执行下列操作：')
        print('  (a) 使用当前分布Dm加权的训练数据集,学习基本分类器Gm(x)')
        print('  (b) 计算基本分类器Gm(x)在加权训练数据集上的分类误差率:')
        print('这里,wmi表示第m轮中第i个实例的权值,∑wmi=1.')
        print('Gm(x)在加权的训练数据集上的分类误差率是被Gm(x)误分类样本的权值之和,',
            '由此可以看出数据权值分布Dm与基本分类器Gm(x)的分类误差率的关系')
        print('  (c) 计算基本分类器Gm(x)的系数am·am表示Gm(x)在最终分类器中的重要性.',
            '当em<=0.5时,am>=0,并且am随着em的减小而增大,',
            '所以分类误差率越小的基本分类器在最终分类器的作用越大')
        print('  (d) 更新训练数据的权值分布为下一轮作准备.')
        print('由此可知,被基本分类器Gm(x)误分类样本的权值得以扩大,而被正确分类样本的权值却得以缩小.',
            '两相比较,误分类样本的权值被放大.因此,误分类样本在下一轮学习中起更大的作用.',
            '不改变所给的训练数据,而不断改变训练数据权值的分布,使得训练数据在基本分类器的学习中起不同的作用,',
            '这是AdaBoost的一个特点.')
        print('步骤(3) 线性组合f(x)实现M个基本分类器的加权表决.系数am表示了基本分类器Gm(x)的重要性',
            '这里,所有am之和并不为1.f(x)的符号决定实例x的类,f(x)的绝对值表示分类的确信度.',
            '利用基本分类器的线性组合构建最终分类器是AdaBoost的另一特点.')
        print('8.1.3 AdaBoost例子')
        print('例子8.1 给定如表8.1所示的训练数据.假设弱分类器由x<v或x>v产生,',
            '其阈值v使该分类器在训练数据集上分类误差率最低',
            '试用AdaBoost算法学习一个强分类器')
        print('解：初始化数据权值分布D1=(w11,w12,...,wl10),w1i=0.1,i=1,2,...,10.',
            '对m=1')
        print(' (a) 在权值分布为D1的训练数据上,阈值v取2.5时分类误差率最低,故基本分类器为',
            'G1(x)=1 x<2.5, G1(x)=-1 x>2.5')
        print(' (b) G1(x)在训练数据集上的误误差率e1=P(G1(xi)!=yi)=0.3')
        print(' (c) 计算G1(x)的系数:a1=0.5log(1-e1)/e1=0.4236')
        print(' (d) 更新训练数据的权值分布:D2=(w21,...,w2i,...,w110)',
            'w2i=w1i/Z1exp(-a1yiG1(xi)),i=1,2,...,10')
        print('  D2=(0.0715m0.0715m0.0715,...)')
        print('  f1(x)=0.4236G1(x)')
        print('分类器sign[f1(x)]在训练数据集上有3个误分类点')
        print('对m=2,')
        print(' (a) 在权值分布为D2的训练数据上,阈值v是8.5时分类误差率最低,基本分类器为：',
            'G2(x)=1, x<8.5; G2(x)=-1, x>8.5')
        print(' (b) G2(x)z在训练数据集上的误差率e2=0.2143')
        print(' (c) 计算a2=0.6496')
        print(' (d) 更新训练数据权值分布:D3=(...),f(2)=0.4236G1(x)+0.6496G2(x); f2(x)',
            '分类器sign[f2(x)]在训练数据集上有3个误分类点')
        print('对m=3,同理可得.')
        print('8.2 AdaBoost算法的训练误差分析.')
        print('定理8.1(AdaBoost的训练误差) 训练误差界为:')
        print('  1/N∑I(G(xi)!=yi)<=1/N∑exp(-yi,f(xi))=∏Zm')
        print('这一定理说明,可以在每一轮选取适当的Gm使得Zm最小,从而使训练误差下降最快.')
        print('定理8.2(二类分类问题AdaBoost的训练误差界)')
        print('  ∏Zm=∏[2sqrt(em(1-em))]=∏sqrt(1-4ym^2)<=exp(-2∑ym^2),这里ym=0.5-em')
        print('推论8.1 如果存在y>0,对所有m有ym>=y,则1/N∑I(G(xi)!=yi)<=exp(-2My^2)')
        print('  表明AdaBoost的训练误差是以指数速率下降的.')
        print('注意:AdaBoost算法不需要知道下界y.与一些早期的提升方法不同,AdaBoost具有适应性,',
            '即它能适应弱分类器各自的训练误差率.Ada是Adaptive(适应)的简写.')
        print('8.3 AdaBoost算法的解释')
        print('AdaBoost算法还有另一解释,即可以认为AdaBoost算法是模型为加法模型、损失函数为指数函数、',
            '学习算法为前向分步算法时的二类分类学习方法.')
        print('8.3.1 前向分布算法')
        print('考虑加法模型:f(x)=∑bmb(x;ym).其中,b(x;ym)为基函数,bm为基函数的系数.',
            '显然式(8,6)是一个加法模型.')
        print('在给定训练数据及损失函数L(y,f(x))的条件下,学习加法模型f(x)成为经验风险极小化即损失函数极小化问题：',
            'min ∑L(yi,∑bmb(xi;ym))')
        print('通常这是一个复杂的优化问题.前向分布算法(forward stagewise algorithm)求解',
            '这一优化问题的想法是:因为学习的是加法模型,如果能够从前向后,每一步只学习一个基函数及其系数,',
            '逐步逼近优化目标函数式,那么就可以简化优化的复杂度.具体地,每步只需优化的复杂度.',
            '具体地,每步只需优化如下损失函数:min∑L(yi,bb(xi;y))')
        print('给定训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},xi∈X∈R^n,',
            'yi∈Y={-1,+1}.损失函数L(y,f(x));基函数集{b(x;y)};')
        print('输出：加法模型f(x).')
        print('(1) 初始化f0(x)=0')
        print('(2) 对m=1,2,...,M')
        print('  (a) 极小化损失函数(bm,ym)=argmin∑L(yi,f(m-1)(xi)+bb(xi;y))',
            '得到参数bm,ym')
        print('  (b) 更新fm(x)=f(m-1)(x)+bmb(x;ym)')
        print('(3) 得到加法模型:f(x)=fM(x)=∑bmb(x;ym)')
        print('这样,前向分布算法将同时求解从m=1到M所有参数bm,ym的优化问题.')
        print('8.3.2 前向分布算法与AdaBoost')
        print('定理8.3 AdaBoost算法是前向分布加法算法的特例.这时,模型是由基本分类器组成的加法模型,',
            '损失函数是指数函数.')
        print('证明:前向分布算法学习的是加法模型,当基函数为基本分类器时,',
            '该加法模型等价于AdaBoost的最终分类器 f(x)=∑amGm(x)')
        print('由基本分类器Gm(x)及其系数am组成,m=1,2,...,M.前向分布算法逐一学习基函数,',
            '这一过程与AdaBoost算法逐一学习基本分类器的过程一致.')
        print('前向分布算法的损失函数是指数损失函数L(y,f(x))=exp[-yf(x)]')
        print('8.4 提升树')
        print('提升树是以分类树或回归树为基本分类器的提升方法.',
            '提升树被认为是统计学习中性能最好的方法之一')
        print('8.4.1 提升树模型')
        print('提升方法实际采用加法模型(即基函数的线性组合)与前向分布算法,',
            '以决策树为基函数的提升方法称为提升树(boosting tree).',
            '对分类问题决策树是二叉分类树',\
            '对回归问题决策树是二叉回归树.')
        print('在例8.1中看到的基本分类器x<v或x>v,可以看做是由一个根结点直接连接两个叶结点的简单决策树',
            '即所谓的决策树桩.提升树模型可以表示为决策树的加法模型:fM(x)=∑T(x;Θm)')
        print('其中,T(x;Θm)表示决策树;Θm为决策树的参数;M为树的个数.')
        print('8.4.2 提升树算法')
        print('提升树算法采用前向分步算法.首先确定初始提升树f0(x)=0,第m步的模型是：')
        print('fm(x)=fm-1(x)+T(x;Θm)')
        print('其中,fm-1(x)为当前模型,通过经验风险极小化确定下一棵决策树的参数Θm')
        print('  Θm=argmin∑L(yi,fm-1(xi)+T(xi;Θm))')
        print('由于树的线性组合可以很好地拟合数据,即使数据中的输入与输出之间的关系很复杂也是如此,',
            '所以提升树是一个高功能的学习算法.')
        print('针对不同问题的提升树学习算法,其主要区别在于使用的损失函数不同.',
            '包括用平方误差损失函数的回归问题,用指数损失函数的分类问题,',
            '以及用一般损失函数的一般决策问题.')
        print('对于二类分类问题,提升树算法只需将AdaBoost算法8.1中的基本分类器限制为二类分类树即可,',
            '可以说这时提升树算法是AdaBoost算法的特殊情况.')
        print('已知一个训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},',
            'xi∈X∈R^n,X为输入空间,yi∈Y∈R,Y为输入空间')
        print('如果将输入空间X划分为J个互不相交的区域R1,R2,...,RJ,并且在每个区域上确定输出的常量cj,',
            '那么树可表示为T(x;Θ)=∑cjI(x∈Rj)')
        print('其中,参数Θ={(R1,c1),(R2,c2),...,(RJ,cJ)}表示树的区域划分和各区域上的常数.',
            'J是回归树的复杂度即叶结点个数.')
        print('回归问题提升树使用一下前向分布算法:')
        print('  f0(x)=0')
        print('  fm(x)=fm-1(x)+T(x;Θ),m=1,2,...,M')
        print('  fM(x)=∑T(x;Θm)')
        print('在前向分布算法的第m步,给定当前模型fm-1(x),需求解')
        print('  Θm=argmin∑L(yi,fm-1(xi)+T(x;Θm))')
        print('得到Θm,即第m棵树的参数.')
        print('当采用平方误差损失函数时,L(y,f(x))=(y-f(x))^2')
        print('其损失变为:')
        print('L(y,fm-1(x)+T(x;Θm))=[y-fm-1(x)-T(x;Θm)]^2=[r-T(x;Θm)]')
        print('这里,r=y-fm-1(x)')
        print('是当前模型拟合数据的残差(residual).所以,对回归问题的提升树算法来说,',
            '只需简单地拟合当前模型的残差.')
        print('算法8.3(回归问题的提升树算法)')
        print('输入：训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},xi∈∈R^n,yi∈Y∈R;')
        print('输出：提升树fM(x)')
        print('(1) 初始化f0(x)=0')
        print('(2) 对m=1,2,...M)')
        print('  (a) 计算残差 rmi=yi-fm-1(xi),i=1,2,...,N')
        print('  (b) 拟合残差rmi学习一个回归树,得到T(x;Θm)')
        print('  (c) 更新fm(x)=fm-1(x)+T(x;Θm)')
        print('(3) 得到回归问题提升树fM(x)=∑T(x;Θm)')
        print('例8.2 如表所示的训练数据,x的取值范围为区间[0.5,10.5],',
            'y的取值范围为区间[5.0,10.0],学习这个回归问题的提升树模型,考虑只用树桩作为基函数.')
        print('解：按照算法8.3 第1步求f1(x)即回归树T1(x),首先求解以下优化问题:',
            'R1={x|x<=s},R2={x|x>s}')
        print('容易求得在R1,R2内部使平方损失误差达到最小值的c1,c2为:')
        print('  c1=1/N1∑yi, c2=1/N2∑yi')
        print('  这里N1,N2是R1,R2的样本点数.')
        print('求训练数据的切分点.根据所给数据,考虑如下切分点:',
            '1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5')
        print('对各切分点,不难求出相应的R1,R2,c1,c2及')
        print('  m(s)=min∑(yi-ci)^2+min∑(yi-ci)^2')
        print('例如,当s=1.5时,R1={1},R2={2,3,...,10},c1=5.56,c2=7.50')
        print('m(s)=min∑(yi-ci)^2+min∑(yi-c2)^2=0+15.72=15.72')
        print('现将s及m(s)的计算结果列表')
        print('由表可知,当s=6.5时m(s)达到最小值,此时R1={1,2,...,6},R2={7,8,9,10},',
            'c1=6.24,c2=8.91,所以回归树T1(x)为：T1(x)=6.24, x<6.5; T1(x)=8.91, x>=6.5')
        print('用f1(x)拟合训练数据的平方损失误差:L(y,f1(x))=∑(yi-f1(xi))^2=1.93')
        print('第2步求T2(x).方法与求T1(x)一样,只是拟合的数据是表的残差,可以得到:',
            'T2(x)=-0.52, x<3.5;  T2(x)=0.22, x>=3.5')
        print('f2(x)=f1(x)+T2(x),用f2(x)拟合训练数据的平方损失误差是：L(y,f2(x))=∑(yi-f2(xi))^2=0.79')
        print('可以继续求得T3(x),T4(x),T5(x),T6(x)')
        print('f6(x)=f5(x)+T6(x)=T1(x)+...+T5(x)+T6(x)')
        print('用f6(x)拟合训练数据的平方损失误差是L(y,f6(x))=∑(yi-f6(xi))^2=0.17')
        print('假设此时已满足误差要求,那么f(x)=f6(x)即为所求提升树')
        print('8.4.3 梯度提升')
        print('提升树利用加法模型与前向分布算法实现学习的优化过程.当损失函数是平方损失和指数损失函数时,',
            '每一步优化都是很简单的.但对一般损失函数而言,往往每一步优化并不是那么容易.',
            'Freidman提出了梯度提升算法.是利用最速下降法的近似方法,',
            '其关键是利用损失函数的负梯度在当前模型的值-[dL(y,f(xi))/df(xi)]f(x)=fm-1(x)')
        print('作为回归问题提升树算法中的残差的近似值,拟合一个回归树')
        print('算法8.4(梯度提升算法)')
        print('输入:训练数据集T={(x1,y1),(x2,y2),...,(xn,yn)},xi∈X∈R^n,yi∈Y∈R^n;',
            '损失函数L(y,f(x))')
        print('输出:回归树f(x)')
        print('(1) 初始化 f0(x)=argmin ∑L(yi,c)')
        print('(2) 对m=1,2,...,M')
        print('  (a) 对i=1,2,...,N,计算rmi=-[dL(y,f(xi))/df(xi)]f(x)=fm-1(x)')
        print('  (b) 对rmi拟合一个回归树,得到第m棵树的')
        print('  (c) 对j=1,2,...,J,计算 cmj=argmin∑L(yi,fm-1(xi)+c)')
        print('  (d) 更新fm(x)=fm-1(x)+∑cmjI(x∈Rmj)')
        print('(3) 得到回归树 f(x)=fM(x)=∑∑cmjI(x∈Rmj)')
        print('算法第1步初始化,估计使损失函数极小化的常数值,它是只有一个根结点的树.',
            '第2(a)步计算损失函数的负梯度在当前模型的值,将它作为残差的估计.',
            '对于平方损失函数,就是通常所说的残差;对于一般损失函数,就是残差的近似值.')
        print('第2(b)步估计回归树叶结点区域,以拟合残差的近似值.第2(c)步利用线性搜索估计叶结点区域的值,',
            '使损失函数极小化.第2(d)步更新回归树.第3步得到输出的最终模型f(x)')
        print('本章概要')
        print('1.提升方法是将弱学习算法提升为强学习算法的统计学习方法.',
            '在分类学习中,提升方法通过反复修改训练数据的权值分布,构建一系列的基本分类器(弱分类器)',
            '并将这些基本分类器线性组合,构成一个强分类器.代表性的提升方法是AdaBoost方法')
        print('2.AdaBoostu算法的特点是通过迭代每次学习的一个基本分类器.',
            '每次迭代中,提高那些被前一轮分类器错误分类数据的权值,',
            '而降低那些被正确分类的数据的权值.最后,AdaBoost将基本分类器的线性组合作为强分类器,',
            '其中给分类误差率小的基本分类器以大的权值,给分类误差率大的基本分类器以小的权值.')
        print('3.AdaBoost的训练误差分析表明,AdaBoost的每次迭代可以减少它在训练数据集上的分类误差率,',
            '这说明了它作为提升方法的有效性.')
        print('4.AdaBoost算法的一个解释是该算法实际是前向分布算法的一个实现.',
            '在这个方法里,模型是加法模型,损失函数是指数损失,算法是前向分布算法.',
            '每一步中极小化损失函数(bm,ym)=argmin∑L(yi,fm-1(xi)+bmb(xi;y))',
            '得到参数bm,ym')
        print('5.提升树是以分类树或回归树为基本分类器的提升方法.提升树被认为是统计学习中最有效的方法之一.')
        print('决策树、SVM、AdaBoost的比较')
        print('决策树')
        print('  真实应用场景：金融方面使用决策树建模分析，用于评估用户的信用，电商推荐系统')
        print('  优势：易于实现和理解，数据准备工作简单，同时处理多种数据类型，通过静态测试来对模型表现进行评价;',
            '可以在较短时间内对大量数据做出非常好的结果,决策树可以很好地扩展到大型数据中,',
            '同时决策树的大小独立于数据库的大小.计算复杂度低,结果易于理解,对部分数据损失不敏感.')
        print('  表现最好的情况:实例是由“属性-值”对表示的;目标函数具有离散的输出值;训练数据集包含分布错误',
            '(决策树对错误有适应性),训练数据缺少少量属性的实例')
        print('  缺点:易于出现过拟合问题;忽略了数据集中属性之间的相关性;',
            '对于类比不一致的样本,决策树的信息增益倾向于那些数据值较多的特征;')
        print('  什么条件表现很差:决策树匹配数据过多时,分类的类别过于复杂;数据的树形之间具有非常强的关联')
        print('  模型适应问题:不需要准备太多的训练数据,不需要对数据过多的处理如删除空白',
            '该问题是非线性问题,决策树能够很好地解决非线性问题.算法的执行效率高,对机器的要求小')
        print('支持向量机SVM')
        print('  真实应用场景：文本和超文本的分类,用于图像分类,用于手写体识别')
        print('  优势：分类效果好;可以有效地处理高维空间的数据,可以有效地处理变量个数大于样本个数的数据',
            '只是使用了一部分子集来进行训练模型,所以SVM模型不需要太大的内存;',
            '可以提高泛化能力,无局部极小值问题.')
        print('  表现最好的情况:数据的维度较高,需要模型具有非常强的泛化能力;样本数据量较小时',
            '解决非线性问题')
        print('  缺点:无法处理大规模的数据集,算法需要的时间较长的训练时间.',
            '无法有效地处理包含噪声太多的数据集;SVM模型没有直接给出概率的估计值,而是利用交叉验证的方式估计,',
            '这种方式耗时较长;对缺失数据敏感;对于非线性问题,有时很难找到一个合适的核函数')
        print('  什么条件表现很差:数据集的数据量过大;数据集中含有噪声;数据集中的缺失较多的数据;',
            '对算法的训练效率要求较高.')
        print('  模型适应问题:该项目所提供的样本数据相对较少;该问题是属于非线性问题;数据集经过“读热编码”后,维度较高')
        print('适应提升方法AdaBoost')
        print('  真实应用场景：二分类或多分类问题,用于特征选择,多标签问题,回归问题.')
        print('  优势：AdaBoost是一种精度非常高的分类器,可以与各种方法构建子分类器,Adaboost算法提供一种计算框架;',
            '弱分类器的构造方法比较简单;算法易于理解,不用做特征筛选,不易发生过拟合.易于编码')
        print('  表现最好的情况:用于解决二分类问题,解决大类单标签问题;处理多类单标签问题;处理回归相关的问题.')
        print('  缺点:AdaBoost算法的迭代次数不好设定,需要使用交叉验证的方式来进行确定;',
            '数据集的不平衡分布导致分类器的分类精度下降;训练比较耗费时间;对异常值比较敏感.')
        print('  什么条件表现很差:数据集分布非常不均匀,数据集中含有较多的异常值,对算法的训练的效率要求比较高')
        print('  模型适应问题:该数据集可以归属为多标签分类问题;数据集中异常值较少;',
            '对算法模型的准确率要求较高.')  

chapter8 = Chapter8()

def main():
    chapter8.note()

if __name__ == '__main__':
    main()