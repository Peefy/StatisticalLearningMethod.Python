
class Chapter5:
    """
    第5章 决策树
    """
    def __init__(self):
        """
        第5章 决策树模型与学习
        """
        pass

    def note(self):
        """
        chapter5 note
        """
        print('第5章 决策树模型')
        print('5.1 决策树模型与学习')
        print('5.1.1 决策树模型')
        print('决策树是一种基本的分类与回归方法.本章主要讨论用于分类的决策树.',
            '决策树模型呈树形结构.在分类问题中,表示基于特征对实例进行分类的过程',
            '可以认为是if—then规则的集合,也可以认为是定义在特征空间与类空间上的条件概率分布.',
            '其主要优点是模型具有可读性,分类速度快.学习时,利用训练数据,根据损失函数最小化的原则建立决策树模型.',
            '预测时,对新的数据,利用决策树模型进行分类.')
        print('决策树学习通常包括3个步骤:特征选择、决策树的生成和决策树的修剪.')
        print('5.1 决策树模型与学习')
        print('5.1.1 决策树模型')
        print('定义5.1(决策树)分类决策树模型是一种描述对实例进行分类的树形结构.',
            '决策树由结点(node)和有向边(directed edge)组成.结点有两种类型：',
            '内部结点(internal node)和叶结点(leaf mode).内部结点表示一个特征或属性,',
            '叶结点表示一个类')
        print('用决策树分类,从根结点开始,对实例的某一特征进行测试,根据测试结果,',
            '将实例分配到其子结点;这时,每一个子结点对应着该特征的一个取值.',
            '如此递归地对实例进行测试并分配,直至达到叶结点.最后将实例分到叶结点的类中')
        print('5.1.2 决策树与if-then规则')
        # !可以将决策树看成一个if-then规则的集合
        print('可以将决策树看成一个if-then规则的集合.将决策树转换成if-then规则过程是这样的:',
            '由决策树的根结点到叶结点的每一条路径构建一条规则;',
            '路径上的内部结点的特征对应着规则的条件,而叶结点的类对应着规则的结论.',
            '决策树的路径或其对应的if-then规则集合具有一个重要的性质:互斥并且完备,',
            '这就是说,每一个实例都被一条路径或一条规则所覆盖,而且只被一条路径或一条规则所覆盖.',
            '这里所谓覆盖是指实例的特征与路径上的特征一致或实例满足规则的条件')
        print('5.1.3 决策树与条件概率分布')
        print('决策树还表示给定特征条件下类的条件概率分布.这一条件概率分布定义在特征空间的一个划分(partition)上.',
            '将特征空间划分为互不相交的单元(cell)或区域(region),并在每个单元定义一个类的概率分布就构成了一个条件概率分布',
            '由各个单元给定条件下类的条件概率分布组成.','假设X为表示特征的随机变量,Y为表示类的随机变量,',
            '那么这个条件概率分布可以表示为P(Y|X).X取值于给定划分下单元的集合,Y取值于类的集合.',
            '各叶结点(单元)上的条件概率往往偏向某一个类,即属于某一类的概率较大.',
            '决策树分类时将该结点的实例强行分到条件概率大的那一类去.')
        print('5.1.4 决策树学习')
        print('假设给定训练数据集D={(x1,y1),(x2,y2),...,(xn,yn)}')
        print('其中,xi=(xi(1),xi(2),...,xi(n))^T为输入实例(特征向量),n为特征个数,yi∈{1,2,...,K}为类标记,',
            'i=1,2,...,N,N为样本容量.学习的目标是根据给定的训练数据集构建一个决策树模型,使它能够对实例进行正确的分类.')
        print('决策树学习本质上是从训练数据集中归纳出一组分类规则.与训练数据集不相矛盾的决策树',
            '(即能对训练数据进行正确分类的决策树)可能有多个,也可能一个也没有')
        print('需要的是一个与训练数据矛盾较小的决策树,同时具有很好的泛化能力.从另一个角度看,',
            '决策树学习是由训练数据集估计条件概率模型.基于特征空间划分的类的条件概率模型有无穷多个.',
            '选择调剂那概率模型应该不仅对训练数据由很好的拟合,而且对未知数据有很好的预测.')
        print('决策树学习用损失函数表示这一目标.如下所述,决策树学习的损失函数通常是正则化的极大似然函数.',
            '决策树学习的策略是以损失函数为目标函数的最小化.')
        print('当损失函数确定以后,学习问题就变为在损失函数意义下的选择最优决策树的问题.',
            '因为从所有可能的决策树中选取最优决策树是NP完全问题,所以现实中决策树学习算法通常采用启发式方法,',
            '近似求解这一最优化问题.这样的得到的决策树是次最优的.')
        print('决策树学习的算法通常是一个递归地选择最优特征,并根据该特征对训练数据进行分割,',
            '使得对各个子数据集有一个最好的分类的过程.这一过程对应着特征空间的划分,对应着决策树的构建.')
        print('开始,构建根结点,将所有的训练数据都放在根结点.选择一个最优特征,按照这一特征将训练数据集分割成子集,',
            '使得各个子集有一个当前条件下最好的分类.如果这些子集已经能够被基本正确分类,',
            '那么构建叶结点,并将这些子集分到所对应的叶结点中去;如果还有子集不能基本正确分类,',
            '那么就对这些子集选择新的最优特征,继续对其进行分割,构建相应的结点.',
            '如此递归地进行下去,直至所有的训练数据子集被基本正确分类,或者没有合适的特征为止.',
            '最后每个子集都被分到叶结点上,即都有了明确的类.这就生成了一棵决策树')
        # !剪枝可以提高决策树预测的能力，避免过拟合
        print('以上方法生成的决策树可能对训练数据有很好的分类能力,但对未知的测试数据却未必有很好的分类能力,',
            '即可能发生过拟合现象.需要对已经生成的树自上而下进行进行剪枝,将树变得简单,从而使它具有更好的泛化能力',
            '具体地,就是去掉过于细分的叶结点,使其回退到父结点,甚至更高的结点,然后将父结点或更高的结点改为新的叶结点')
        print('如果特征数量很多,也可以在决策树学习开始的时候,对特征进行选择,只留下对训练数据有足够分类能力的特征.',
            '可以看出,决策树学习算法包含特征选择、决策树的生成与决策树的剪枝过程.',
            '由于决策树表示一个条件概率分布,所以深浅不同的决策树对应着不同复杂度的概率模型.',
            '决策树的生成对应于模型的局部选择,决策树的剪枝对应于模型的全局选择.',
            '决策树的生成只考虑局部最优,相对地,决策树的剪枝则考虑全局最优')
        print('决策树学习常用的算法有ID3、C4.5与CART,下面结合这些算法分别叙述决策树学习的特征选择、',
            '决策树的生成和剪枝过程')
        print('5.2 特征选择')
        print('5.2.1 特征选择问题')
        print('特征选择在于选取对训练数据具有分类能力的特征.这样可以提高决策树学习的效率.',
            '如果利用一个特征进行分类的结果与随机分类的结果没有很大差别,则称这个特征是没有分类能力的',
            '经验上扔掉这样的特征对决策树学习的精度影响不大.通常特征选择的准则是信息递增或信息增益比.')
        print('例5.1 表5.1是一个由15个样本组成的贷款申请训练数据.数据包包括贷款申请人的4个特征(属性):',
            '第1个特征是年龄,有3个可能值:青年,中年,老年;第2个特征是有工作,有2个可能值:是,否;',
            '第3个特征是有自己的房子,有2个可能值:是,否;',
            '第4个特征是信贷情况,有3个可能值:非常好,好,一般.',
            '表的最后一列是类别,是否同意贷款,取2个值:是,否.')
        print('希望通过所给的训练数据学习一个贷款申请的决策树,用以对未来的贷款申请进行分类,',
            '即当新的客户提出贷款申请时,根据申请人的特征利用决策树决定是否批准贷款申请.')
        # !特征选择是决定用哪个特征来划分特征空间.
        print('特征选择是决定用哪个特征来划分特征空间.')
        print('5.2.2 信息增益')
        print('在信息论与概率统计中,熵是表示随机变量不确定性的度量,',
            '设X是一个取有限个值的离散随机变量,其概率分布为:P(X=xi)=pi,i=1,2,..,n',
            '则随机变量X的熵定义为:H(X)=-∑pilogpi;若pi=0,则定义0log0=0,对数以2为底或以e为底(自然对数)',
            '这时熵的单位分别称作比特(bit)或纳特(nat).由定义可知,熵只依赖于X的分布,而与X的取值无关,',
            '所以也可将X的熵记做H(p),即H(p)=-∑pilogpi')
        print('熵越大,随机变量的不确定性就越大.从定义可验证:0<H(p)<=logn')
        print('当随机变量只取两个值,例如1,0时,即X的分布为:P(X=1)=p,P(X=0)=1-p,0<=p<=1',
            '熵为H(p)=-plog2p-(1-p)log2(1-p)')
        print('这时,熵H(p)随概率p变化的曲线如图5.4所示(单位为比特)')
        print('当p=0或p=1时H(p)=0,随机变量完全没有不确定性.当p=0.5时,H(p)=1,熵取值最大,随机变量不确定性最大.')
        print('设有随机变量(X,Y),其联合概率分布为:P(X=xi,Y=yi)=pij,i=1,2,...,n;j=1,2,...,m')
        print('条件熵H(Y|X)表示在已知随机变量X的条件下随机变量Y的不确定性.随机变量X给定的条件下随机变量Y的条件熵H(Y|X)',
            '定义为X给定条件下Y的条件概率分布的熵对X的数学期望H(Y|X)=∑piH(Y|X=xi)',
            '这里,pi=P(X=xi),i=1,2,...,n')
        print('当熵和条件熵中的概率由数据估计(特别是极大似然估计)得到时,所对应的熵与条件熵分别称为经验熵和经验条件熵.',
            '此时，如果有0概率,令0log0=0')
        print('信息增益表示得知特征X的信息而使得类Y的信息不确定性减少的程度')
        print('定义5.2(信息增益)特征A对训练数据集D的信息增益g(D,A),',
            '定义为集合D的经验熵H(D)与特征A给定条件下D的经验条件熵H(D|A)之差,即',
            'g(D,A)=H(D)-H(D|A)')
        print('一般地,熵H(Y)与条件熵H(Y|X)之差称为互信息.决策树学习中的信息增益等价于训练数据集中类与特征的互信息')
        print('决策树学习应用信息增益准则选择特征.给定训练数据集D和特征A,经验熵H(D)表示对数据集D进行分类的不确定性.',
            '而经验条件熵H(D|A)表示在特征值A给定的条件下对数据集D进行分类的不确定性.',
            '那么它们的差,即信息增益,就表示由于特征A而使得对数据集D的分类的不确定性减少的程度.',
            '显然,对于数据集D而言,信息增益依赖于特征,不同的特征往往具有不同的信息增益.',
            '信息增益大的特征具有更强的分类能力')
        print('根据信息增益准则特征选择方法是:对训练数据集(或子集)D,计算其每个特征的信息增益,',
            '并比较他们的大小,选择信息增益最大的特征.')
        print('设训练数据集为D,|D|表示其样本容量,即样本个数.设有K个类Ck,k=1,2,...,K,|Ck|为属于类Ck的样本个数',
            '∑|Ck|=|D|.设特征A有n个不同的取值{a1,a2,...,an},根据特征A的取值将D划分为n个子集D1,D2,...,Dn',
            '|Di|为Di的样本个数,∑|Di|=|D|.记子集Di中属于类Ck的样本的集合为Dik,',
            '即Dik=Di∩Ck,|Dik|为Dik的样本个数.于是信息增益的算法如下:')
        print('算法5.1（信息增益的算法）')
        print('输入:训练数据集D和特征A;')
        print('输出:特征A对训练数据集D的信息增益g(D,A)')
        print('(1) 计算数据集D的经验熵H(D)=-∑|Ck|/|D|log2|Ck|/|D|')
        print('(2) 计算特征A对数据集D的经验条件熵H(D|A)')
        print('(3) 计算信息增益:g(D,A)=H(D)-H(D|A)')
        print('例5.2 对表5.1所给的训练数据集D,根据信息增益准则选择最优特征.')
        print('解:首先计算经验熵H(D)=-9/15log2(9/15)-6/15log2(6/15)=0.971')
        print('然后计算各特征对数据集D的信息增益.分别以A1,A2,A3,A4表示',
            '年龄、有工作、有自己的房子和信贷情况4个特征')
        print('(1) g(D,A1)=H(D)-[5/15H(D1)+5/15H(D2)+5/15H(D3)]=',
            '0.9710-[5/15(-2/5log2(2/5)-3/5log2(3/5))+5/15(-3/5log2(3/5)-2/5log2(2/5)+5/15(-4/5log2(4/5)-1/5log2(1/5))]=0.971-0.888=0.083')
        print('这里D1,D2,D3分别是D中A1(年龄)取值为青年、中年和老年的样本子集.类似的,')
        print('(2) g(D,A2)=H(D)-[5/15H(D1)+10/15H(D2)]=0.971-[5/15*0+10/15(-4/10log2(4/10)-6/10log2(6/10))]=0.324')
        print('(3) g(D,A3)=0.971-[6/15*0+9/15(-3/9log2(3/9)-6/9log2(6/9))]=0.971-0.551=0.42')
        print('(4) g(D,A4)=0.971-0.608=0.363')
        print('最后,比较各特征的信息增益值.由于特征A3(有自己的房子)的信息增益值最大,所以选择特征A3作为最优特征')
        print('5.2.3 信息增益比')
        print('信息增益值的大小是相对于训练数据集而言的,并没有绝对意义.在分类问题困难时,',
            '也就是说在训练数据集的经验熵大的时候,信息增益值会偏大.反之,信息增益值会偏小.',
            '使用信息增益比可以对这一问题进行校正.这是特征选择的另一准则.')
        print('定义5.3（信息增益比）特征A对训练数据集D的信息增益比gR(D,A)定义为其信息增益g(D,A)',
            '与训练数据集D的经验熵H(D)之比:gR(D,A)=g(D,A)/H(D)')
        print('5.3 决策树的生成')
        print('5.3.1 ID3算法')
        print('ID3算法的核心是在决策树各个结点上应用信息增益准则选择特征,递归地构建决策树.',
            '具体方法是:从根结点(root node)开始,对结点计算所有可能的特征的信息增益，',
            '选择信息增益最大的特征作为结点的特征,由该特征的不同取值建立子结点;',
            '再对子结点递归地调用以上方法,构建决策树;知道所有的特征信息增益均很小或没有特征可以选择为止.',
            '最后得到一个决策树.ID3相当于用极大似然法进行概率模型的选择')
        print('算法5.2（ID3算法）')
        print('输入:训练数据集D，特征集A，阈值e')
        print('输出:决策树T')
        print('(1) 若D中所有实例属于同一类Ck,则T为单结点树,并将类Ck作为该结点的类标记,返回T;')
        print('(2) 若A!=None,则T为单结点树,并将D中实例数最大的类Ck作为该结点的类标记,返回T;')
        print('(3) 否则,按算法5.1计算A中各特征对D的信息增益,选择让信息增益最大的特征Ag;')
        print('(4) 如果Ag信息增益小于阈值e,则置T为单结点树,并将D中实例数最大的类Ck作为该结点的类标记,返回T;')
        print('(5) 否则,对Ag的每一可能值ai,依Ag=ai将D分割为若干非空子集Di,将Di中实例数最大的类作为标记,',
            '构建子结点,由结点及其子结点构成树T,返回T;')
        print('(6) 对第i个子结点,以Di为训练集,以A-{Ag}为特征值,递归地调用步(1)-步(5),得到子树Ti,返回Ti')
        print('例5.3 对表5.1的训练数据集,利用ID3算法建立决策树')
        print('解:利用例5.2的结果,由于特征A3(有自己的房子)的信息增益值最大,所以选择特征A3作为根结点的特征.',
            '它将训练数据集D划分为两个子集D1(A3取值为“是”)和D2(A3取值为“否”).由于D1只有同一类的样本点,',
            '所以它成为一个叶结点,结点的类标记为“是”')
        print('对D2则需要从特征A1(年龄)，A2(有工作)和A4(信贷情况)中选择新的特征.计算各个特征的信息增益:')
        print('  g(D2,A1)=H(D2)-H(D2|A1)=0.918-0.667=0.251')
        print('  g(D2,A2)=H(D2)-H(D2|A2)=0.918')
        print('  g(D2,A4)=H(D2)-H(D2|A4)=0.474')
        print('选择信息增益最大的特征A2(有工作)作为结点的特征.由于A2有两个可能的取值,',
            '从这一结点引出两个子结点:一个对应“是”(有工作)的子结点,包含3个样本,它们属于同一类,',
            '所以这是一个叶结点,类标记为“是”;另一个是对应“否”(无工作)的子结点,包含6个样本,也属于同一类,',
            '所以这也是一个叶结点,类标记为“否”')
        print('注意:ID3算法只有树的生成,所以该算法生成的数容易产生过拟合')
        print('5.3.2 C4.5的生成算法')
        print('C4.5算法与ID3算法相似,C4.5算法对ID3算法进行了改进.C4.5在生成的过程中,用信息增益比来选择特征.')
        print('算法5.3(C4.5的生成算法)')
        print('输入：训练数据集D,特征集A,阈值e;')
        print('输出:决策树T')
        print('(1) 如果D中所有实例属于同一类Ck,则置T为单结点树,并将Ck作为该结点的类,返回T')
        print('(2) 如果A=None,则置T为单结点树,并将D中实例数最大的类Ck作为该结点的类,返回T;')
        print('(3) 否则,按式(5.10)计算A中各特征对D的信息增益比,选择信息增益比最大的特征Ag;')
        print('(4) 如果Ag的信息增益比小于阈值e,则置T为单结点树,并将D中实例数最大的类Ck作为该结点的类,返回T;')
        print('(5) 否则,对Ag的每一可能值ai,依Ag=ai将D分割为子集若干非空Di,将Di中实例数最大的类作为标记,',
            '构建子结点,由结点及其子结点构成树T,返回T;')
        print('(6) 对结点i,以Di为训练集,以A-{Ag}为特征集,递归地调用(1)~(5),得到子树Ti,返回Ti')
        print('5.4 决策树的剪枝')
        print('决策树生成算法递归地产生决策树,直到不能继续下去为止.这样产生的树往往对训练数据的分类很准确,',
            '但对未知的测试数据的分类没有那么准确,即出现过拟合现象.')
        # !对决策树进行剪枝可以降低复杂度，避免过拟合现象.
        print('对决策树进行剪枝可以降低复杂度，避免过拟合现象.')
        print('具体来说,就是从已经生成的树上裁减掉一些子树或者叶子结点,',
            '并将其根结点或父结点作为新的叶结点,从而简化分类树模型')
        # !决策树的剪枝往往通过极小化决策树整体的损失函数或代价函数来实现.
        print('决策树的剪枝往往通过极小化决策树整体的损失函数或代价函数来实现.')
        print('设树T的叶结点个数为|T|,t是树T的叶结点,该叶结点有Nt个样本点,',
            '其中k类的样本点有Ntk个,k=1,2,...,K,Ht(T)为叶结点t上的经验熵,a>=0为参数,',
            '则决策树学习的损失函数可以定义为:Ca(T)=∑NtHt(T)+a|T|')
        print('其中经验熵为：Ht(T)=-∑Ntk/NtlogNtk/Nt,在损失函数中,将上上式右端写为:')
        print('  C(T)=∑NtHt(T)=-∑∑Ntklog(Ntk/Nt),这时有:Ca(T)=C(T)+a|T|')
        print('其中,C(T)表示模型对训练数据的预测误差,即模型与训练数据的拟合程度,|T|表示模型复杂度,',
            '参数a>=0控制两者之间的影响.较大的a促使选择较简单的模型(树),较小的a促使选择较复杂的模型(树)',
            'a=0意味着只考虑模型与训练数据的拟合程度,不考虑模型的复杂度')
        print('剪枝,就是当a确定时,选择损失函数最小的模型,即损失函数最小的子树',
            '当a的值确定时,子树越大,往往与训练数据的拟合越好,但是模型的复杂度就越高',
            '相反,子树越小,模型的复杂度就越低,但是往往与训练数据的拟合不好',
            '损失函数正好表示对两者的平衡')
        print('可以看出,决策树生成只考虑了通过提高信息增益(或信息增益比)对训练数据进行更好的拟合.',
            '而决策树剪枝通过优化损失函数还考虑了减小模型复杂度.决策树生成学习局部的模型',
            '而决策树剪枝学习整体的模型')
        # !利用损失函数最小原则进行剪枝就是用正则化的极大似然估计进行模型选择
        print('利用损失函数最小原则进行剪枝就是用正则化的极大似然估计进行模型选择')
        print('算法5.4(树的剪枝算法)')
        print('输入:生成算法产生的整个树T,参数a;')
        print('输出:修剪后的子树Ta')
        print('(1) 计算每个结点的经验熵')
        print('(2) 递归地从树的叶结点向上回缩')
        print('(3) 如果剪枝后整体树的损失函数值更小，则进行剪枝,将父结点变为新的叶结点')
        print('注意:剪枝时只需考虑两个树的损失函数的差,其计算可以在局部进行.所以,',
            '决策树的剪枝算法可以由一种动态规划的算法实现.')
        print('5.5 CART算法')
        print('分类与回归树(classfication and regression tree, CART)模型由1984年提出,',
            '是应用广泛的决策树学习方法.CART同样由特征选择,树的生成及剪枝组成,',
            '既可以用于分类也可以用于回归.以下将用于分类与回归的树统称为决策树')
        print('CART是在给定输入随机变量X条件下输出随机变量Y的条件概率分布的学习方法.CART假设决策树是二叉树,',
            '内部结点特征的取值为“是”和“否”,左分支是取值为“是”的分支,右分支是取值为“否”的分支')
        print('这样的决策树等价于递归地二分每个特征,将输入空间即特征空间互粉为有限个单元,',
            '并在这些单元上确定预测的概率分布,也就是在输入给定的条件下输出的条件概率分布.')
        print('CART算法由以下两步组成')
        print('(1) 决策树生成:基于训练数据集生成决策树,生成的决策树要尽量大;')
        print('(2) 决策树剪枝:用验证数据集对已经生成的树进行剪枝并选择最优子树,',
            '这时用损失函数最小作为剪枝的标准')
        print('5.5.1 CART生成')
        print('决策树的生成就是递归地构建二叉决策树的过程.对回归树用平方误差最小化准则,',
            '对分类树用基尼指数(Gini index)最小化准则,进行特征选择,生成二叉树')
        print('1.回归树的生成.')
        print('假设X与Y分别为输入和输出变量,并且Y是连续变量,给定训练数据集D={(x1,y1),(x2,y2),...,(xn,yn)}',
            '考虑如何生成回归树.')
        print('一个回归树对应着输入空间(即特征空间)的一个划分以及在划分的单元上的输出值.',
            '假设已将输入空间划分为M个单元R1,R2,...,RM,并且在每个单元Rm上有一个固定的输出值cm,',
            '于是回归树模型可表示为:f(x)=∑cmI(x∈Rm)')
        print('当输入空间的划分确定时,可以用平方误差∑(yi-f(xi))^2来表示回归树对于训练数据的训练误差',
            '用平方误差最小的准则求解每个单元上的最优输出值.易知,单元Rm上的cm最优值cm是Rm上的所有输入实例xi对应的输出yi的均值,',
            '即cm=ave(yi|xi∈Rm)')
        print('问题是这样对输入空间进行划分.这里采用启发式的方法,选择第j个变量x(j)和它的取的值s,',
            '作为切分变量和切分点,并定义两个区域：R1(j,s)={x|x(j)<=s}和R2(j,s)={x|x(j)>s}')
        print('然后寻找最优切分变量j和最优切分点s.具体地,求解')
        print('对固定输入变量j可以找到最优切分点s')
        print('遍历所有输入变量,找到最优的切分变量j,构成一个对(j,s).依次将输入空间划分为两个区域.',
            '接着,对每个区域重复上述划分过程,直到满足停止条件为止.这样就生成一棵回归树.',
            '这样的回归树通常称为最小二乘回归树(least squares regression tree)')
        print('算法5.5(最小二乘回归树生成算法)')
        print('输入:训练数据集D;')
        print('输出:回归树f(x)')
        print('在训练数据集所在的输入空间中,递归地将每个区域划分为两个子区域并决定每个子区域上的输出值,',
            '构建二叉决策树:')
        print('(1) 选择最优切分变量j与切分点s,求解:min[min∑()^2+min∑()^2]')
        print('  遍历变量j,对固定的切分变量j扫描切分点s,选择使式达到最小值的对(j,s)')
        print('(2) 用选定的对(j,s)划分区域并决定相应的输出值:')
        print('  R1(j,s)={x|x(j)<=s},R2(j,s)={x|x(j)>s}')
        print('  cm=1/Nm∑yi,x∈Rm,m=1,2')
        print('(3) 继续对两个子区域调用步骤(1),(2),直至满足停止条件.')
        print('(4) 将输入空间划分为M个区域R1,R2,...,RM,生成决策树:f(x)=∑cmI(x∈Rm)')
        print('2. 分类树的生成')
        print('分类树用基尼指数选择最优特征,同时决定该特征的最优二值切分点')
        print('定义5.4（基尼指数）分类问题中,假设有K个类,样本点属于第k类的概率为pk,则概率分布的基尼指数定义为:')
        print('  Gini(p)=∑pk(1-pk)=1-∑pk^2')
        print('对于二类分类问题,若样本点属于第1个类的概率是p,则概率分布的基尼指数为Gini(D)=1-∑(|Ck|/|D|)^2')
        print('这里,Ck是D中属于第k类的样本子集,K是类的个数')
        print('如果样本集合D根据特征A是否取某一可能值a被分割成D1和D2两部分,即')
        print('   D1={(x,y)∈D|A(x)=a},D2=D-D1')
        print('则在特征A的条件下,集合D的基尼指数定义为:')
        print('   Gini(D,A)=|D1|/|D|Gini(D1)+|D2|/|D|Gini(D2)')
        print('基尼指数Gini(D)表示集合D的不确定性,基尼指数Gini(D,A)表示经A=a分割后',
            '集合D的不确定性.基尼指数值越大,样本集合的不确定性也就越大,这与熵相似')
        print('算法5.6(CART生成算法)')
        print('输入:训练数据集D,停止计算的条件')
        print('输出:CART决策树')
        print('根据训练数据集,从根结点开始,递归地对每个结点进行一下操作,构建二叉决策树:')
        print('(1) 设结点的训练数据集为D,计算现有特征对该数据集的基尼指数.此时,对每一个特征A,',
            '对其可能取的每个值a,根据样本点对A=a的测试为“是”或“否”将D分割成D1和D2两部分,利用上式计算A=a时的基尼指数.')
        print('(2) 在所有可能的特征A以及它们所有可能的切分点a中,选择基尼指数最小的特征及其对应的切分点',
            '作为最优特征与最优切分点.依最优特征与最优切分点,从现结点生成两个子结点,将训练数据集依特征分配到两个子结点中去.')
        print('(3) 对两个子结点递归地调用(1),(2),直至满足停止条件')
        print('(4) 生成CART决策树')
        print('算法停止计算的条件是结点中的样本个数小于预订阈值,或样本集的基尼指数小于预订阈值',
            '(样本基本属于同一类),或者没有更多特征.')
        print('例5.4 根据表5.1所给训练数据集,应用CART算法生成决策树')
        print('解:首先计算各特征的基尼指数,选择最优特征以及最优切分点.仍采用例5.2的记号,',
            '分别以A1,A2,A3,A4表示年龄、有工作、有自己的房子和信贷情况4个特征,',
            '并以1,2,3表示年龄的值为青年、中年和老年,以1,2表示有工作和有自己的房子的值为是和否,',
            '以1,2,3表示信贷情况的值为非常好、好和一般.')
        print('求特征A1的基尼指数:')
        print('Gini(D,A1=1)=5/15(2*2/5*(1-2/5))+10/15(2*7/10*(1-7/10))=0.44')
        print('Gini(D,A1=2)=0.48')
        print('Gini(D,A1=3)=0.44')
        print('由于Gini(D,A1=1)和Gini(D,A1=3)相等,且最小,所以A1=1和A1=3都可以以选作A1的最优切分点.')
        print('求特征A2和A3的基尼指数:')
        print('Gini(D,A2=1)=0.32')
        print('Gini(D,A3=1)=0.27')
        print('由于A2和A3只有一个切分点,所以它们就是最优切分点.')
        print('求特征A4的基尼指数:')
        print('Gini(D,A4=1)=0.36')
        print('Gini(D,A4=2)=0.47')
        print('Gini(D,A4=3)=0.32')
        print('Gini(D,A4=3)最小,所以A4=3为A4的最优切分点')
        print('在A1,A2,A3,A4几个特征中,Gini(D,A3=1)=0.27最小,所以选择特征A3为最优特征,',
            'A3=1为其最优切分点.于是根结点生成两个子结点,一个是叶结点.对另一个结点继续使用以上方法在',
            'A1,A2,A4中选择最优特征及其最优切分点,结果是A2=1.依次计算得知,所得结点都是叶结点.')
        print('5.5.2 CART的剪枝')
        print('CART剪枝算法从“完全生长”的决策树的底端剪去一些子树,使决策树变小(模型变简单),',
            '从而能够对未知数据有更准确的预测.CART剪枝算法由两步组成;首先从生成算法产生的决策树T0底端开始不断剪枝,',
            '直到T0的根结点,形成一个子树序列{T0,T1,...,Tn};然后通过交叉验证法在独立的验证数据集上对子树序列进行测试,',
            '从中选择最优子树.')
        print('1.剪枝,形成一个子树序列')
        print('  在剪枝过程中,计算子树的损失函数:Ca(T)=C(T)+a|T|')
        print('其中,T为任意子树,C(T)为对训练数据的预测误差(如基尼指数),|T|为子树的叶结点个数,',
            'a>=0为参数,Ca(T)为参数是a时的子树T的整体损失,参数a权衡训练数据的拟合程度与模型的复杂度')
        print('对固定的a,一定存在使损失函数Ca(T)最小的子树,将其表示为Ta.',
            'Ta在损失函数Ca(T)最小的意义下是最优的.容易验证这样的最优子树是唯一的.',
            '当a大的时候,最优子树Ta偏小;当a小的时候,最优子树Ta偏大.极端情况,当a=0时,',
            '整体树是最优的.当a->∞时,根结点组成的单结点树是最优的.')
        print('可以用递归的方法对树进行剪枝.将a从小增大,0=a0<a1<...<an<+∞,',
            '产生一系列的区间[ai,ai+1),i=0,1,...,n;剪枝得到的子树序列对应着区间a∈[ai,ai+1),',
            'i=0,1,...,n的最优子序列{T0,T1,...,Tn},序列中的子树是嵌套的.')
        print('具体地,从整体树T0开始剪枝.对T0的任意内部结点t,以t为单结点树的损失函数是:Ca(t)=C(t)+a',
            '以t为根结点的子树Tt的损失函数是Ca(Tt)=C(Tt)+a|Tt|')
        print('当a=0及a充分小时,有不等式Ca(Tt)<Ca(t)')
        print('当a增大时,在某一a有Ca(Tt)=Ca(t)')
        print('当a再增大时,不等式Ca(Tt)<Ca(t)反向.只要a=(C(t)-C(Tt))/(|Tt|-1),',
            'Tt与t有相同的损失函数值,而t的结点少,因此t比Tt更可取,对Tt进行剪枝.')
        print('为此,对T0中每一内部结点t,计算g(t)=(C(t)-C(Tt))/(|Tt|-1)')
        print('表示剪枝后整体损失函数减少的程度.在T0中剪去g(t)最小的Tt,将得到的子树作为T1,',
            '同时将最小的g(t)设为a1.T1为区间[a1,a2)的最优子树.')
        print('如此剪枝下去,直至得到跟结点.在这一过程中,不断增加a的值,产生新的区间')
        print('2. 在剪枝得到的子树序列T0,T1,...,Tn中通过交叉验证选取最优子树Ta')
        print('具体地,利用独立的验证数据集,测试子树序列T0,T1,...,Tn中各棵子树的平方误差或基尼指数.',
            '平方误差或基尼指数.平方误差或基尼指数最小的决策树被认为是最优的决策树.在子树序列中,',
            '每棵子树T1,T2,...,Tn都对应于一个参数a1,a2,...,an.',
            '所以,当最优子树Tk确定时,对应的ak也确定了,即得到最优决策树Ta')
        print('算法5.7 (CART剪枝算法)')
        print('输入：CART算法生成的决策树T0;')
        print('输出：最优决策树Ta.')
        print('(1) 设k=0,T=T0')
        print('(2) 设a=+∞')
        print('(3) 自下而上地对各内部结点t计算C(Tt),|Tt|以及:')
        print('  g(t)=(C(t)-C(Tt))/(|Tt|-1), a=min(a, g(t))')
        print('  这里,Tt表示以t为根结点的子树,C(Tt)是对训练数据的预测误差,|Tt|是Tt的叶结点个数.')
        print('(4) 自上而下地访问内部结点t,如果有g(t)=a,进行剪枝,并对叶结点t以多数表决法确定其类,得到树T')
        print('(5) 设k=k+1,ak=a,Tk=T.')
        print('(6) 如果T不是由根结点单独构成的树,则回到步骤(4)')
        print('(7) 采用交叉验证法在子树序列T0,T1,...,Tn中选取最优子树Ta')
        print('本章概要')
        print('1.分类决策树模型是表示基于特征对实例进行分类的树形结构.',
            '决策树可以转换成一个if-then规则的集合,也可以看做是定义在特征空间划分上的类条件概率分布')
        print('2.决策树学习旨在构建一个与训练数据拟合很好,并且复杂度小的决策树.',
            '因为从可能的决策树中直接选取最优决策树是NP完全问题.现实中采用启发式学习次优的决策树.')
        print('  决策树学习算法包括3部分：特征选择、树的生成和树的剪枝.常用的算法有ID3,C4.5和CART')
        print('3.特征选择的目的在于选取对训练数据能够分类的特征.特征选择的关键是其准则.常用准则如下:')
        print('(1) 样本集合D对特征A的信息增益(ID3)')
        print('(2) 样本集合D对特征A的信息增益比(C4.5)')
        print('(3) 样本集合D的基尼指数(CART)')
        print('4.决策树的生成.通常使用信息增益最大、信息增益比最大或基尼指数最小作为特征选择的准则.',
            '决策树的生成往往通过计算信息增益或其他指标,从根结点开始,递归地产生决策树.',
            '这相当于用信息增益或其他准则不断地选取局部最优的特征,或将训练集分割能够基本正确分类的子集')
        print('5.决策树的剪枝.由于生成的决策树存在过拟合问题,需要对它进行剪枝,',
            '以简化学到的决策树.决策树的剪枝,往往从已生成的树上剪掉一些叶结点或叶结点以上的子树,',
            '并将其父结点或根结点作为新的叶结点,从而简化生成的决策树')

chapter5 = Chapter5()

def main():
    chapter5.note()

if __name__ == '__main__':
    main()