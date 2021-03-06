
class Chapter10:
    """
    第10章 隐马尔可夫模型的基本概念
    """
    def __init__(self):
        """
        第10章 隐马尔可夫模型的基本概念
        """
        pass

    def note(self):
        """
        chapter10 note
        """
        print('第10章 隐马尔可夫模型的基本概念')
        print('隐马尔可夫模型(Hidden Markov Model, HMM)是可用于标注问题的统计学习模型,',
            '描述由隐藏的马尔可夫链随机生成观测序列的过程，属于生成模型.',
            '本章介绍隐马尔可夫模型基本概念,概率计算算法、学习算法、预测算法.',
            '应用：语音识别、自然语言处理、生物信息、模式识别')
        print('第10章 隐马尔可夫模型的基本概念')
        print('定义10.1 (隐马尔可夫模型) 隐马尔可夫模型是关于时序的概率模型,描述一个隐藏的马尔可夫链',
            '随机生成不可观测的状态随机序列,再由各个状态生成一个观测而产生观测随机序列的过程.',
            '隐藏的马尔可夫链随机生成的状态序列,每个状态生成一个观测,而由此产生的观测的随机序列,',
            '序列的每一个位置又可以看做是一个时刻.')
        print('HMM由初始概率分布、状态转移概率分布以及观测概率分布确定.HMM的形式定义如下:')
        print('设Q是所有可能的状态的集合,V是所有可能的观测的集合.')
        print('   Q={q1,q1,...,qN},V={v1,v2,...,vM}')
        print('其中,N是可能的状态数,M是可能的观测数.')
        print('I是长度为T的状态序列,O是对应的观测序列.')
        print('   I={i1,i1,...,iN},V={i1,i2,...,iM}')
        print('A是状态转移概率矩阵：A=[aij]N*N')
        print('其中,aij=P(i+1=q|it=qi),i=1,2,...,N; j=1,2,...,N')
        print('是在时刻t处于状态qi的条件下在时刻t+1转移到状态qj的概率')
        print('B是观测概率矩阵:B=[bj(k)]N*M')
        print('其中,bj(k)=P(ot=vk|it=qj),k=1,2,...,M;j=1,2,...,N')
        print('  是在时刻t处于状态qj的条件下生成观测vk的概率')
        print('pi是初始状态的概率向量:pi=(pii).其中,pii=P(i1=qi),i=1,2,...,N',
            '是时刻t=1处于状态qi的概率.')
        print('隐马尔可夫模型由初始状态概率向量pi,状态转移概率矩阵A和观测概率矩阵B决定.pi和A决定状态序列,',
            'B决定观测序列.因此,隐马尔可夫模型l可以用三元符号表示,即l=(A,B,pi)',
            'A,B,pi称为HMM的三要素')
        print('状态转移概率矩阵A与初始状态概率向量pi确定了隐藏的马尔可夫链,生成不可观测的状态序列.',
            '观测概率矩阵B确定了如何从状态生成观测,与状态序列综合确定了如何产生观测序列.')
        print('HMM的两个基本假设:')
        print('(1) 齐次马尔可夫性假设,即假设隐藏的马尔可夫链在任意时刻t的状态只依赖于其前一时刻的状态,',
            '与其他时刻的状态及观测无关,也与时刻t无关.')
        print('    P(it|it-1,ot-1,...,i1,o1)=P(it|it-1),t=1,2,...,T')
        print('(2) 观测独立性假设,即假设任意时刻的观测只依赖于该时刻的马尔可夫链的状态,',
            '与其他观测及状态无关.')
        print('    P(ot|iT,oT,iT-1,oT-1,...,it+1,ot+1,it-1,ot-1,...,i1,o1)=P(it|it)')
        print('HMM可以用于标注,这时状态对应着标记.标注问题是给定观测的序列预测其对应的标记序列.',
            '可以假设标注问题的数据是由HMM模型生成的.这样可以利用隐马尔可夫模型的学习与观测算法进行标注.')
        print('例10.1 (盒子和球模型) 假设有4个盒子,每个盒子里面都装有红白两种颜色的球,',
            '盒子面的红白球数由表列出')
        print('盒  子 1 2 3 4')
        print('红球数 5 3 6 8')
        print('白球数 5 7 4 2')
        print('按照下面的方法抽球,产生一个球的颜色的观测序列：开始,从4个盒子里以等概率随机选取1个盒子,',
            '从这个盒子里随机抽出一个球,记录其颜色后,放回;然后,从当前盒子随机转移到下一个盒子,',
            '规则是:如果当前盒子是盒子1,那么下一个盒子一定是盒子2,如果当前是盒子2或3,',
            '那么分别以概率0.4和0.6转移到左边或右边的盒子,如果当前是盒子4,',
            '那么各以0.5的概率停留在盒子4或转移到盒子3;确定转移的盒子后,再从这个盒子里随机抽出一个球,',
            '记录其颜色,放回;如此下去,重复进行5次,得到一个球的颜色的观测序列:',
            'O={红,红,白,白,红}')
        print('在这个过程中,观察者只能观测到球的颜色的序列,观测不到球是从哪个盒子取出的,即观测不到盒子的序列.')
        print('在这个例子中有两个随机序列,一个是盒子的序列(状态序列),一个是球的颜色的观测序列.',
            '前者是隐藏的,只有后者是可观测的.这是一个HMM的例子,根据所给条件,可以明确状态集合,',
            '观测集合,序列长度以及模型的三要素.')
        print('盒子对应状态，状态的集合是:Q={盒子1,盒子2,盒子3,盒子4},N=4')   
        print('球的颜色对应观测.观测的集合是:V={红,白},M=2')
        print('状态序列和观测序列长度T=5.初始概率分布为:pi=(0.25,0.25,0.25,0.25)^T')
        print('状态转移概率分布为:[[0,1,0,0],[0.4,0,0.6,0],[0,0.4,0,0.6],[0,0,0.5,0.5]]')   
        print('观测概率分布为:[[0.5,0.5],[0.3,0.7],[0.6,0.4],[0.8,0.2]]')
        print('10.1.2 观测序列的生成过程')
        print('根据隐马尔可夫模型定义,可以将一个长度为T的观测序列O=(o1,o2,...,oT)的生成过程描述如下:')
        print('算法10.1 (观测序列的生成)')
        print('输入:隐马尔可夫模型lambda=(A,B,pi),观测序列长度T;')
        print('输出:观测序列O=(o1,o2,...,oT)')
        print('(1) 按照初始状态分布pi产生状态i1')
        print('(2) 令t=1')
        print('(3) 按照状态it的观测概率分布bit(k)生成ot')
        print('(4) 按照状态it的状态转移概率分布{ait,ai(t+1)}产生状态i(t+1),i(t+1)=1,2,...,N')
        print('(5) 令t=t+1;如果t<T,转步(3);否则,终止.')
        print('10.1.3 隐马尔可夫模型的3个基本问题')
        print('隐马尔可夫模型有3个基本问题:')
        print('(1) 概率计算问题.给定模型l=(A,B,pi)和观测序列O=(o1,o2,...,oT),',
            '计算在模型l下观测序列O出现的概率P(O|t)')
        print('(2) 学习问题.已知观测序列O=(o1,o2,...,oT),估计模型t=(A,B,pi)参数,',
            '使得在该模型下观测序列概率P(O|l)最大.即用极大似然估计的方法估计参数.')
        print('(3) 预测问题,也称为解码问题.已知模型l=(A,B,pi)和观测序列O=(o1,o2,...,oT),',
            '求对给定观测序列条件概率P(I|O)最大的状态序列I=(i1,i2,...,iT).',
            '即给定观测序列,求最有可能的对应的状态序列.')
        print('10.2.1 直接计算法')
        print('给定模型l=(A,B,pi)和观测序列O=(o1,o2,...,oT),计算观测序列O出现的概率P(O|l).',
            '最直接的方法是按概率公式直接计算.通过列举所有可能的长度为T的状态序列I=(i1,i2,...,iT),',
            '求各个状态序列I与观测序列O=(o1,o2,...,oT)的联合概率P(O,I|l),然后对所有可能的状态序列求和,',
            '得到P(O|l)')
        print('状态序列I=(i1,i2,...,iT)的概率是P(I|l)=pi(i1)a(i1)(i2)a(i2)(i3)...a(iT-1)a(T)')
        print('对固定的状态序列I=(i1,i2,...,iT),观测序列O=(o1,o2,...,oT)的概率是P(O|I,l),',
            'P(O|I,l)=bi1(o1)bi2(o2)')
        print('O和I同时出现的联合概率为:P(O,I|l)=P(O|I,l)P(I|l)')
        print('然后,对所有可能的状态序列I求和,得到观测序列O的概率P(O|l),即',
            'P(O|l)=∑P(O|I,l)P(I|l)')
        print('但是,利用上述公式计算量很大,是O(TN^T)阶,这种算法不可行')
        print('接下来说明观测序列概率P(O|l)的有效算法:前向-后向算法(forward-backward algorithm)')
        print('10.2.2 前向算法')
        print('定义10.2（前向概率）给定隐马尔可夫模型l,定义到时刻t部分观测序列为o1,o2,...,ot且状态为qi的概率为前向概率,记作',
            'at(i)=P(o1,o2,...,ot,it=qi|l)')
        print('可以递推地求前向概率ai(i)及观测序列概率P(O|l)')
        print('算法10.2（观测序列概率的前向算法）')
        print('输入:隐马尔可夫模型l，观测序列O;')
        print('输出:观测序列概率P(O|l)')
        print('(1) 初值a1(i)=piibi(o1),i=1,2,...,N')
        print('(2) 递推 对t=1,2,...,T-1, ai+1(i)=[∑at(j)aji]bi(ot+1),i=1,2,...,N')
        print('(3) 终止 P(O|l)=∑aT(i)')
        print('前向算法,步骤(1)初始化前向概率,是初始时刻的状态i1=qi和观测o1的联合概率.',
            '步骤(2)是前向概率的递推公式,计算到时刻t+1部分观测序列为o1,o2,...,ot,ot+1且在时刻t+1',
            '处于状态qi的前向概率,既然at(j)是到时刻t观测到o1,o2,...,ot并在时刻t处于状态qj的前向概率,',
            '那么乘积at(j)aji就是到时刻t观测到o1,o2,...,ot并在时刻t处于状态qj,',
            '而在时刻t+1到达状态qi的联合概率.')
        print('对这个乘积在时刻t的所有可能的N个状态qj求和,其结果就是到时刻t观测为o1,o2,...,ot',
            '并在时刻t+1处于状态qi的联合概率.')
        print('前向算法实际是基于“状态序列的路径结构”递推计算P(O|l)的算法.',
            '前向算法高效的关键是局部计算前向概率,然后利用路径结构将前向概率“递推”到全局,',
            '得到P(O|l).具体地,在时刻t=1,计算a1(i)的N个值(i=1,2,...,N);在各个时刻t=1,2,...,T-1,',
            '计算at+1(i)的N个值(i=1,2,...,N),而且每个at+1(i)的计算利用前一时刻N个at(j).',
            '减少计算量的原因在于每一次计算直接引用前一个时刻的计算结果,避免重复计算.这样,',
            '利用前向概率计算P(O|l)的计算量是O(N^2T)阶的,而不是直接计算的O(TN^T)阶')
        print('例10.2 考虑盒子和球模型l=(A,B,pi),状态集合Q={1,2,3},观测集合V={红,白},')
        print('A=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]], B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]],pi=(0.2,0.4,0.4)^T')
        print('设T=3,O=(红,白,红),试用前向算法计算P(O|l)')
        print('解:按照算法10.2')
        print('(1) 计算初值 a1(l)=p1b1(o1)=0.10; a1(2)=p2b2(o1)=0.16; a1(3)=p3b3(o1)=0.28')
        print('(2) 递推计算')
        print('  a2(1)=[∑a1(i)ai1]b1(o2)=0.154*0.5=0.077')
        print('  a2(2)=[∑a1(i)ai2]b2(o2)=0.184*0.6=0.1104')
        print('  a2(3)=[∑a1(i)ai3]b3(o2)=0.202*0.3=0.0606')
        print('  a3(1)=[∑a2(i)ai1]b1(o3)=0.04187')
        print('  a3(2)=[∑a2(i)ai2]b2(o3)=0.03551')
        print('  a3(3)=[∑a2(i)ai3]b3(o3)=0.05284')
        print('(3) 终止 P(O|l)=∑a3(i)=0.13022')
        print('10.2.3 后向算法')
        print('定义10.3（后向概率）给定马尔可夫模型l,定义在时刻t状态为qi的条件下,从t+1到T的部分观测序列为',
            'ot+1,ot+2,...,oT的概率为后向概率,记作',
            'bt(i)=P(ot+1,ot+2,...,oT|it=qi,l)')
        print('可以用递推的方法求得后向概率bt(i)及观测序列概率P(O|l).')
        print('算法10.3（观测序列概率的后向算法）')
        print('输入：隐马尔可夫模型l,观测序列O;')
        print('输出：观测序列概率P(O|l)')
        print('(1) bT(i)=1,i=1,2,...,N')
        print('(2) 对t=T-1,T-2,...,1 bt(i)=∑aijbj(at+1)bt+1(j),i=1,2,...,N')
        print('(3) P(O|l)=∑pibi(o1)')
        print('步骤(1)初始化后向概率,对最终时刻的所有状态qi规定bT(i)=1.步骤(2)是后向概率的递推公式.',
            '为了计算在时刻t状态为qi条件下时刻t+1之后的观测序列为ot+1,ot+2,...,oT的后向概率bt(i),',
            '只需考虑在时刻t+1所有可能的N个状态qj的转移概率(即aij项),以及在此状态下的观测ot+1的观测概率,',
            '即bj(ot+1)项),然后考虑状态qj之后的观测序列的后向概率(即bt+1(j)项).',
            '步骤(3)求P(O|l)的思路与步骤(2)一致,只是初始概率pi代替转移概率.')
        print('利用前向概率和后向概率的定义可以将观测序列概率P(O|l)统一写成',
            'P(O|l)=∑∑at(i)aijbj(ot+1)bt+1(j),t=1,2,..,T-1')
        print('此式当t=1和t=T-1时分别为式')
        print('10.2.4 一些概率与期望值的计算')
        print('利用前向概率和后向概率,可以得到关于单个状态和两个状态概率的计算公式.')
        print('1.给定模型l和观测O,在时刻t处于状态qi的概率.记yt(i)=P(it=qi|O,l)可以通过前向后向概率计算.事实上,',
            'yt(i)=P(it=qi|O,l)=P(it=qi,O|l)/P(O|l)')
        print('由前向概率at(i)和后向概率bt(i)定义可知:at(i)bt(i)=P(it=qi,O|l)')
        print('于是得到:yt(i)=at(i)bt(i)/P(O|l)=P(it=qi,O|l)')
        print('2.给定模型l和观测O,在时刻他处于状态qi且在时刻t+1处于状态qj的概率.记:',
            'fi(i,j)=P(it=qi,it+1=qj|O,l)')
        print('可以通过前向后向概率计算:',
            'fi(i,j)=P(it=qi,it+1=qj,O|l)/P(O|l)=P(it=qi,it+1=qj,O|l)/∑∑P(it=qi,it+1=qj,O|l)',
            '而P(it=qi,it+1=qj,O|l)=at(i)aijbj(ot+1)bt+1(j)')
        print('所以 ft(i,j)=at(i)aijbj(ot+1)bt+1(j)/∑∑at(i)')
        print('3.将yt(i)和ft(i,j)对各个时刻t求和,可以得到一些有用的期望值:')
        print('(1) 在观测O下状态i出现的期望值∑yt(i)')
        print('(2) 在观测O下由状态i转移的期望值∑yt(i)')
        print('(3) 在观测O下由状态i转移到状态j的期望值∑ft(i,j)')
        print('10.3 学习算法')
        print('HMM的学习,根据训练数据是包括观测序列和对应的状态序列还只是有观测序列,可以分别由监督学习与非监督学习实现.',
            '先介绍监督学习算法,后介绍非监督学习算法-Baum-Welch算法(也就是EM算法)')
        print('10.3.1 监督学习方法')
        print('假设已给训练数据包含S个长度相同的观测序列和对应的状态序列{(O1,I1),(O2,I2),...,(Os,Is)}',
            '那么可以利用极大似然估计法来估计HMM模型的参数.具体方法如下：')
        print('1.转移概率aij的估计')
        print('设样本中时刻t处于i时刻t+1转移到状态j的频数为Aij,那么状态转移概率aij的估计是',
            'aij=Aij/∑Aij,i=1,2,...,N;j=1,2,...,N')
        print('2.观测概率bj(k)的估计')
        print('设样本中状态j并观测为k的频数是Bjk,那么状态为j的观测为k的概率bj(k)的估计是',
            'bj(k)=Bjk/∑Bjk,j=1,2,...,N;k=1,2,...,M')
        print('3.初始状态概率pi的估计pi为S个样本中初始状态为qi的频率')
        # !由于监督学习需要使用训练数据,而人工标注训练数据往往代价很高,有时就会利用非监督学习的方法
        print('由于监督学习需要使用训练数据,而人工标注训练数据往往代价很高,有时就会利用非监督学习的方法')
        print('10.3.2 Baum-Welch算法')
        print('假设给定训练数据只包含S个长度为T的观测序列{O1,O2,...,Os}而没有对应的状态序列,',
            '目标是学习隐马尔可夫模型l=(A,B,pi)的参数.将观测序列数据看作观测数据O,状态序列看作不可观测的隐数据I,',
            '那么隐马尔可夫模型事实上是一个含有隐变量的概率模型',
            'P(O|l)=∑P(O|I,l)P(I|l)它的参数学习可以由EM算法实现.')
        print('1.确定完全数据的对数似然函数')
        print('所有观测数据写成O=(o1,o2,...,oT),所有隐数据写成I=(i1,i2,...,iT),完全数据是',
            '(O,I)=(o1,o2,...,oT,i1,i2,...,iT).完全数据的对数似然函数是logP(O,I|l).')
        print('2.EM算法的E步:求Q函数Q(l,lbar):Q(l,lbar)=∑logP(O,I|l)P(O,I|lbar)')
        print('其中,lbar是HMM参数的当前估计值,l是要极大化的HMM模型参数')
        print('  P(O,I|l)=pi1bi1(o1)ai1ai2bi2(o2)...aiT-1biT(oT)')
        print('于是函数Q(l,lbar)可以写成:Q(l,lbar)=∑logpi1P(O,I|lbar)+',
            '∑(∑logai,i+1)P(O,I|l)+∑(∑logbit(ot))P(O,I|lbar)')
        print('式中求和都是对所有训练数据的序列总长度T进行的.')
        print('3.EM算法的M步,极大化Q函数Q(l,lbar)求模型参数A,B,pi')
        print('由于要极大化的参数在式中单独地出现在3个项中,所以只需对各个项分别极大化.')
        print('(1) 式的第1项可以写成:∑logpi0P(O,I|lbar)=∑logpiP(O,i1=i|lbar)')
        print('注意到pi满足约束条件∑pi=1,利用拉格朗日乘子法,写出拉格朗日函数:',
            '∑logpiP(O,i1=i|lbar)+y(∑pi-1)')
        print('对其求偏导数并令结果为0得:P(O,i1=i|lbar)+ypi=0,对i求和得到y=-P(O|lbar)',
            '然后得到pi=P(O,i1=i|lbar)/P(O|lbar)')
        print('(2) 类似第1项,第2项可以应用具有约束条件∑aij=1的拉格朗日乘子法可以求出aij')
        print('(3) 式的第3项为,同样用拉格朗日乘子法,约束条件是∑bj(k)=1.注意,只有在ot=vk时bj(ot)',
            '对bj(k)的偏导数才不为0,以I(ot=vk)表示.求得bj(k)')
        print('10.3.3 Baum-Welch模型参数估计公式')
        print('将上式中的各概率分别用yt(i),ft(i,j)表示,则可将相应的公式写成：',
            'aij=∑ft(i,j)/∑yt(i), bj(k)=∑yt(j)/=∑yt(j), pi=y1(i)')
        print('是EM算法在HMM学习中的具体实现,由Baum和Welch提出.')
        print('算法10.4 (Baum-Welch算法)')
        print('输入:观测数据O=(o1,o2,...,oT)')
        print('输出:HMM模型参数')
        print('(1) 初始化')
        print('  对n=0,选取aij(0),bj(k)(0),pi(0),得到模型l(0),(A(0),B(0),pi(0))')
        print('(2) 递推.对n=1,2,...,')
        print('  aij(n+1)=∑ft(i,j)/∑yt(i)')
        print('  bj(k)(n+1)=∑yt(j)/∑yt(j)')
        print('  pi(n+1)=y1(i)')
        print('右端各值按观测O=(o1,o2,...,oT)和模型l(n)=(A(n),B(n),pi(n))计算.',
            '式中yt(i),ft(i,j)')
        print('(3) 终止.得到模型参数l(n+1)=(A(n+1),B(n+1),pi(n+1))')
        print('10.4 预测算法')
        print('HMM预测的两种算法：近似算法与维特比算法')
        print('10.4.1 近似算法')
        print('近似算法的想法是,在每个时刻t选择在该时刻最有可能出现的状态it*,从而得到一个状态序列I*=(i1*,i2*,...,iT*)',
            '将它作为预测的结果')
        print('给定隐马尔可夫模型l和观测序列O,在时刻t处于状态qi的概率yt(i)是:',
            'yt(i)=at(i)bt(i)/P(O|l)=at(i)bt(i)/∑at(j)bt(j)')
        print('在每一时刻t最有可能的状态it*是it*=argmax[yt(i)],t=1,2,...,T',
            '从而得到状态序列I*=(i1*,i2*,...,iT*)')
        print('近似算法的优点是计算简单,其缺点是不能保证预测的状态序列整体是最有可能的状态序列,',
            '因为预测的状态序列可能有实际不发生的部分.事实上,上述方法得到的状态序列中有可能存在转移概率为0的相邻状态,',
            '即对某些i,j,aij=0时.尽管如此,近似算法仍然是有用的')
        print('10.4.2 维特比算法')
        print('维特比算法实际是比用动态规划解HMM预测问题,用DP求概率最大路径(最优路径)',
            '这时一条路径对应着一个状态序列.')
        print('最优路径具有这样的特性:如果最优路径在时刻t通过结点it*,那么这一路径从结点it*到终点iT*的部分路径,',
            '对于从it*到iT*的所有可能的部分路径来说,必须是最优的.因为假如不是这样,',
            '那么存在另一条更好的部分路径存在,根据这一矛盾,',
            '只需从时刻t=1开始,递推地计算在时刻t状态为i的各部分路径的最大概率,',
            '直至得到时刻t=T状态为i的各条路径的最大概率.')
        print('之后,为了找出最优路径的各个结点,从终结点iT*开始,由后向前逐步求得结点iT-1*,...,i1*,',
            '得到最优路径I*=(i1*,i2*,...,iT*).这就是维特比算法.')
        print('首先导入两个变量d和p.定义时刻t状态为i的所有单个路径(i1,i2,...,it)中概率最大值为:',
            'dt(i)=maxP(it=i,it-1,...,i1,ot,...,o1|l),i=1,2,...,N')
        print('由定义可得变量d的递推公式：',
            'dt+1(i)=maxP(it+1=i,it,...,i1,ot+1,...,o1|l)=max[dt(j)aji]bi(ot+1),i=1,2,...,N;t=1,2,...,T-1')
        print('定义在时刻t状态为i的所有单个路径(i1,i2,...,i-1,i)中概率最大的路径的第t-1个结点为:',
            'pt(i)=argmax[dt-1(j)aji],i=1,2,...,N')
        print('算法10.5（维特比算法）')
        print('输入:模型l=(A,B,pi)和观测O=(o1,o2,...,oT)')
        print('输出:最优路径I*=(i1*,i2*,...,iT*)')
        print('(1) 初始化 d1(i)=pibi(o1),i=1,2,...,N p1(i)=0,i=1,2,...,N')
        print('(2) 递推.对t=2,3,...,T')
        print('  dt(i)=max[dt-1(j)aji]bi(ot),i=1,2,...,N')
        print('  pt(i)=argmax[dt-1(j)aji],i=1,2,...,N')
        print('(3) 终止 P*=maxdT(i) iT*=argmax[dT(i)]')
        print('(4) 最优路径回溯.对t=T-1,T-2,...,1  it*=pt+1(it+1*)')
        print('   求得最优路径I*=(i1*,i2*,...,iT*)')
        print('例10.3 A=[[0.5,0.2,0.3],[0.3,0.5,0.2],[0.2,0.3,0.5]], B=[[0.5,0.5],[0.4,0.6],[0.7,0.3]],pi=(0.2,0.4,0.4)^T')
        print('  已知观测序列O=(红,白,红),试求最优状态序列,即最优路径I*=(i1*,i2*,i3*)')
        print('解:如图10.4所示,要在所有可能的路径中选择一条最优路径,按照以下步骤处理：')
        print('(1) 初始化.在t=1时,对每一个状态i,i=1,2,3,求状态为i观测o1为红的概率,',
            '记此概率为d1(i),则d1(i)=pibi(o1)=pibi(红),i=1,2,3')
        print('代入实际数据 d1(1)=0.10, d1(2)=0.16, d1(3)=0.28')
        print('记p1(i)=0,i=1,2,3. ')
        print('(2) 在t=2时,对每个状态i,i=1,2,3,求在t=1时状态为j观测为红并在t=2时状态为i观测o2为白的路径的最大概率,',
            '记此最大概率为d2(i),则d2(i)=max[d1(j)aji]bi(o2)')
        print('同时,对每个状态i,i=1,2,3,记录概率最大路径的前一个状态j:p2(i)=argmax[d1(j)aji],i=1,2,3')
        print('计算:d2(1)=max[d1(j)aj1]b1(o2), p2(1)=3, d2(2)=0.0504, p2(2)=3, d2(3)=0.042, p2(3)=3')
        print('同样,在t=3时,d3(i)=max[d2(j)aji]bi(o3) p3(i)=argmax[d2(j)aji]',
            'd3(1)=0.00756, p3(1)=2  d3(2)=0.01008, p3(2)=2 d3(3)=0.0147, p3(3)=3')
        print('(3) 以P*表示最优路径的概率,则 P*=maxd3(i)=0.0147')
        print('  最优路径的终点i3*:i3*=argmax[d3(i)]=3')
        print('(4) 由最优路径的终点i3*,逆向找到i2*,i1*:')
        print('  在t=2时,i2*=p3(i3*)=p3(3)=3')
        print('  在t=1时,i1*=p2(i2*)=p2(3)=3')
        print('于是求得最优路径,即最优状态序列I*=(i1*,i2*,i3*)=(3,3,3)')
        print('本章概要')
        print('1.隐马尔可夫模型是关于时序的概率模型,描述一个隐藏的马尔可夫链随机生成的不可观测的状态的序列,',
            '再由各个状态随机生成一个观测而产生观测的序列的过程.')
        print('  隐马尔可夫模型由初始状态概率向量pi,状态转移概率矩阵A和观测概率矩阵B决定.',
            '因此,隐马尔可夫模型可以写成l=(A,B,pi)')
        print('  隐马尔可夫模型是一个生成模型,表示状态序列和观测序列的联合分布,但是状态序列是隐藏的,不可观测的')
        print('  隐马尔可夫模型可以用于标注,这时状态对应着标记.标注问题是给定观测序列预测其对应的标记序列.')
        print('2.概率计算问题.给定模型l=(A,B,pi)和观测序列O=(o1,o2,...,oT),',
            '计算在模型l下观测序列O出现的概率P(O|l).前向-后向算法是通过递推地计算',
            '前向-后向概率可以高效地进行隐马尔可夫模型的概率计算')
        print('3.学习问题.已知观测序列O=(o1,o2,...,oT),估计模型l=(A,B,pi)参数,',
            '使得在该模型下观测序列概率P(O|l)最大,即用极大似然估计的方法估计参数.',
            'Baum-Welch算法,也就是EM算法可以高效地对HMM进行训练.它是一种非监督学习算法')
        print('4.预测问题.已知模型l=(A,B,pi)和观测序列O=(o1,o2,...,oT),',
            '求对给定观测序列条件概率P(I|O)最大的状态序列I=(i1,i2,...,iT).',
            '维特比算法应用动态规划高效地求解最优路径,即概率最大的状态序列.')

chapter10 = Chapter10()

def main():
    chapter10.note()

if __name__ == '__main__':
    main()