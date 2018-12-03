
class Chapter11:
    """
    第11章 条件随机场
    """
    def __init__(self):
        """
        第11章 条件随机场
        """
        pass

    def note(self):
        """
        chapter11 note
        """
        print('第11章 条件随机场')
        print('条件随机场(conditional random field, CRF)是给定一组输入随机变量下',
            '另一组输出随机变量的条件概率分布模型,其特点是假设输出随机变量构成马尔可夫随机场.',
            '条件随机场可以用于不同的预测问题,标注问题的应用.因此主要讲述线性链条件随机场,',
            '问题变成了由输入序列对输出序列预测的判别模型,形式为对数线性模型,',
            '其学习方法通常是极大似然估计或正则化的极大似然估计.')
        print('线性链条件随机场应用于标注问题是由Lafferty等人于2001年提出的.')
        print('11.1 概率无向图模型')
        print('概率无向图模型,又称为马尔可夫随机场,是一个可以由无向图表示的联合概率分布.')
        print('11.1.1 模型定义')
        print('图(graph)是由结点(node)及连接结点的边(edge)组成的集合.结点和边分别记作v和e,',
            '结点和边的集合分别记作V和E,图记作G=(V,E).无向图是指边没有方向的图.')
        print('概率图模型是由图表示的概率分布.设有联合概率分布P(Y),Y∈Y是一组随机变量.',
            '由无向图G=(V,E)表示概率分布P(Y),即在图G中,结点v∈V表示一个随机变量Yv,',
            'Y=(Yv)v∈V;边e∈E表示随机变量之间的概率依赖关系.')
        print('给定一个联合概率分布P(Y)和表示它的无向图G.首先定义无向图表示的随机变量之间存在成对的马尔可夫性,',
            '局部马尔可夫性和全局马尔可夫性')
        print('成对马尔可夫性:设u和v是无向图G中任意两个没有边连接的结点,结点u和v分别对应随机变量Yu和Yv.',
            '其他所有结点为O,对应的随机变量组是Yo.成对马尔可夫性是指给定随机变量组Yo.成对')
        print('成对马尔可夫性:设u和v是无向图G中任意两个没有边连接的结点,',
            '结点u和v分别对应随机变量Yu和Yv.其他所有结点为O,对应的随机变量组是Yo.',
            '成对马尔可夫性是指给定随机变量组Yo的条件下随机变量Yu和Yv是条件独立的,即',
            'P(Yv,Yo|Yw)=P(Yv|Yw)P(Yo|Yw)')
        print('在P(Yo|Yw)>0时,等价地, P(Yv|Yw)=P(Yv|Yw,Yo)')
        print('全局马尔可夫性:设结点集合A,B是在无向图G中被结点集合C分开的任意结点集合,',
            '结点集合A,B和C所对应的随机变量组分别是YA,YB和YC.全局马尔可夫性是指给定随机变量组YC条件下随机变量组',
            'YA和YB是条件独立的,即P(YA,YB|YC)=P(YA|YC)P(YB|YC)')
        print('上述成对的、局部的、全局的马尔可夫性定义是等价的.')
        print('定义11.1 (团与最大团)无向图G中任何两个结点均有边连接的结点子集称为团(clique).',
            '若C是无向图G的一个团,并且不能再加进任何一个G的结点使其成为一个更大的团,则称此C为最大团.')
        print('图11.3表示由4个结点组成的无向图,图中由2个结点组成的团有5个:{Y1,Y2},',
            '{Y2,Y3},{Y3,Y4},{Y4,Y2},{Y1,Y3}.有两个最大团：{Y1,Y2,Y3}和{Y2,Y3,Y4}.',
            '而{Y1,Y2,Y3,Y4}不是一个团,因为Y1和Y4没有边连接')
        print('将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作,',
            '称为概率无向图模型的因子分解(factorization)')
        print('给定概率无向图模型,设其无向图为G,C为G上的最大团,',
            'YC表示C对应的随机变量.那么概率无向图模型的联合概率分布P(Y)可写作图中所有最大团C上的函数fC(YC)的乘积形式,',
            'P(Y)=1/Z∏fC(YC)')
        print('其中,Z是规范化因子,由式Z=∑∏fC(YC)给出.规范化因子保证P(Y)构成一个概率分布.',
            '函数fC(YC)称为势函数.这里要求势函数fC(YC)是严格正的,通常定义为指数函数:',
            'fC(YC)=exp{-E(YC)}')
        print('概率无向图模型的因子分解由下述定理来保证.')
        print('定理11.1 (Hammersley-Clifford定理)概率无向图模型的联合概率分布P(Y)可以表示为如下形式:',
            'P(Y)=1/Z∏fC(YC), Z=∑∏fC(YC)')
        print('其中,C是无向图的最大团,YC是C的结点对应的随机变量,fC(YC)是C上定义的严格正函数,',
            '乘积是在无向图所有的最大团上进行的')
        print('11.2 条件随机场的定义与形式')
        print('11.2.1 条件随机场的定义')
        print('条件随机场(conditional random field) 是给定随机变量X条件下,随机变量Y的马尔可夫随机场.',
            '主要介绍定义在线性链上的特殊的条件随机场,称为线性链条件随机场.',
            '线性链条件随机场可以用于标注问题.这时,在条件概率模型P(Y|X)中,',
            'Y是输出变量,表示标记序列,X是输入变量,表示需要标注的观测序列.',
            '也把标记序列称为状态序列(参见HMM).学习时,利用训练数据集通过极大似然估计或正则化',
            '的极大似然估计得到条件概率模型P(Y|X);预测时,对于给定的输入序列x,',
            '求出条件概率P(y|x)最大的输出序列y')
        print('定义11.3（条件随机场）设X与Y是随机变量,P(Y|X)是在给定X的条件下Y的条件概率分布.',
            '若随机变量Y构成一个由无向图G=(V,E)表示的马尔可夫随机场,即:',
            'P(Yv|X,Yw,w!=v)=P(Yv|X,Yw,w~v)')
        print('对任意结点v成立,则称条件概率分布P(Y|X)为条件随机场.式中w~v表示在图G=(V,E)',
            '中与结点v有边连接的所有结点w,w!=v表示结点v以外的所有结点,Yv,Yu与Yw为结点v,u与w对应的随机变量.')
        print('在定义中并没有要求X和Y具有相同的结构.现实中,一般假设X和Y有相同的图结构.',
            '本书主要考虑无向图为图11.4与图11.5所示的线性链的情况,即',
            'G=(V={1,2,...,n},E={(i,i+1)}),i=1,2,...,n-1')
        print('在此情况下,X=(X1,X2,...,Xn),Y=(Y1,Y2,...,Yn),最大团是相邻两个结点集合.',
            '线性链条件随机场有下面的定义.')
        print('定义11.4（线性链条件随机场）设X=(X1,X2,...,Xn),Y=(Y1,Y2,...,Yn)均为线性链表示的随机变量序列,',
            '若在给定随机变量序列X的条件下,随机变量序列的Y的条件概率分布P(Y|X)构成条件随机场,',
            '即满足马尔可夫性P(Yi|X,Y1,...,Yi-1,Yi+1,...,Yn)=P(Yi|X,Yi-1,Yi+1),',
            'i=1,2,...,n (在i=1和n时只考虑单边)')
        print('则称P(Y|X)为线性链条件随机场.在标注问题中,X表示输入观测序列,Y表示对应的输出标记序列或状态序列')
        print('11.2.2 条件随机场的参数化形式')
        print('根据定理11.1,可以给出线性链条件随机场P(Y|X)的因子分解式,',
            '各因子是定义在相邻两个结点上的函数')
        print('定理11.2 (线性链条件随机场的参数化形式) 设P(Y|X)为线性链条件随机场,',
            '则在随机变量X取值为x的条件下,随机变量取值为y的条件概率具有如下形式：',
            'P(y|x)=1/Z(x)exp(∑lktk(yi-1,yi,x,i)+∑ulsl(yi,x,i))')
        print('其中,Z(x)=∑exp(∑lktk(yi-1,yi,x,i)+∑ulsl(yi,x,i))')
        print('式中,tk和sl是特征函数,lk和ul是对应的权值.Z(x)是规范化因子,',
            '求和是在所有可能的输出序列上进行的')
        print('上式是线性链条件随机场模型的基本形式,表示给定输入序列x,',
            '对输出序列y预测的条件概率.tk是定义在边上的特征函数,',
            '称为转移特征,依赖于当前和前一个位置,sl是定义在结点上的特征函数,',
            '称为状态特征,依赖于当前位置.',
            'tk和sl都依赖于位置,是局部特征函数.通常,特征函数tk和sl取值为1或0;',
            '当满足特征条件时取值为1,否则为0.条件随机场完全由特征函数tk,',
            'sl和对应的权值lk,ul确定.')
        print('线性链条件随机场也是对数线性模型(log linear model)')
        print('例11.1 s设有一标注问题:输入观测序列为X=(X1,X2,X3),',
            '输出标记序列为Y=(Y1,Y2,Y3),Y1,Y2,Y3取值于Y={1,2}')
        print('假设特征tk,sl和对应的权值lk,ul如下:')
        print('  t1=t1(yi-l=1,yi=2,x,i), i=2,3, l1=1')
        print('这里只注明特征取值为1的条件,取值为0的条件省略,即',
            't1(yi-l,yi,x,i)=1 yi-1=1,yi=2,x,i,(i=2,3); t1(yi-l,yi,x,i)=0 其他')
        print('对给定的观测序列x,求标记序列为y=(y1,y2,y3)=(1,2,2)的非规范化条件概率',
            '(即没有除以规范化因子的条件概率)')
        print('解 由式(11.10),线性链条件随机场模型为：')
        print('  P(y|x)∝exp[∑lk∑tk(yi-1,yi,x,i)+∑uk∑sk(yi,x,i)]')
        print('对给定的观测序列x,标记序列y=(1,2,2)的非规范化条件概率为：',
            'P(yi=1,y2=2,y3=2|x)∝exp(3.2)')
        print('11.2.3 条件随机场的简化形式')
        print('条件随机场还可以由简化形式表示.注意到条件随机场中同一特征',
            '在各个位置都有定义,可以对同一个特征在各个位置求和,将局部特征函数转换为一个全局特征函数,',
            '这样就可以将条件随机场写成权值向量和特征向量的内积形式,',
            '条件随机场的简化形式.')
        print('为简便起见,首先将转移特征和状态特征及其权值用同一的符号表示.',
            '设有K1个转移特征,K2个转移特征,K=K1+K2,记')
        print('  fk(yi,yi,x,i)=tk(yi-1,yi,x,i), k=1,2,...,K1;  ')
        print('  fk(yi,yi,x,i)=sl(yi,x,i), k=K1+l; l=1,2,...,K2')
        print('然后,对转移与状态特征在各个位置i求和,记作',
            'fk(y,x)=∑fk(yi-1,yi,x,i), k=1,2,...,K')
        print('用wk表示特征fk(y,x)的权值,即')
        print('  wk=lk, k=1,2,...,K1;  wk=ul, k=K1+l; l=1,2,...,K2')
        print('于是条件随机场可表示为：')
        print('  P(y|x)=1/Z(x)exp∑wkfk(y,x)')
        print('  Z(x)=∑exp∑wkfk(y,x)')
        print('若以w表示权值向量,即w=(w1,w2,...,wK)^T,以F(y,x)表示全局特征向量,即:',
            'F(y,x)=(f1(y,x),f2(y,x),...,fK(y,x))^T')
        print('则条件随机场可以写成向量w与F(y,x)的内积形式:',
            '其中Pw(y|x)=exp(w·F(y,x)/Zw(x))')
        print('其中,Zw(x)=∑exp(w·F(y,x))')
        print('11.2.4 条件随机场的矩阵形式')
        print('条件随机场还可以由矩阵表示.假设Pw(y|x)是由式给的线性链条件随机场,表示对给定观测序列x,',
            '相应的标记序列y的条件概率.引进特殊的起点和终点状态标记y0=start,yn+1=stop,',
            '这时Pw(y|x)可以通过矩形形式表示.')
        print('对观测序列x的每一个位置i=1,2,...,n+1,定义一个m阶矩阵(m是标记yi取值的个数)')
        print('Mi(x)=[Mi(yi-1,yi|x)] Mi(yi-1,yi|x)=exp(Wi(yi,yi|x)) Wi(yi-1,yi|x)=∑wkfk(yi-1,yi,x,i)')
        print('这样,给定观测序列x,标记序列y的非规范化概率可以通过n+1个矩阵的乘积∏Mi(yi-1,yi|x)表示,',
            '于是,条件概率Pw(y|x)是 P(y|x)=1/Z(x)∏Mi(yi-1,yi|x)')
        print('其中,Zw(x)为规范化因子,是n+1个矩阵的乘积的(start,stop)元素.')
        print('Zw(x)=(M1(x)M2(x)...Mn+1(x))start,stop')
        print('注意,y0=start与yn+1=stop表示开始状态与终止状态,规范化因子Zw(x)是以start为起点stop为终点',
            '通过状态的所有路径y1y2,...,yn的非规范化概率∏Mi(yi-1,yi|x)之和.')
        print('例11.2 给定一个由图11.6所示的线性链条件随机场,观测序列x,状态序列y,i=1,2,3,n=3,',
            '标记yi∈{1,2},假设y0=start=1,y4=stop=1,各个位置的随机矩阵M1(x),M2(x),M3(x),M4(x)分别是',
            'M1(x)=[[a01,a02],[0,0]] M1(x)=[[b11,b12],[b21,b22]] M1(x)=[[c11,c12],[c21,c22]] M1(x)=[[1,0],[1,0]]')
        print('试求状态序列y以start为起点stop为终点所有路径的非规范化概率及规范化因子')
        print('解：首先计算图11.6中从start到stop对应于y=(1,1,1),y=(1,1,2),...,y=(2,2,2)各路径的非规范化概率分别是:',
            'a01b11c11, a01b11c12, a01b12c21, a01b12c22',
            'a02b21c11, a02b21c12, a02b22c21, a02b22c22')
        print('然后按式求规范化因子. 通过计算矩阵乘积M1(x)M2(x)M3(x)M4(x)可知,其第1行第1列的元素为:',
            'a01b11c11+a02b21c11+a01b12c21+a02b22c22+a02b21c11+a02b21c12+a02b22c21+a02b22c22')
        print('恰好等于从start到stop的所有路径的非规范化概率之和,即规范化因子Z(x)')
        print('11.3 条件随机场的概率计算问题')
        print('条件随机场的概率计算问题是给条件随机场P(Y|X),输入序列x和输出序列y,计算条件概率P(Yi=yi|x),',
            'P(Yi-1=yi-1,Yi=yi|x)以及相应的数学期望的问题,为了方便,像HMM,引进前向-后向向量,递归地计算',
            '以上概率及期望值.这样的算法称为前向-后向算法')
        print('11.3.1 前向-后向算法')
        print('对每个指标i=0,1,...,n+1,定义前向向量ai(x):',
            'a0(y|x)=1, y=start;  a0(y|x)=1, y=star0, 否则')
        print('递推公式为ai^T(yi|x)=ai-1^T(yi-1|x)Mi(yi-1,yi|x), i=1,2,...,n+1')
        print('又可以表示为ai^T(x)=ai-1^T(x)Mi(x)')
        print('ai(yi|x)表示在位置i的标记是yi并且到位置i的前部分标记序列的非规范化概率,',
            'yi可取的值有m个,所以ai(x)是m维列向量.')
        print('同样,对每个指标i=0,1,...,n+1,定义后向向量bi(x):',
            'bn+1(yn+1|x)=1, yn+1=stop; bn+1(yn+1|x)=0, 否则')
        print('bi(yi|x)=Mi(yi,yi+1|x)bi-1(yi+1|x)')
        print('又可表示为：bi(x)=Mi+1(x)bi+1(x)')
        print('bi(yi|x)表示在位置i的标记为yi并且从i+1到n的后部分标记序列的非规范化概率.')
        print('由前向-后向向量定义不难得到: Z(x)=an^T(x)·1=1^T·b1(x)',
            '这里,1是元素均为1的m维列向量')
        print('11.3.2 概率计算')
        print('按照前向-后向向量的定义,很容易计算标记序列在位置i是标记yi的条件概率和在位置i-1与i是标记yi-1和yi的条件概率：',
            'P(Yi=yi|x)=ai^T(yi|x)bi(yi|x)/Z(x)',
            'P(Yi-1=yi-1,Yi=yi|x)=ai-1^T(yi-1|x)Mi(yi-1,yi|x)bi(yi|x)/Z(x)')
        print('其中,Z(x)=an^T(x)·1')
        print('11.3.3 期望值的计算')
        print('利用前向-后向向量,可以计算特征函数关于联合概率分布P(X,Y)和条件分布P(Y|x)的数学期望.')
        print('特征函数fk关于条件分布P(Y|X)的数学期望是：Ep(Y|X)[fk]=∑P(y|x)fk(y,x),k=1,2,...,K')
        print('其中,Z(x)=an^T·1')
        print('假设经验分布为P(X),特征函数fk关于联合分布P(X,Y)的数学期望是',
            'Ep(Y|X)[fk]=∑P(y|x)∑fk(yi-1,yi,x,i),k=1,2,...,K;其中,Z(x)=an^T(x)·1')
        print('是特征函数数学期望的一般计算公式.对于转移特征tk(yi-1,yi,x,i),k=1,2,...,K1,',
            '可以将式中的fk换成tk;对于状态特征,可以将式中的fk换成si,表示sl(yi,x,i),k=K1+l,l=1,2,...,K2')
        print('对于给定的观测序列x与标记序列y,可以通过一次前向扫描计算ai及Z(x),通过一次后向扫描计算bi,',
            '从而计算所有的概率和特征的期望.')
        print('11.4 条件随机场的学习算法')
        print('条件随机场的学习问题,条件随机场模型实际上是定义在时序数据上的对数线性模型,',
            '其学习方法包括极大似然估计和正则化的极大似然估计.',
            '具体的优化实现算法有改进尺度法IIS,梯度下降法以及拟牛顿法')
        print('11.4.1 改进的迭代尺度法')
        print('已知训练数据集,由此可知经验概率分布P(X,Y).可以通过极大化训练数据的对数似然函数来求模型参数.',
            '训练数据的对数似然函数为',
            'L(w)=Lp(Pw)=log∏Pw(y|x)^P(x,y)=∑P(x,y)logPw(y|x)')
        print('当Pw是一个由式给出的条件随机场模型时,对数似然函数为:',
            'L(w)=∑P(x,y)logPw(y|x)=∑∑wkfk(yj,xj)-∑logZw(xj)')
        print('改进的迭代尺度法通过迭代的方法不断优化对数似然函数改变量的下界,',
            '达到极大化对数似然函数的目的.假设模型的当前参数向量为w=(w1,w2,...,wK)^T,',
            '向量的增量为d=(d1,d2,...,dK)^T,更新参数向量为w+d=(w1+d1,w2+d2,...,wK+dK)^T')
        print('关于转移特征tk的更新方程为：Ep[tk]=∑P(x,y)∑tk(yi-1,yi,x,i), k=1,2,...,K1')
        print('关于状态特征sl的更新方程为：Ep[sl]=∑P(x,y)∑sl(yi,x,i), l=1,2,...,K2')
        print('这里,T(x,y)是在数据(x,y)中出现所有特征数的总和:',
            'T(x,y)=∑fk(y,x)=∑∑fk(yi-1,yi,x,i)')
        print('算法11.1 (条件随机场模型学习的改进的迭代尺度法)')
        print('输入:特征函数t1,t2,...,tK1,s1,s2,...,sK2;经验分布P(x,y);')
        print('输出:参数估计值w;模型Pw')
        print('(1) 对所有k∈{1,2,...,K},取初值wk=0')
        print('(2) 对每一k∈{1,2,...,K};')
        print('(a) 当k=1,2,...,K1时,令dk是方程∑P(x)P(y|x)∑tk(yi-1,yi,x,i)exp(dkT(x,y))=Ep[tk]的解')
        print('当k=K1+l,l=1,2,...,K2时,令dK1+i是方程∑P(x)P(y|x)∑sl(yi,x,i)exp(dK1+lT(x,y))=Ep[sl]的解')
        print('(b) 更新wk值:wk<-wk+dk')
        print('(3) 如果不是所有wk都收敛,重复步骤(2)')
        print('在式中,T(x,y)表示数据(x,y)中的特征总数,对不同的数据(x,y)取值可能不同,',
            '为了处理这个问题,定义松弛特征:s(x,y)=S-∑∑fk(yi-1,yi,x,i)')
        print('式中S是一个常数.选择足够大的常数S使得对训练数据集的所有数据(x,y),s(x,y)>=0成立,',
            '这时特征总数可取S.')
        print('对于转移特征tk,dk的更新方程是：∑P(x)P(y|x)∑tk(yi-1,yi,x,i)exp(dkS)=Ep[tk]',
            'dk=1/SlogEp[tk]/Ep[tk].')
        print('其中,Ep(tk)=∑P(x)∑∑tk(yi-1,yi,x,i)ai-1^T(yi-1|x)Mi(yi-1,yi|x)bi(yi|x)/Z(x)')
        print('同样由式,对于状态特征sl,dk的更新方程是:',
            '∑P(x)P(y|x)∑sl(yi,x,i)exp(dK1+lS)=Ep[sl]',
            'dK1+lS=1/SlogEp[sl]/Ep[sl]')
        print('其中,Ep(sl)=∑P(x)∑∑sl(yi,x,i)ai^T(yi|x)bi(yi|x)/Z(x)')
        print('以上算法称为算法S.在算法S中需要使得常数S取足够大,这样一来,',
            '每步迭代增量向量会变大,算法收敛会变慢.算法T试图解决这个问题.',
            '算法T对每个观测序列x计算其特征总数最大值T(x):T(x)=maxT(x,y)')
        print('利用前向-后向递推公式,可以很容易地计算T(x)=t.')
        print('这时,关于转移特征参数的更新方程可以写成:',
            'Ep[tk]=∑P(x)P(y|x)∑tk(yi-1,yi,x,i)exp(dkT(x))=∑ak,tbk')
        print('这里,ak,t是特征tk的期待值,dk=logbk.bk是多项式方程唯一的实根,',
            '可以用牛顿法求得,从而求得相关的dk.')
        print('同样,关于状态特征的参数更新方程可以写成:')
        print('Ep[sl]=∑P(x)P(y|x)∑sl(yi,x,i)exp(dk1+lT(x))=∑bl,yl')
        print('这里,bl,t是特征sl的期望值,dl=logyl,yl是多项式方程唯一的实根,也可以用牛顿法求得.')
        print('11.4.2 拟牛顿法')
        print('条件随机场模型学习还可以应用牛顿法或拟牛顿法.对于条件随机场模型',
            'Pw(y|x)=exp(∑wifi(x,y))/∑exp(∑wifi(x,y))')
        print('学习的优化目标函数是minf(w)=P(x)log∑exp(∑wifi(x,y))-∑P(x,y)∑fi(x,y),',
            '其梯度函数是g(w)=∑P(x)Pw(y|x)f(x,y)-Ep(f)')
        print('拟牛顿法的BSGS算法如下.')
        print('算法11.2 (条件随机场模型学习的BFGS算法)')
        print('输入：特征函数f1,f2,...,fn：经验分布P(X,Y)')
        print('输出：最优参数值w:最优模型Pw(y|x)')
        print('(1) 选定初始点w(0),取B0为正定对称矩阵,置k=0')
        print('(2) 计算gk=g(w(k)).若gk=0,则停止计算;否则转(3)')
        print('(3) 由Bkpk=-gk求出pk')
        print('(4) 一维搜索:求lk使得f(w(k)+lkpk)=minf(w(k)+lpk)')
        print('(5) 置w(k+1)=w(k)+lkpk')
        print('(6) 计算gk+1=g(w(k+1)),若gk=0,则停止计算；否则,按下式求出Bk+1',
            'Bk+1=Bk+ykyk^T/yk^Tdk-Bkdkdk^TBk/dk^TBkdk',
            '其中,yk=gk+1-gk, dk=w(k+1)-w(k)')
        print('(7) 置k=k+1,转(3)')
        print('11.5 条件随机场的预测算法')
        print('条件随机场的预测问题是给定条件随机场P(Y|X)和输入序列(观测序列)x,',
            '求条件概率最大的输出序列(标记序列)y*,即对观测序列进行标注.条件随机场的预测算法是',
            '著名的维特比算法.')
        print('由式子可得:y*=argmaxPw(y|x)=argmaxexp(w·F(y,x))/Z(w)=',
            'argmaxexp(w·F(y,x))=argmax(w·F(y,x))')
        print('于是,条件随机场的预测问题成为求非规范化概率最大的最优路径问题max(w·F(y,x))')
        print('这里,路径表示标记序列.其中,w=(w1,w2,...,wK)^T',
            'F(y,x)=(f1(y,x),f2(y,x),...,fK(y,x)^T)',
            'fk(y,x)=∑fk(yi-1,yi,x,i),k=1,2,...,K')
        print('注意,这时只需计算非规范化概率,而不必计算概率,可以大大提高效率.为了求解最优路径,',
            '将式写成如下形式:max∑w·Fi(yi-1,yi,x)')
        print('其中,Fi(yi-1,yi,x)=(f1(yi-1,yi,x,i),f2(yi-1,yi,x,i),...,fK(yi-1,yi,x,i))^T',
            '是局部特征向量.')
        print('下面叙述维特比算法.首先求出位置1的各个标记j=1,2,...,m的非规范化概率:',
            'd1(j)=w·F1(y0=start,y1-j,x),j=1,2,...,m')
        print('一般地,由递推公式,求出到位置i的各个标记l=1,2,...,m的非规范化概率的最大值',
            '同时记录非规范化概率最大值的路径',
            'di(l)=max{di-1(j)+w·Fi(yi-1=j,yi=l,x)},l=1,2,...,m')
        print('fi(l)=argmax{di-1(j)+w·Fi(yi-1=j,yi=l,x)},l=1,2,...,m')
        print('直到i=n时终止.这时求得非规范化概率的最大值为：',
            'max(w·F(y,x))=maxdn(j)')
        print('及最优路径的终点:yn*=argmaxdn(j).')
        print('由此最优路径终点返回,yi*=fi+1(yi+1*), i=n-1,n-2,...,1')
        print('求得最优路径y*=(y1*,y2*,...,yn*)')
        print('综上所述,得到条件随机场预测的维特比算法:')
        print('算法11.3 (条件随机场预测的维特比算法)')
        print('输入：模型特征向量F(y,x)和权值向量w,观测序列x=(x1,x2,...,xn);')
        print('输出：最优路径y*=(y1*,y2*,...,yn*)')
        print('(1) 初始化 d1(j)=w·F1(y0=start,y1=j,x),j=1,2,...,m')
        print('(2) 递推.对i=2,3,...,n')
        print('  di(l)=max{di-1(j)+w·Fi(yi-1=j,yi=l,x)},l=1,2,...,m')
        print('  fi(l)-argmax{di-1(j)+w·Fi(yi-1=j,yi=l,x)},l=1,2,...,m')
        print('(3) 终止')
        print('  max(w·F(y,x))=maxdn(j), yn*=argmaxdn(j)')
        print('(4) 返回路径 yi*=fi+1(yi+1*),i=n-1,n-2,...,1',
            '求得最优路径y*=(y1*,y2*,...,yn*)')
        print('例11.3 在例11.1中,用维特比算法求给定的输入序列(观测序列)x对应的最优输出序列(标记序列) y*=(y1*,y2*,y3*)')
        print('解：特征函数及对应的权值在例11.1中给出.')
        print('利用维特比算法求最优路径问题:max∑w·Fi(yi-1,yi,x)')
        print('(1) 初始化 d1(j)=w·F1(y0=start,y1=j,x),j=1,2,',
            'i=1,d1(1)=1,d1(2)=0.5')
        print('(2) 递推')
        print('i=2 d2(l)=max{d1(j)+w·F2(j,l,x)}')
        print('    d2(1)=max{1+l2t2,0.5+l4t4}=1.6, f2(1)=1')
        print('    d2(2)=max{1+l1t1+u2s2,0.5+u2s2}=2.5, f2(2)=1')
        print('i=3 d3(l)=max{d2(j)+w·F3(j,l,x)}')
        print('    d3(l)=max{1.6+u5s5,2.5+l3t3+u3s3}=4.3, f3(1)=2')
        print('    d3(2)=max{1.6+l1t1+u4s4,2.5+l5t5+u4s4}=3.2, f3(2)=1')
        print('(3) 终止')
        print('    max(w·F(y,x))=maxd3(l)=d3(1)=4.3  y3*=argmaxd3(l)=1')
        print('(4) 返回')
        print('   y2*=f3(y3*)=f3(1)=2  y1*=f2(y2*)=f2(2)=1')
        print('最优标记序列')
        print('   y*=(y1*,y2*,y3*)=(1,2,1)')
        print('本章概要')
        print('1.概率无向图模型是由无向图表示的联合概率分布.无向图上的结点之间的连接关系表示了联合分布的随机变量集合之间的条件独立性',
            '即马尔可夫性.因此,概率无向图模型也称为马尔可夫随机场.',
            '概率无向图模型或马尔可夫随机场',
            '概率无向图模型或马尔可夫随机场的联合概率分布可以分解为无向图最大团上的正值函数的乘积的形式.')
        print('2.条件随机场是给定输入随机变量X条件下,输出随机变量Y的条件概率分布模型,其形式为参数化的对数线性模型.',
            '条件随机场的最大特点是假设输出变量之间的联合概率分布构成概率无向图模型,',
            '即马尔可夫随机场.条件随机场是判别模型')
        print('3.线性链条条件随机场是定义在观测序列与标记序列上的条件随机场.',
            '线性链随机场一般表示为给定观测序列条件下的标记序列的条件概率分布,',
            '由参数化的对数线性模型表示.模型包含特征及相应的权值,',
            '特征是定义在线性链的边与结点上的.线性链条件随机场的数学表达式是:',
            'P(y|x)=1/Z(x)exp(∑lktk(yi-1,yi,x,i)+∑ulsl(yi,x,i))','其中,',
            'Z(x)=∑exp(∑lktk(yi-1,yi,x,i)+∑ulsl(yi,x,i))')
        print('4.线性链条件随机场的概率计算通常利用前向-后向算法')
        print('5.条件随机场的学习方法通常是极大似然估计方法或正则化的极大似然估计,',
            '即在给定训练数据下,通过极大化训练数据的对数似然函数以估计模型参数.',
            '具体的算法有改进的迭代尺度算法、梯度下降法、拟牛顿法')
        print('6.线性链条件随机场的一个重要应用是标注.维特比算法是给定观测序列求条件概率最大的标记序列的方法.')

chapter11 = Chapter11()

def main():
    chapter11.note()

if __name__ == '__main__':
    main()