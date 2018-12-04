
class Appendix:
    """
    附录A 梯度下降法
    """
    def __init__(self):
        """
        附录A 梯度下降法
        """
        pass

    def note(self):
        """
        附录A 梯度下降法
        """
        print('附录A 梯度下降法')
        print('梯度下降法(gradient descent)或最速下降法(steepest descent)是求解无约束最优化问题的一种最常用的方法',
            '有实现简单的优点.梯度下降法是迭代算法,每一步需要求解目标函数的梯度向量')
        print('假设f(x)是R^n上具有一阶连续偏导数的函数.要求解的无约束最优化问题是:minf(x)',
            'x*表示目标函数f(x)的极小点.x*表示目标函数f(x)的极小点.')
        print('梯度下降法是一种迭代算法.选取适当的初值x(0),不断迭代,更新x的值,',
            '进行目标函数的极小化,直到收敛.由于负梯度方向是使函数值下降最快的方向,',
            '在迭代的每一步,以负梯度方向更新x的值,从而达到减少函数值的目的.')
        print('由于f(x)具有一阶连续偏导数,若第k次迭代值为x^(k),则可将f(x)在x^(k)附近进行一阶泰勒展开:',
            'f(x)=f(x(k))+gk^T(x-x(k)).这里,gk=g(x(k))=gradf(x^(k))为f(x)在x^(k)的梯度.',
            '这里,gk=g(x(k))=gradf(x(k))为f(x)在x^(k)的梯度.',
            '求出第k+1次迭代值x(k+1):x(k+1)<-x(k)+lkpk')
        print('其中,pk是搜索方向,取负梯度方向pk=-gradf(x(k)),lk是步长,',
            '由一维搜索确定,即lk使得f(x(k),lkpk)=minf(x^(k)+lkpk)',
            '梯度下降算法如下：')
        print('算法A.1 (梯度下降法)')
        print('输入：目标函数f(x),梯度函数g(x)=gradf(x),计算精度e;')
        print('输出：f(x)的极小点x*.')
        print('(1) 取初始值x(0)∈R^n,置k=0')
        print('(2) 计算f(x^(k))')
        print('(3) 计算梯度gk=g(x^(k)),当||gk||<e,停止迭代,令x*=x^(k);否则,',
            '令pk=-g(x^(k)),求lk,使')
        print('附录B 牛顿法和拟牛顿法')
        print('牛顿法(Newton method)和拟牛顿法(quasi Newton method)也是求解无约束最优化问题的常用方法,',
            '有收敛速度快的优点.牛顿法是迭代算法,每一步需要求解目标函数的海森矩阵的逆矩阵,计算比较复杂.',
            '拟牛顿法通过正定矩阵近似海森矩阵的逆矩阵或海森矩阵,')
        print('1.牛顿法')
        print('考虑无约束最优化问题minf(x),其中x*为目标函数的极小点.')
        print('假设f(x)具有二阶连续偏导数,若第k次迭代值为x^(k),则可将f(x)在x^(k)附近进行二阶泰勒展开:',
            'f(x)=f(x^(k))+gk^T(x-x(k))+0.5(x-x^(k))^TH(x^(k))(x-x(k))')
        print('这里,gk=g(x^(k))=gradf(x^(k))是f(x)的梯度向量在点x^(k)的值,H(x^(k))是f(x)的海森矩阵(Hesse matrix),',
            'H(x)=[d2f/dxidxj](n*n)')
        print('在点x^(k)的值.函数f(x)有极值的有必要条件是在极值点处一阶导数为0,',
            '即梯度向量为0.特别是当H(x^(k))是正定矩阵时,函数f(x)的极值为极小值')
        print('牛顿法利用极小点的必要条件gradf(x)=0,每次迭代中从点x^(k)开始,求目标函数的极小点,',
            '作为第k+1次迭代值x(k+1).具体地,假设x(k+1)满足:gradf(x(k+1))=0',
            'gradf(x)=gk+Hk(x-x(k))')
        print('其中Hk=H(x^(k)). 这样有gk+Hk(x^(k+1)-x^(k))=0')
        print('因此,x^(k+1)=x^(k)-Hk^-1gk')
        print('或者,x^(k+1)=x^(k)+pk')
        print('其中,Hkpk=-gk')
        print('算法B.1 (牛顿法)')
        print('输入:目标函数f(x),梯度g(x)=gradf(x),海森矩阵H(x),精度要求e;')
        print('输出:f(x)的极小点x*')
        print('(1) 取初始点x^(0),置k=0')
        print('(2) 计算gk=g(x^(k))')
        print('(3) 若||gk||<e,则停止计算,得近似解x*=x^(k)')
        print('(4) 计算Hk=H(x^(k)),并求pk,Hkpk=-gk')
        print('(5) 置x^(k+1)=x^(k)+pk')
        print('(6) 置k=k+1,转(2)')
        print('步骤(4)求pk,pk=-Hk^-1gk,要求Hk^-1,计算比较负责,所以有其他改进的方法.')
        print('2.拟牛顿法的思路')
        print('在牛顿法的迭代中,需要计算海森矩阵的逆矩阵H^-1,这一计算比较复杂,',
            '考虑用一个n阶矩阵Gk=G(x^(k))来近似代替Hk^-1=H^-1(x^(k)).',
            '这就是拟牛顿法的基本想法.')
        print('先看牛顿法迭代中海森矩阵Hk满足的条件.首先,Hk满足以下关系.',
            '在式子中取x=x^(k+1),即得gk+1-gk=Hk(x^(k+1)-x(k))')
        print('记yk=gk+1-gk,dk=x(k+1)-x(k),则 yk=Hkdk 或Hk^-1yk=dk')
        print('如果Hk是正定的(Hk^-1也是正定的),那么可以保证牛顿法搜索方向pk是下降方向.',
            '这是因为搜索方向是pk=-lgk,x=x(k)+lpk=x(k)-lHk^-1gk')
        print('所以f(x)在x^(k)的泰勒展开式可以近似写成:f(x)=f(x^(k))-lgk^THk^-1gk')
        print('因Hk^-1正定,故有gk^THk^-1gk>0.当l为一个充分小的正数时,总有f(x)<f(x^(k)),',
            '也就是说pk是下降方向.')
        print('拟牛顿法将Gk作为Hk^-1的近似,要求矩阵Gk满足同样的条件.首先,每次迭代矩阵Gk是正定的.',
            '同时,Gk满足下面的逆牛顿条件：Gk+1=Gk+ΔGk')
        print('这种选择有一定的灵活性,因此有多种具体实现方法.下面介绍Broyden类拟牛顿法.')
        print('3.DFP (Davidon-Fletcher-Powell)算法(DFP algorithm)')
        print('DFP算法选择Gk+1的方法是,假设每一步迭代中矩阵Gk+1是由Gk加上两个附加项构成的,即',
            'Gk+1=Gk+Pk+Qk. 其中Pk,Qk是待定矩阵.这时,Gk+1yk=Gkyk+Pkyk+Qkyk')
        print('为使Gk+1满足拟牛顿条件,可使Pk和Qk满足:Pkyk=dk,Qkyk=-Gkyk')
        print('事实上,不难找出这样的Pk和Qk,例如取：',
            'Pk=dkdk^T/dk^Tyk Qk=-Gkykyk^TGk/yk^TGkyk')
        print('这样就可以得到矩阵Gk+1的迭代公式:',
            'Gk+1=Gk+dkdk^T/dk^Tyk-Gkykyk^TGk/yk^TGkyk称为DFP算法')
        print('可以证明:如果初始矩阵G0是正定的,则迭代过程中每个矩阵Gk都是正定的.')
        print('DFP算法如下:')
        print('算法B.2 (DFP算法)')
        print('输入:目标函数f(x),梯度g(x)=gradf(x),精度要求e;')
        print('输出:f(x)的极小值点x*')
        print('(1) 选定初始点x(0),取G0为正定对称矩阵,置k=0')
        print('(2) 计算gk=g(x(k)).若||gk||<e,则停止计算,得近似解x*=x^(k);否则转(3)')
        print('(3) 置pk=-Gkgk')
        print('(4) 一维搜索:求lk使得 f(x^(k)+lkpk)=minf(x^(k)+lpk)')
        print('(5) 置x^(k+1)=x^(k)+lkpk')
        print('(6) 计算gk+1=g(x^(k+1)),若||gk+1||<e,则停止计算,得近似解x*=x^(k+1);',
            '否则,计算Gk+1')
        print('(7) 置k=k+1,转(3)')
        print('4.BFGS算法(BFGS algorithm)')
        print('BFGS算法是最流行的拟牛顿算法.')
        print('可以考虑用Gk逼近海森矩阵的逆矩阵H^-1,也可以考虑用Bk逼近海森矩阵H.',
            '这时,相应的拟牛顿条件是Bk+1dk=yk')
        print('可以用同样的方法得到另一迭代公式.首先令Bk+1=Bk+Pk+Qk',
            'Bk+1dk=Bkdk+Pkdk+Qkdk')
        print('考虑使Pk和Qk满足:Pkdk=yk, Qkdk=-Bkdk')
        print('找出适合条件的Pk和Qk,得到BFGS算法矩阵Bk+1的迭代公式:',
            'Bk+1=Bk+ykyk^T/yk^Tdk-Bkdkdk^TBk/dk^TBkdk')
        print('可以证明,如果初始矩阵B0是正定的,则迭代过程中的每个矩阵Bk都是正定的.',
            '下面写出BFGS拟牛顿算法')
        print('算法B.3 (BFGS算法)')
        print('输入:目标函数f(x),g(x)=gradf(x),精度要求e;')
        print('输出:f(x)的极小点x*.')
        print('(1) 选定初始点x(0),取B0为正定对称矩阵,置k=0')
        print('(2) 计算gk=g(x(k)).若||gk||<e,则停止计算,得近似解x*=x(k);',
            '否则转(3)')
        print('(3) 由Bkpk=-gk求出pk')
        print('(4) 一维搜索:求lk使得f(x(k)+lkpk)=minf(x(k)+lpk)')
        print('(5) 置x(k+1)=x(k)+lkpk')
        print('(6) 计算gk+1=g(x(k+1)),若||gk+1||<e,则停止计算,得近似解x*=x^(k+1);',
            '否则,算出Bk+1')
        print('(7) 置k=k+1,转(3)')
        print('5.Broyden类算法 (Broyden\'s algorithm)')
        print('可以从BFGS算法矩阵Bk的迭代公式得到BFGS算法关于Gk的迭代公式.',
            '若记Gk=Bk^-1, Gk+1=Bk+1^-1,那么对BFGS算法矩阵的迭代公式应用两次Sherman-Morrison公式即得:',
            'Gk+1=(I-dkyk^T/dk^Tyk)Gk(I-dkyk^T/dk^Tyk)+dkdk^T/dk^Tyk',
            '称为BFGS算法关于Gk的迭代公式.')
        print('由DFP算法Gk的迭代公式得到的Gk+1记作G(DFP),由BFGS算法Gk的迭代公式得到的Gk+1记作G(BFGS),',
            '它们都满足方程拟牛顿条件式,所以它们的线性组合:Gk+1=aG(DFP)+(1-a)G(BFGS)',
            '也满足拟牛顿条件式,而且是正定的.其中0<=a<=1.这样就得到了一类拟牛顿法,',
            '称为Broyden类算法')
        print('附录C 拉格朗日对偶性')
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
    
appendix = Appendix()

def main():
    appendix.note()

if __name__ == '__main__':
    main()
