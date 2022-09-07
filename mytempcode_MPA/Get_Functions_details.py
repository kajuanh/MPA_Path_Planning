import numpy as np
import random


def f1(x):
    o = np.sum(x**2)
    return o


def f2(x):
    o = np.sum(np.absolute(x))+np.prod(np.absolute(x))
    return o


def f3(x):
    # print("xf3",x)
    # print("xf3",x.shape)
    # dim = x.shape[0]
    dim = 100
    o = 0
    for i in range(1, dim):
        o = o+(np.sum(x[:i]))**2
    return o


def f4(x):
    o = max(abs(x))
    return o


def f5(x):
    #     dim=size(x,2);
    # o=sum(100*(x(2:dim)-(x(1:dim-1).^2)).^2+(x(1:dim-1)-1).^2);
    dim = x.shape[0]
    # dim = 200
    # t100 = np.ones(dim)*100
    o = np.sum(100*(x[1:]-((x[:dim-1])**2))**2+((x[:dim-1]-1)**2))
    return o


def f6(x):
    # o=sum(abs((x+.5)).^2);
    o = np.sum(abs((x+0.5)**2))
    return o


def f7(x):
    #     dim=size(x,2);
    # o=sum([1:dim].*(x.^4))+rand;
    dim = x.shape[0]
    h = np.arange(1, dim+1)
    rand = random.uniform(0, 1)
    o = np.sum(h*(x**4))+rand
    return o


def f8(x):
    # o=sum(-x.*sin(sqrt(abs(x))))
    o = sum((-x)*(np.sin(np.sqrt(abs(x)))))
    return o


def f9(x):
    # dim=size(x,2);
    # o=sum(x.^2-10*cos(2*pi.*x))+10*dim;
    dim = x.shape[0]
    o = np.sum(x**2-10*np.cos(2*np.pi*x))+10*dim
    return o


def f10(x):
    #     dim=size(x,2);
    # o=-20*exp(-.2*sqrt(sum(x.^2)/dim))-exp(sum(cos(2*pi.*x))/dim)+20+exp(1);
    dim = x.shape[0]
    o = -20*np.exp(-0.2*np.sqrt(np.sum(x**2)/dim)) - \
        np.exp(np.sum(np.cos(2*np.pi*x)/dim))+20+np.exp(1)
    return o


def F1():
    lb = -100
    ub = 100
    dim = 50
    x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
    y = x
    # def fobj(x): return f1(x)
    def fobj(x): return f1(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F2():
    # fobj = @F2
    lb = -10
    ub = 10
    dim = 50
    x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f2(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F3():
    lb = -100
    ub = 100
    dim = 50
    x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f3(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F4():
    # fobj = @F4;
    lb = -100
    ub = 100
    dim = 50
    x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f4(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F5():
    # fobj = @F5;
    lb = -30
    ub = 30
    dim = 50
    x = np.arange(-200, 202, 2)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f5(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F6():
    # fobj = @F6;
    lb = -100
    ub = 100
    dim = 50
    x = np.arange(-100, 102, 2)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f6(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F7():
    # fobj = @F7;
    lb = -1.28
    ub = 1.28
    dim = 50
    x = np.arange(-1, 1.01, 0.03)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f7(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F8():
    # fobj = @F8;
    lb = -500
    ub = 500
    dim = 50
    x = np.arange(-500, 510, 10)  # x=-100:2:100; y=x;
    y = x
    def fobj(x): return f8(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F9():
    # fobj = @F9;x=-5:0.1:5;   y=x; %[-5,5]
    lb = -5.12
    ub = 5.12
    dim = 50
    x = np.arange(-5, 5.1, 0.1)
    y = x
    def fobj(x): return f9(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F10():
    # fobj = @F10;x=-20:0.5:20; y=x;%[-500,500]
    lb = -32
    ub = 32
    dim = 50
    x = np.arange(-20, 20.5, 0.5)
    y = x
    def fobj(x): return f10(x)
    return {'lb': lb, 'ub': ub, 'dim': dim, 'fobj': fobj}


def F11():
    # fobj = @F11;
    lb = -600
    ub = 600
    dim = 50


def F12():
    # fobj = @F12;
    lb = -50
    ub = 50
    dim = 50


def F13():
    # fobj = @F13;
    lb = -50
    ub = 50
    dim = 50


def F14():
    # fobj = @F14;
    lb = -65.536
    ub = 65.536
    dim = 2


def F15():
    # fobj = @F15;
    lb = -5
    ub = 5
    dim = 4


def F16():
    # fobj = @F16;
    lb = -5
    ub = 5
    dim = 2


def F17():
    # fobj = @F17;
    lb = [-5, 0]
    ub = [10, 15]
    dim = 2


def F18():
    # fobj = @F18;
    lb = -2
    ub = 2
    dim = 2


def F19():
    # fobj = @F19;
    lb = 0
    ub = 1
    dim = 3


def F20():
    # fobj = @F20;
    lb = 0
    ub = 1
    dim = 6


def F21():
    # fobj = @F21;
    lb = 0
    ub = 10
    dim = 4


def F22():
    # fobj = @F22;
    lb = 0
    ub = 10
    dim = 4


def F23():
    # fobj = @F23;
    lb = 0
    ub = 10
    dim = 4


def switcher(F):
    switcher = {
        1: F1(),
        2: F2(),
        3: F3(),
        4: F4(),
        5: F5(),
        6: F6(),
        7: F7(),
        8: F8(),
        9: F9(),
        10: F10(),
        # 11: F11(),
        # 12: F12(),
        # 13: F13(),
        # 14: F14(),
        # 15: F15(),
        # 16: F16(),
        # 17: F17(),
        # 18: F18(),
        # 19: F19(),
        # 20: F20(),
        # 21: F21(),
        # 22: F22(),
        # 23: F23()
    }
    return switcher.get(F, "nothing")
