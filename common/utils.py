'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2022-11-14 21:03:06
Discription: 
Environment: 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties

def save_results(rewards,ma_rewards,tag='train',path='./results'):
    '''save rewards and ma_rewards
    '''
    np.save(path+'{}_rewards.npy'.format(tag), rewards)
    np.save(path+'{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')

def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
def del_empty_dir(*paths):
    '''del_empty_dir delete empty folders unders "paths"
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))



# plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
# plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
#有中文出现的情况，需要u'内容'

def chinese_font():  
    return FontProperties(fname=r'C:/Windows/Fonts/msyh.ttc',size=15)  # 系统字体路径，此处是mac的
def plot_rewards(rewards,ma_rewards,tag="train",env='CartPole-v0',algo = "DQN",save=True,path='./'):
    sns.set()
    plt.title("average learning curve of {} for {}".format(algo,env))
    plt.xlabel('epsiodes')
    plt.plot(rewards,label='rewards')
    plt.plot(ma_rewards,label='ma rewards')
    plt.legend()
    if save:
        plt.savefig(path+"{}_rewards_curve".format(tag))
    plt.show()

def plot_rewards_cn(rewards,ma_rewards,tag="train",env='CartPole-v0',algo = "DQN",save=True,path='./'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(env,algo),fontproperties=chinese_font())
    # plt.title(u"{}环境下{}算法的学习曲线".format(env,algo))
    plt.xlabel(u'回合数',fontproperties=chinese_font())
    # plt.xlabel(u'回合数')
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励',u'滑动平均奖励',),loc="best",prop=chinese_font())
    if save:
        plt.savefig(path+f"{tag}_rewards_curve_cn")
    plt.show()

def plot_losses(losses,algo = "DQN",save=True,path='./'):
    sns.set()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses,label='rewards')
    plt.legend(prop=chinese_font())
    if save:
        plt.savefig(path+"losses_curve")
    plt.show()
   
