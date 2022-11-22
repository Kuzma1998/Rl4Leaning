from common.utils import save_results, make_dir
from Env import NormalizedActions, OUNoise, Env
from common.plot import plot_rewards, plot_rewards_cn
from agent import DDPG
import torch
import datetime
import sys
import os
from dataAndModel.Narx import Net
curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
sys.path.append(parent_path)  # 添加父路径到系统路径sys.path


curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # 获取当前时间


class DDPGConfig:
    def __init__(self):
        self.algo = 'DDPG'  # 算法名称
        self.env = 'Leaching'  # 环境名称
        self.result_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/results/'  # 保存结果的路径
        self.model_path = curr_path+"/outputs/" + self.env + \
            '/'+curr_time+'/models/'  # 保存模型的路径
        self.train_eps = 100 # 训练的回合数
        self.eval_eps = 1  # 测试的回合数
        self.gamma = 0.95  # 折扣因子
        self.critic_lr = 1e-3  # 评论家网络的学习率
        self.actor_lr = 1e-3  # 演员网络的学习率
        self.memory_capacity = 1000000
        self.batch_size = 128
        self.hidden_dim = 128
        self.soft_tau = 1e-2  # 软更新参数
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")


def env_agent_config(cfg):
    env = Env()
    state_dim = 7  # 7维 连续
    action_dim = 1  # 1维连续
    agent = DDPG(state_dim, action_dim, cfg)
    return env, agent


def train(cfg, env, agent):
    print('开始训练！')
    print(f'环境：{cfg.env}，算法：{cfg.algo}，设备：{cfg.device}')
    # ou_noise = OUNoise()  # 动作噪声
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(cfg.train_eps):
        state, action = env.reset()  # 初始化一个state
        ep_reward = 0
        i_step = 3
        env.noise.reset()
        env.update_times += 1
        # if env.update_times % 10 == 0:
        #     env.normalize()
        while i_step < 515:
            # action = action.detach()
            action = agent.choose_action(state, action).detach().numpy()
            next_state, reward, action = env.step(action, i_step, True)
            ep_reward += reward
            # 存入经验池的四元组(这个时刻状态，动作，奖励，下一时刻奖励，是否结束游戏)
            agent.memory.push(state, action, reward, next_state)
            agent.update()
            state = next_state
            i_step += 1

        print('回合：{}/{}，奖励：{:.2f}'.format(i_ep+1,
                                          cfg.train_eps, ep_reward[0]))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
        RL_pH, RL_feisuan = env.get_result()
    print('完成训练！')
    return rewards, ma_rewards


def eval(cfg, env, agent):
    print('开始测试！')
    print(f'环境：{cfg.env}, 算法：{cfg.algo}, 设备：{cfg.device}')
    rewards = []  # 记录奖励
    ma_rewards = []  # 记录滑动平均奖励
    for i_ep in range(cfg.eval_eps):
        state, action = env.reset()  # 初始化一个state
        ep_reward = 0
        i_step = 3
        while i_step < 515:
            action = agent.choose_action(state, action).detach().numpy()
            next_state, reward, action = env.step(action, i_step, False)
            ep_reward += reward
            state = next_state
            i_step += 1

        print('回合：{}/{}, 奖励：{}'.format(i_ep+1, cfg.train_eps, ep_reward))
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9*ma_rewards[-1]+0.1*ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print('完成测试！')
    return rewards, ma_rewards


if __name__ == "__main__":
    cfg = DDPGConfig()
    # 训练
    env, agent = env_agent_config(cfg)
    rewards, ma_rewards = train(cfg, env, agent)
    make_dir(cfg.result_path, cfg.model_path)
    agent.save(path=cfg.model_path)
    save_results(rewards, ma_rewards, tag='train', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="train",
                    env=cfg.env, algo=cfg.algo, path=cfg.result_path)
    # 测试
    env, agent = env_agent_config(cfg)
    agent.load(path=cfg.model_path)
    rewards, ma_rewards = eval(cfg, env, agent)
    save_results(rewards, ma_rewards, tag='eval', path=cfg.result_path)
    plot_rewards_cn(rewards, ma_rewards, tag="eval",
                    env=cfg.env, algo=cfg.algo, path=cfg.result_path)
