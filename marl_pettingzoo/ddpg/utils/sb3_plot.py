import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np


def plot(data, 
              en: bool = True, 
              only_fig: bool = True, 
              fig_path: str = None):
    path = fig_path
    if en:
        plot_en(data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3'], data['arr_5'])
    elif not en:
        plot_zh(data, only_fig, path)


def plot_en(
    frame_idx: int, 
    scores: List[float],
    actor_losses: List[float],
    critic_losses: List[float],
    result_array: List[int],
):
    training_plot(frame_idx, scores, actor_losses, critic_losses)
    rate_plot(result_array)


def training_plot(
    frame_idx: int, 
    scores: List[float],
    actor_losses: List[float],
    critic_losses: List[float],
):
    """Plot the training progresses."""
    plt.figure(figsize=(21, 5))
    plt.subplot(131)
    plt.title('the total steps: %s. the mean score: %.3f.' % (frame_idx, np.mean(scores[-10:])))
    plt.plot(scores)
    plt.xlabel("episode")
    plt.ylabel("score")
    plt.subplot(132)
    plt.title('actor_loss')
    plt.plot(actor_losses)
    plt.xlabel("step")
    plt.ylabel("actor_loss")
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    plt.subplot(133)
    plt.title('critic_loss')
    plt.xlabel("step")
    plt.ylabel("critic_loss")
    plt.plot(critic_losses)
    plt.ticklabel_format(style='sci', scilimits=(0,0), axis='x')
    plt.show()


def rate_plot(
    result_array: List[int],
):    
    # win_rate
    plt.figure()
    plt.title('rates in every 50 episode')
    win_array, tie_array, lose_array = rate_val(result_array)
    plt.plot(win_array, label = 'win_rates')
    plt.plot(tie_array, label = 'tie_rates')
    plt.plot(lose_array, label = 'lose_rates')
    plt.xlabel("every 50 episode")
    plt.ylabel("rates")
    plt.legend()
    plt.show()


def rate_val(
    result_array: List[int]
):
    win_array = []
    tie_array = []
    lose_array = []
    for i in range(len(result_array)):
        if i >= 100:
            result_part = np.copy(result_array[i-100:i+1])
        else:
            result_part = np.copy(result_array[:i+100])
        wins = 0
        loses = 0
        ties = 0
        for result in result_part:
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            elif result == -1:
                loses += 1
        win_rate = wins / len(result_part)
        tie_rate = ties / len(result_part)
        lose_rate = loses / len(result_part)
        win_array.append(win_rate)
        tie_array.append(tie_rate)
        lose_array.append(lose_rate)
    return win_array, tie_array, lose_array


def plot_zh(data_1, only_fig: bool, path: str):
    """
    NOTE: 实现方式和sb3_plot_en.py不同
    """
    reward_array_1 = data_1['arr_1']
    average_reward_array_1 = average_reward(reward_array_1)
    one_reward_plot(average_reward_array_1, "MADDPG", only_fig, path)
    rate_plot_cn(data_1['arr_5'], only_fig, path)


def average_reward(reward_array: np.ndarray):
    # 每100个episode的平均奖励
    average_reward_array = np.zeros(len(reward_array))
    for i in range(len(reward_array)):
        if i >= 100:
            reward_part = np.copy(reward_array[i-100:i+1])
        else:
            reward_part = np.copy(reward_array[:i+100])
        average_reward_array[i] = np.mean(reward_part)
    return average_reward_array


def one_reward_plot(reward_array: np.ndarray,
                    array_name_1: str,
                    only_fig: bool,
                    fig_path: str):
    plt.figure(figsize=(10, 7))  # (7, 5)
    plt.plot(reward_array, label = array_name_1)
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xlabel("轮次", fontdict={"family": "SimSun", "size": 25})
    plt.ylabel("平均奖励", fontdict={"family": "SimSun", "size": 25})
    plt.legend(prop={'family': 'Times New Roman', 'size': 22})
    plt.grid()
    plt.margins(x=0)
    plt.savefig(os.path.join(fig_path, 'rew.png'))
    if only_fig is False:
        plt.show()


def two_reward_plot(reward_array_1: np.ndarray, 
                    reward_array_2: np.ndarray,
                    array_name_1: str,
                    array_name_2: str):
    # 需要输入reward_array来自哪个算法
    plt.figure(figsize=(10, 7))
    plt.plot(reward_array_1, label = array_name_1)
    plt.plot(reward_array_2, label = array_name_2)
    plt.tick_params(axis='both', labelsize=15)
    plt.xlabel("轮次", fontdict={"family": "SimSun", "size": 25})
    plt.ylabel("平均奖励", fontdict={"family": "SimSun", "size": 25})
    plt.legend(prop={'family': 'Times New Roman', 'size': 22})
    plt.show()


def rate_plot_cn(
    result_array: List[int],
    only_fig: bool,
    fig_path: str,
):    
    # win_rate
    plt.figure(figsize=(10, 7))
    win_array, tie_array, lose_array = rate_val(result_array)
    plt.plot(win_array, label = "胜率")
    plt.plot(tie_array, label = "平局率")
    plt.plot(lose_array, label = "败率")
    plt.xticks(fontproperties='Times New Roman', size=20)
    plt.yticks(fontproperties='Times New Roman', size=20)
    plt.xlabel("轮次", fontdict={"family": "SimSun", "size": 25})
    plt.ylabel("胜负率", fontdict={"family": "SimSun", "size": 25})
    plt.legend(prop={'family': 'SimSun', 'size': 22})
    plt.grid()
    plt.margins(x=0)
    plt.savefig(os.path.join(fig_path, 'rate.png'))
    if only_fig is False:
        plt.show()
