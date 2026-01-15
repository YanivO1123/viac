# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import pathlib
import traceback
from dmc2gymnasium import DMCGym
from dm_control import suite


@dataclass
class Args:
    results_path: str = "viac"
    """The path to the location where the results are to be saved"""
    exp_name: str = "vi_sac"
    """the name of this experiment"""
    run_name: str = None
    """the full name of this run"""
    seed: int = None
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "viac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "hopper-hop"
    """the environment id of the task"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    n: int = 1
    """number of actions sampled for to compute the sampling-based improved policy"""
    value_improvement_operator: str = "none"
    """options: 
            none: The baseline TD3.
            expectile: Assymetric loss on the critic.
        """
    evaluation_frequency: int = 5000
    """How often to run evaluation episodes, in environment interactions. Only runs eval episodes if num_evaluation_episodes > 0
    """
    num_evaluation_episodes: int = 3
    """Number of evaluation episodes to evalaute over.
    """
    deterministic_evaluation: bool = True
    """Whether to evaluate deterministically or not
    """
    number_of_vi_gradient_updates: int = 0
    expectile: float = 0.75
    """The expectile to be used for critic optimization"""
    logging_frequency: int = 1000


def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff ** 2)


def is_dm_control_env(env_name):
    """Check if the environment name belongs to dm_control"""
    domain, task = env_name.split('-')
    all_dm_control_envs = suite.ALL_TASKS
    return (domain, task) in all_dm_control_envs


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if is_dm_control_env(env_id):
            domain, task = env_id.split('-')
            env = DMCGym(domain, task)
        else:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


def execute_test_episode(eval_actor, eval_device, run_name, args, greedy=False):
    # Setup
    num_test_episodes = args.num_evaluation_episodes
    seeds = []
    for i in range(num_test_episodes):
        low = 0
        high = 10000000
        random_seed = np.random.randint(low, high=high)
        seeds.append(random_seed)
    evaluation_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, seeds[i], 0, args.capture_video, run_name)
                                                for i in range(num_test_episodes)])
    all_dones = np.array([False for _ in range(num_test_episodes)])
    eval_obs, _ = evaluation_envs.reset()
    ep_max_len = 10000
    episode_returns = np.asarray([0 for _ in range(num_test_episodes)])
    rewards = np.zeros((ep_max_len, num_test_episodes))
    qs = np.zeros((ep_max_len, num_test_episodes))
    filled = np.ones((ep_max_len, num_test_episodes), dtype=bool)

    to_th = lambda tnsr: torch.Tensor(tnsr).to(eval_device)
    q_avg_fn = lambda o, a: (0.5 * (qf1(to_th(o), to_th(a)) + qf2(to_th(o), to_th(a))).detach().cpu().numpy().squeeze())

    step_counter = 0
    while not all_dones.all():
        sampled_actions, _log_prob, mean_actions = eval_actor.get_action(torch.Tensor(eval_obs).to(eval_device))

        if args.deterministic_evaluation:
            eval_actions = mean_actions
        else:
            eval_actions = sampled_actions

        eval_actions = eval_actions.detach().cpu().numpy()
        eval_obs, r, _, _, eval_infos = evaluation_envs.step(eval_actions)
        rewards[step_counter] = r
        qs[step_counter] = q_avg_fn(eval_obs, eval_actions)
        filled[step_counter] = filled[step_counter] * filled[step_counter - 1]

        step_counter += 1
        if "final_info" in eval_infos:
            for index, is_final in enumerate(eval_infos["_final_info"]):
                if is_final and not all_dones[index]:
                    episode_returns[index] = eval_infos['final_info'][index]['episode']['r']
                    all_dones[index] = True
                    filled[step_counter, index] = 0

    episode_disc_returns = np.zeros((step_counter + 1, num_test_episodes))
    for t in reversed(range(step_counter)):
        episode_disc_returns[t] = args.gamma * episode_disc_returns[t + 1] + rewards[t]
    episode_biases = (qs[:step_counter] - episode_disc_returns[:-1])
    bias_avg = episode_biases[filled[:step_counter]].mean()
    bias_var = episode_biases[filled[:step_counter]].var()
    evaluation_envs.close()
    return episode_returns, dict(bias_avg=bias_avg, bias_var=bias_var)


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

LOG_STD_MAX = 2
LOG_STD_MIN = -5

class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)

    if args.seed is None:
        low = 0
        high = 10000000
        seed = np.random.randint(low, high=high)
        args.seed = seed

    if args.run_name is None:
        run_name = f"{args.exp_name}_{args.env_id}_{args.value_improvement_operator}_{args.seed}_{time.asctime(time.localtime(time.time()))}"
    else:
        run_name = args.run_name

    try:
        pathlib.Path(f"{args.results_path}/{run_name}").mkdir(parents=True, exist_ok=True)
    except:
        traceback.print_exc()

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    results_path = f"{args.results_path}/{run_name}"
    writer = SummaryWriter(results_path)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    # Arrays to save results locally
    evaluation_episodes_mean = []
    evaluation_episodes_steps = []
    running_diagnostics = dict()

    average_base_bootstraps = []
    averaged_vi_bootstraps = []
    averaged_bootstrap_differences = []
    bootstrap_steps = []
    expectile = args.expectile

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step > 0 and args.num_evaluation_episodes > 0 and global_step % args.evaluation_frequency == 0:
            eval_episode_returns, diagnostics = execute_test_episode(actor, device, run_name, args=args)
            mean_test_return = np.mean(eval_episode_returns)
            max_test_return = np.max(eval_episode_returns)
            min_test_return = np.min(eval_episode_returns)
            std_test_return = np.std(eval_episode_returns)
            writer.add_scalar("charts/test_mean_return", mean_test_return, global_step)
            writer.add_scalar("charts/test_max_return", max_test_return, global_step)
            writer.add_scalar("charts/test_min_return", min_test_return, global_step)
            writer.add_scalar("charts/test_std_return", std_test_return, global_step)
            evaluation_episodes_mean.append(mean_test_return)
            evaluation_episodes_steps.append(global_step)
            np.save(results_path + "/evaluation_episodes_mean", np.asarray(evaluation_episodes_mean))
            np.save(results_path + "/evaluation_episodes_steps", np.asarray(evaluation_episodes_steps))
            for k, v, in diagnostics.items():
                writer.add_scalar(f"charts/{k}", v, global_step)
                running_diagnostics[k] = running_diagnostics.get(k, []) + [v, ]
                np.save(f"{results_path}/{k}", np.array(running_diagnostics[k]))

            print(f"%%%%% Evaluation episode. %%%%%\n"
                  f"global_step={global_step}, max_test_return = {max_test_return}, "
                  f"mean_test_return = {mean_test_return}, std_test_return = {std_test_return} \n"
                      f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)

            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                    min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)

            if args.value_improvement_operator == "expectile":
                qf1_loss = expectile_loss(next_q_value - qf1_a_values, expectile).mean()
                qf2_loss = expectile_loss(next_q_value - qf2_a_values, expectile).mean()
            else:
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)

            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                        args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % args.logging_frequency == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                if args.value_improvement_operator == "min_unc":
                    writer.add_scalar("charts/percent_chosen_max", percent_chosen_max / 100, global_step)
                    print(f"avg. percent_chosen_max: {percent_chosen_max / 100}")
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                percent_chosen_max = 0

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Completed training. \n"
          f"Total runtime: {elapsed // (60 * 60)} hours, {elapsed // 60} minutes, {elapsed % 60} seconds \n"
          f"Number of timesteps: {args.total_timesteps}", flush=True)

    envs.close()
    writer.close()
