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
    exp_name: str = "vi_td3"
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
    q_lr: float = 3e-4  # this was in sac: 1e-3 and this was in the TD3 code: 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    n: int = 1
    """number of actions sampled for to compute the sampling-based improved policy"""
    value_improvement_operator: str = "none"
    """options: 
            none: The baseline TD3.
            policy_gradient: TD3's deterministic policy gradient.
            sampled_argmax: Samples n actions, takes the action that has the maximum min(q_1,q_2) as the improved policy
            expectile: Assymetric loss on the critic.
        """
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    policy_noise: float = 0.2
    """the scale of policy noise"""
    evaluation_frequency: int = 5000
    """How often to run evaluation episodes, in environment interactions. Only runs eval episodes if num_evaluation_episodes > 0
    """
    num_evaluation_episodes: int = 3
    """Number of evaluation episodes to evalaute over.
    """
    deterministic_evaluation: bool = True
    """Whether to evaluate deterministically or not
    """
    number_of_vi_gradient_updates: int = 1
    expectile: float = 0.75
    """The expectile to be used for critic optimization"""
    logging_frequency: int = 1000
    update_ratio: int = 1
    """The update to data ratio, = number of updates per each interaction with the environment. 
    Defaults (and tuned) to 1. Ints >= 1"""
    number_of_pi_gradient_updates: int = 1
    """For ablations, runs the policy-improvement this number of time over the same batch"""


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


def update_policy_on_next_states(actor_to_update, critic_1, local_optimizer, data):
    with torch.enable_grad():
        local_actor_loss = -critic_1(data.observations, actor_to_update(data.observations)).mean()
        local_optimizer.zero_grad()
        local_actor_loss.backward()
        local_optimizer.step()


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
        eval_actions = eval_actor(torch.Tensor(eval_obs).to(eval_device))
        if greedy:
            normal = torch.distributions.Normal(torch.zeros_like(eval_actions), torch.ones_like(eval_actions))
            clipped_noise = (normal.rsample(sample_shape=(args.n,)) * args.policy_noise).clamp(
                -args.noise_clip, args.noise_clip
            ) * target_actor.action_scale

            n_eval_actions = (eval_actor(torch.Tensor(eval_obs).to(eval_device)) + clipped_noise).clamp(
                evaluation_envs.single_action_space.low[0], evaluation_envs.single_action_space.high[0])

            obs_n_fold = torch.Tensor(eval_obs).to(device).unsqueeze(0).repeat(args.n, 1, 1).flatten(0, 1)
            flattened_n_eval_actions = n_eval_actions.flatten(0, 1)

            q1 = qf1(obs_n_fold, flattened_n_eval_actions).view(args.n, args.num_evaluation_episodes)
            q2 = qf2(obs_n_fold, flattened_n_eval_actions).view(args.n, args.num_evaluation_episodes)

            max_action_indexes = (q1 + q2).argmax(dim=0).squeeze()
            eval_actions = torch.zeros(size=(num_test_episodes, eval_actions.shape[-1]))
            for i in range(num_test_episodes):
                eval_actions[i] = n_eval_actions[max_action_indexes[i], i]
        elif not args.deterministic_evaluation:
            normal = torch.distributions.Normal(torch.zeros_like(data.eval_actions), torch.ones_like(data.eval_actions))
            clipped_noise = (normal.rsample() * args.policy_noise).clamp(
                -args.noise_clip, args.noise_clip
            ) * target_actor.action_scale
            eval_actions = eval_actions + clipped_noise

        eval_actions = eval_actions.clamp(
            envs.single_action_space.low[0], envs.single_action_space.high[0]
        )
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
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
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
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


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

    assert args.update_ratio >= 1, f"update_ratio has to be greater or equal to 1."

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
    qf1 = QNetwork(envs).to(device)
    qf2 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    qf2_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    if args.value_improvement_operator == "policy_gradient":
        vi_update_actor = Actor(envs).to(device)
        vi_update_actor_optimizer = optim.Adam(list(vi_update_actor.parameters()), lr=args.policy_lr)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    percent_chosen_max = 0

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
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

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
            for update in range(args.update_ratio):
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    # Compute the original target
                    clipped_noise_base = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                        -args.noise_clip, args.noise_clip
                    ) * target_actor.action_scale
                    det_next_state_actions_base = (target_actor(data.next_observations)).clamp(
                        envs.single_action_space.low[0], envs.single_action_space.high[0]
                    )
                    sampled_next_state_actions_base = (target_actor(data.next_observations) + clipped_noise_base).clamp(
                        envs.single_action_space.low[0], envs.single_action_space.high[0]
                    )
                    # Sampled base bootstrap
                    qf1_next_target_base = qf1_target(data.next_observations, sampled_next_state_actions_base)
                    qf2_next_target_base = qf2_target(data.next_observations, sampled_next_state_actions_base)
                    min_qf_next_target_base_sampled = torch.min(qf1_next_target_base, qf2_next_target_base)

                    # Det. base bootstrap
                    qf1_next_target_base = qf1_target(data.next_observations, det_next_state_actions_base)
                    qf2_next_target_base = qf2_target(data.next_observations, det_next_state_actions_base)
                    min_qf_next_target_base_det = torch.min(qf1_next_target_base, qf2_next_target_base)
                    max_policy_value = None

                    if not "gaussian" in args.value_improvement_operator and not "policy_gradient" in args.value_improvement_operator:
                        normal = torch.distributions.Normal(torch.zeros_like(data.actions),
                                                            torch.ones_like(data.actions))
                        clipped_noise = (normal.rsample(sample_shape=(args.n,)) * args.policy_noise).clamp(
                            -args.noise_clip, args.noise_clip
                        ) * target_actor.action_scale

                        next_state_N_actions = (target_actor(data.next_observations) + clipped_noise).clamp(
                            envs.single_action_space.low[0], envs.single_action_space.high[0])

                        next_states_Nfold = data.next_observations[None].repeat(args.n, 1, 1).flatten(0, 1)
                        next_state_N_actions = next_state_N_actions.flatten(0, 1)  # flatten to batchdim

                        qf1_next_target = qf1_target(next_states_Nfold, next_state_N_actions).view(args.n,
                                                                                                   args.batch_size, 1)
                        qf2_next_target = qf2_target(next_states_Nfold, next_state_N_actions).view(args.n,
                                                                                                   args.batch_size, 1)

                    if args.value_improvement_operator == "none" or args.value_improvement_operator == "expectile":
                        min_qf_next_target = torch.min(qf1_next_target,
                                                       qf2_next_target)  # - alpha * next_state_K_log_pis
                        min_qf_next_target = min_qf_next_target.mean(dim=0)
                    elif args.value_improvement_operator == "sampled_argmax":
                        # Sample n actions
                        action_noise_normal = torch.distributions.Normal(torch.zeros_like(data.actions),
                                                                         torch.ones_like(data.actions))

                        n_actions_noise = (action_noise_normal.sample(
                            sample_shape=(args.n,)) * args.policy_noise). \
                                              clamp(-args.noise_clip, args.noise_clip) * target_actor.action_scale
                        n_new_action_means = (
                                target_actor(data.next_observations) + n_actions_noise).clamp(
                            envs.single_action_space.low[0], envs.single_action_space.high[0]
                        ).flatten(0, 1)

                        # Take their q values
                        next_states_Nfold = data.next_observations[None].repeat(args.n, 1, 1).flatten(0, 1)
                        qf1_next_target = qf1_target(next_states_Nfold, n_new_action_means).view(args.n,
                                                                                                 args.batch_size,
                                                                                                 1)
                        qf2_next_target = qf2_target(next_states_Nfold, n_new_action_means).view(args.n,
                                                                                                 args.batch_size,
                                                                                                 1)
                        min_qs = torch.min(qf1_next_target, qf2_next_target)

                        # Find the maximizing action
                        max_policy_value, max_policy_index = min_qs.max(dim=0)
                        max_policy_index = max_policy_index[None].repeat(1, 1, envs.single_action_space.shape[-1])

                        improved_policy_mean = torch.gather(
                            n_new_action_means.view(args.n, args.batch_size, data.actions.shape[-1]),
                            dim=0,
                            index=max_policy_index).view(args.batch_size, data.actions.shape[-1])

                        # For debugging, compute the value of this action

                        # Resampled from the maximizing action
                        new_action_from_improved_policy = (improved_policy_mean +
                                                           (
                                                                   (
                                                                           action_noise_normal.sample() * args.policy_noise)
                                                                   .clamp(-args.noise_clip, args.noise_clip)
                                                                   * target_actor.action_scale
                                                           )
                                                           ).view(
                            args.batch_size,
                            data.actions.shape[-1]).clamp(
                            envs.single_action_space.low[0], envs.single_action_space.high[0])
                        # Get the q values of the resampled action
                        qf1_next_target = qf1_target(data.next_observations, new_action_from_improved_policy)
                        qf2_next_target = qf2_target(data.next_observations, new_action_from_improved_policy)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    elif args.value_improvement_operator == "policy_gradient":
                        # Update the viac-actor and optimizer state dicts
                        vi_update_actor.load_state_dict(target_actor.state_dict())
                        vi_update_actor_optimizer.load_state_dict(actor_optimizer.state_dict())
                        for _ in range(args.number_of_vi_gradient_updates):
                            # update the viac actor number_of_vi_gradient_updates times
                            update_policy_on_next_states(vi_update_actor, qf1_target,
                                                         vi_update_actor_optimizer,
                                                         data)

                        # Sample action from the improved actor
                        # First we sample noise
                        clipped_noise = (torch.randn_like(data.actions, device=device) * args.policy_noise).clamp(
                            -args.noise_clip, args.noise_clip
                        ) * target_actor.action_scale
                        # Then we compute the action
                        next_state_actions = (vi_update_actor(data.next_observations) + clipped_noise).clamp(
                            envs.single_action_space.low[0], envs.single_action_space.high[0]
                        )
                        qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                        qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target)
                    else:
                        raise ValueError(
                            f"Unrecognized value improvement operator. Was {args.value_improvement_operator} and expecting either ")

                    next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (
                        min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                if args.value_improvement_operator == "expectile":
                    qf1_loss = expectile_loss(next_q_value - qf1_a_values, expectile).mean()
                    qf2_loss = expectile_loss(next_q_value - qf2_a_values, expectile).mean()
                    qf_loss = qf1_loss + qf2_loss
                else:
                    qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                    qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                    qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()

                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    policy_training_observations = data.observations
                    for pi_update in range(args.number_of_pi_gradient_updates):
                        actor_loss = -qf1(policy_training_observations, actor(policy_training_observations)).mean()
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        actor_optimizer.step()

                    # update the target network
                    for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                        target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
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

                base_bootstrap = min_qf_next_target_base_sampled.mean().item()
                det_base_bootstrap = min_qf_next_target_base_det.mean().item()
                vi_bootstrap = min_qf_next_target.mean().item()
                if max_policy_value is not None:
                    mean_improved_policy_value = max_policy_value.mean().item()
                    writer.add_scalar("losses/det. bootstrap difference",
                                      mean_improved_policy_value - det_base_bootstrap, global_step)
                    writer.add_scalar("losses/mean_improved_policy_value-vi_bootstrap",
                                      mean_improved_policy_value - vi_bootstrap, global_step)
                writer.add_scalar("losses/mean_base_bootstrap", base_bootstrap, global_step)
                writer.add_scalar("losses/mean_vi_bootstrap", vi_bootstrap, global_step)
                writer.add_scalar("losses/vi_bootstrap-base_bootstrap", vi_bootstrap - base_bootstrap, global_step)
                writer.add_scalar("losses/det_base_bootstrap-base_bootstrap", det_base_bootstrap - base_bootstrap,
                                  global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                if args.value_improvement_operator == "min_unc":
                    writer.add_scalar("charts/percent_chosen_max", percent_chosen_max / 100, global_step)
                    print(f"avg. percent_chosen_max: {percent_chosen_max / 100}")
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                percent_chosen_max = 0

                if global_step % args.evaluation_frequency == 0:
                    # Update arrays
                    average_base_bootstraps.append(base_bootstrap)
                    averaged_vi_bootstraps.append(vi_bootstrap)
                    averaged_bootstrap_differences.append(vi_bootstrap - base_bootstrap)
                    bootstrap_steps.append(global_step)

                    # Save results
                    np.save(results_path + "/average_baseline_bootstraps", np.asarray(average_base_bootstraps))
                    np.save(results_path + "/averaged_vi_bootstraps", np.asarray(averaged_vi_bootstraps))
                    np.save(results_path + "/averaged_bootstrap_differences",
                            np.asarray(averaged_bootstrap_differences))
                    np.save(results_path + "/bootstrap_steps", np.asarray(bootstrap_steps))

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Completed training. \n"
          f"Total runtime: {elapsed // (60 * 60)} hours, {elapsed // 60} minutes, {elapsed % 60} seconds \n"
          f"Number of timesteps: {args.total_timesteps}", flush=True)

    envs.close()
    writer.close()
