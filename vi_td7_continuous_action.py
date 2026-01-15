# Original code from CleanRL and TD7 (MIT License)
# Modifications Copyright (c) 2025 Yaniv Oren et al.
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
from typing import Callable
import copy
from dmc2gymnasium import DMCGym
from dm_control import suite


@dataclass
class Args:
    results_path: str = "viac"
    """The path to the location where the results are to be saved"""
    exp_name: str = "vi_td7"
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
    q_lr: float = 3e-4 # this was in sac: 1e-3 and this was in the TD3 code: 3e-4
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    n: int = 1
    """number of actions sampled for to compute the sampling-based improved policy"""
    value_improvement_operator: str = "none"
    """options: 
            none: The baseline TD3.
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
    number_of_vi_gradient_updates: int = 5
    use_checkpoints: bool = True
    """If True, uses TD7's checkpointing"""
    expectile: float = 0.75


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


def execute_test_episode(agent, run_name, args):
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
    episode_returns = np.asarray([0 for _ in range(num_test_episodes)])

    # Interaction loop
    step_counter = 0
    while not all_dones.all():
        eval_actions = agent.select_action(np.array(eval_obs), use_exploration=not args.deterministic_evaluation)
        eval_obs, _, _, _, eval_infos = evaluation_envs.step(eval_actions)
        step_counter += 1

        if "final_info" in eval_infos:
            for index, is_final in enumerate(eval_infos["_final_info"]):
                if is_final and not all_dones[index]:
                    episode_returns[index] = eval_infos['final_info'][index]['episode']['r']
                    all_dones[index] = True

    evaluation_envs.close()
    return episode_returns


@dataclass
class Hyperparameters:
    # Generic
    batch_size: int = 256
    # buffer_size: int = 1e6
    discount: float = 0.99
    target_update_rate: int = 250
    exploration_noise: float = 0.1

    # TD3
    target_policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2

    # LAP
    alpha: float = 0.4
    min_priority: float = 1

    # TD3+BC
    lmbda: float = 0.1

    # Checkpointing
    max_eps_when_checkpointing: int = 20
    steps_before_checkpointing: int = 75e4
    reset_weight: float = 0.9

    # Encoder Model
    zs_dim: int = 256
    enc_hdim: int = 256
    enc_activ: Callable = F.elu
    encoder_lr: float = 3e-4

    # Critic Model
    critic_hdim: int = 256
    critic_activ: Callable = F.elu
    critic_lr: float = 3e-4

    # Actor Model
    actor_hdim: int = 256
    actor_activ: Callable = F.relu
    actor_lr: float = 3e-4


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


def LAP_huber(x, min_priority=1):
    return torch.where(x < min_priority, 0.5 * x.pow(2), min_priority * x).sum(1).mean()


def expectile_loss(diff, expectile):
    weight = torch.where(diff > 0, expectile, 1 - expectile)
    return weight * (diff**2)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.relu, env=None):
        super(Actor, self).__init__()

        self.activ = activ

        self.l0 = nn.Linear(state_dim, hdim)
        self.l1 = nn.Linear(zs_dim + hdim, hdim)
        self.l2 = nn.Linear(hdim, hdim)
        self.l3 = nn.Linear(hdim, action_dim)

        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, state, zs):
        a = AvgL1Norm(self.l0(state))
        a = torch.cat([a, zs], 1)
        a = self.activ(self.l1(a))
        a = self.activ(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a * self.action_scale + self.action_bias


class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, zs_dim=256, hdim=256, activ=F.elu):
        super(Critic, self).__init__()

        self.activ = activ

        self.q01 = nn.Linear(state_dim + action_dim, hdim)
        self.q1 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q2 = nn.Linear(hdim, hdim)
        self.q3 = nn.Linear(hdim, 1)

        self.q02 = nn.Linear(state_dim + action_dim, hdim)
        self.q4 = nn.Linear(2 * zs_dim + hdim, hdim)
        self.q5 = nn.Linear(hdim, hdim)
        self.q6 = nn.Linear(hdim, 1)

    def forward(self, state, action, zsa, zs):
        sa = torch.cat([state, action], 1)
        embeddings = torch.cat([zsa, zs], 1)

        q1 = AvgL1Norm(self.q01(sa))
        q1 = torch.cat([q1, embeddings], 1)
        q1 = self.activ(self.q1(q1))
        q1 = self.activ(self.q2(q1))
        q1 = self.q3(q1)

        q2 = AvgL1Norm(self.q02(sa))
        q2 = torch.cat([q2, embeddings], 1)
        q2 = self.activ(self.q4(q2))
        q2 = self.activ(self.q5(q2))
        q2 = self.q6(q2)
        return torch.cat([q1, q2], 1)


class Agent(object):
    def __init__(self, state_dim, action_dim, max_action, offline=False, hp=Hyperparameters(), replay_buffer=None,
                 args=None, env=None):
        # Changing hyperparameters example: hp=Hyperparameters(batch_size=128)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hp = hp
        self.args = args
        self.actor = Actor(state_dim, action_dim, hp.zs_dim, hp.actor_hdim, hp.actor_activ, env=env).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=hp.actor_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.critic = Critic(state_dim, action_dim, hp.zs_dim, hp.critic_hdim, hp.critic_activ).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=hp.critic_lr)
        self.critic_target = copy.deepcopy(self.critic)

        self.encoder = Encoder(state_dim, action_dim, hp.zs_dim, hp.enc_hdim, hp.enc_activ).to(self.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=hp.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.checkpoint_actor = copy.deepcopy(self.actor)
        self.checkpoint_encoder = copy.deepcopy(self.encoder)

        self.replay_buffer = replay_buffer

        self.max_action = max_action
        self.offline = offline

        self.training_steps = 0

        # Checkpointing tracked values
        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0
        self.min_target = 0

        # Env max and min action values
        self.env_min_action = torch.from_numpy(env.single_action_space.low).to(self.device)
        self.env_max_action = torch.from_numpy(env.single_action_space.high).to(self.device)

    def select_action(self, state, use_checkpoint=False, use_exploration=True):
        with torch.no_grad():
            # state = torch.tensor(state.reshape(1, -1), dtype=torch.float, device=self.device)
            state = torch.tensor(state, dtype=torch.float, device=self.device)

            if use_checkpoint:
                zs = self.checkpoint_encoder.zs(state)
                action = self.checkpoint_actor(state, zs)
            else:
                zs = self.fixed_encoder.zs(state)
                action = self.actor(state, zs)

            if use_exploration:
                action = action + torch.randn_like(action) * self.hp.exploration_noise

            return action.clamp(self.env_min_action, self.env_max_action).cpu().data.numpy() * self.max_action

    def train(self):
        self.training_steps += 1
        data = self.replay_buffer.sample(self.hp.batch_size)
        state = data.observations
        action = data.actions
        next_state = data.next_observations
        reward = data.rewards
        done = data.dones

        #########################
        # Update Encoder
        #########################
        with torch.no_grad():
            next_zs = self.encoder.zs(next_state)

        zs = self.encoder.zs(state)
        pred_zs = self.encoder.zsa(zs, action)
        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_optimizer.zero_grad()
        encoder_loss.backward()
        self.encoder_optimizer.step()

        #########################
        # Update Critic
        #########################
        with torch.no_grad():
            fixed_target_zs = self.fixed_encoder_target.zs(next_state)

            noise = (torch.randn_like(action) * self.hp.target_policy_noise).clamp(-self.hp.noise_clip,
                                                                                   self.hp.noise_clip)
            next_action = (self.actor_target(next_state, fixed_target_zs) + noise).clamp(self.env_min_action, self.env_max_action) * self.max_action

            fixed_target_zsa = self.fixed_encoder_target.zsa(fixed_target_zs, next_action)
            bootstrap = self.critic_target(next_state, next_action, fixed_target_zsa, fixed_target_zs).min(1, keepdim=True)[0]

            Q_target = reward + (1 - done) * self.hp.discount * bootstrap.clamp(self.min_target, self.max_target)

            self.max = max(self.max, float(Q_target.max()))
            self.min = min(self.min, float(Q_target.min()))

            fixed_zs = self.fixed_encoder.zs(state)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, action)

        Q = self.critic(state, action, fixed_zsa, fixed_zs)
        td_loss = (Q - Q_target).abs()
        if args.value_improvement_operator == "expectile":
            critic_loss = expectile_loss(Q_target - Q, args.expectile).mean()
        else:
            critic_loss = LAP_huber(td_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        #########################
        # Update Actor
        #########################
        if self.training_steps % self.hp.policy_freq == 0:
            actor = self.actor(state, fixed_zs)
            fixed_zsa = self.fixed_encoder.zsa(fixed_zs, actor)
            Q = self.critic(state, actor, fixed_zsa, fixed_zs)

            actor_loss = -Q.mean()
            if self.offline:
                actor_loss = actor_loss + self.hp.lmbda * Q.abs().mean().detach() * F.mse_loss(actor, action)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        else:
            actor_loss = None

        #########################
        # Update Iteration
        #########################
        if self.training_steps % self.hp.target_update_rate == 0:
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
            self.fixed_encoder_target.load_state_dict(self.fixed_encoder.state_dict())
            self.fixed_encoder.load_state_dict(self.encoder.state_dict())
            self.max_target = self.max
            self.min_target = self.min

        return Q, td_loss, critic_loss, actor_loss, bootstrap

    # If using checkpoints: run when each episode terminates
    def maybe_train_and_checkpoint(self, ep_timesteps, ep_return):
        self.eps_since_update += 1
        self.timesteps_since_update += ep_timesteps

        self.min_return = min(self.min_return, ep_return)

        # End evaluation of current policy early
        if self.min_return < self.best_min_return:
            Qs, td_losses, critic_losses, actor_losses = self.train_and_reset()

        # Update checkpoint
        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor.state_dict())
            self.checkpoint_encoder.load_state_dict(self.fixed_encoder.state_dict())

            Qs, td_losses, critic_losses, actor_losses = self.train_and_reset()
        else:
            Qs, td_losses, critic_losses, actor_losses = None, None, None, None

        return Qs, td_losses, critic_losses, actor_losses

    # Batch training
    def train_and_reset(self):
        Qs, td_losses, critic_losses, actor_losses, base_bootstraps, vi_bootstraps = [], [], [], [], [], []
        for _ in range(self.timesteps_since_update):
            if self.training_steps == self.hp.steps_before_checkpointing:
                self.best_min_return *= self.hp.reset_weight
                self.max_eps_before_update = self.hp.max_eps_when_checkpointing

            Q, td_loss, critic_loss, actor_loss, bootstrap = self.train()
            Qs.append(Q)
            td_losses.append(td_loss)
            critic_losses.append(critic_loss)
            actor_losses.append(actor_loss)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8
        Qs = torch.stack(Qs, dim=0)
        td_losses = torch.stack(td_losses, dim=0)
        critic_losses = torch.stack(critic_losses, dim=0)
        actor_losses = torch.stack([loss for loss in actor_losses if loss is not None], dim=0)
        return Qs, td_losses, critic_losses, actor_losses

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
    results_path = args.results_path + "/" + run_name
    writer = SummaryWriter(f"{args.results_path}/{run_name}")
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

    print(f"run_name = {run_name} \n"
          f"args.env_id = {args.env_id} \n"
          f"n = {args.n} \n"
          f"aggregator (which is really just value-target-type) = {args.value_improvement_operator} \n"
          f"Savings results to: {results_path} \n"
          # f"Clipping grad norm? {args.clip_gradient_norm}. If yes, max norm = {args.max_norm} \n"
          , flush=True)

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])
    state_dim = np.array(envs.single_observation_space.shape).prod()
    action_dim = np.prod(envs.single_action_space.shape)
    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=False,
    )
    agent = Agent(state_dim, action_dim, max_action, offline=False, replay_buffer=rb, args=args, env=envs)
    start_time = time.time()

    # Arrays to save results locally
    evaluation_episodes_mean = []
    evaluation_episodes_steps = []

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        if global_step > 0 and args.num_evaluation_episodes > 0 and global_step % args.evaluation_frequency == 0:
            eval_episode_returns = execute_test_episode(agent, run_name, args=args)
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

            print(f"%%%%% Evaluation episode. %%%%%\n"
                  f"global_step={global_step}, max_test_return = {max_test_return}, "
                  f"mean_test_return = {mean_test_return}, std_test_return = {std_test_return} \n"
                  f"%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = agent.select_action(np.array(obs))

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        agent.replay_buffer.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if not args.use_checkpoints:
                Q, td_loss, critic_loss, actor_loss, base_bootstrap, vi_bootstrap = agent.train()
            if "final_info" in infos:
                if args.use_checkpoints:
                    episode_length = infos["final_info"][0]["episode"]["l"][0]
                    episode_return = infos["final_info"][0]['episode']['r'][0]
                    Q, td_loss, critic_loss, actor_loss = agent.maybe_train_and_checkpoint(episode_length, episode_return)

            if (global_step % 100 == 0 and not args.use_checkpoints) or \
                    ("final_info" in infos and args.use_checkpoints and Q is not None):

                writer.add_scalar("losses/qf1_values", Q[...,0].mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", Q[...,1].mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", td_loss[..., 0].mean().item(), global_step)
                writer.add_scalar("losses/qf2_loss", td_loss[..., 1].mean().item(), global_step)
                writer.add_scalar("losses/qf_loss", critic_loss.mean().item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.mean().item(), global_step)

                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                percent_chosen_max = 0

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Completed training. \n"
          f"Total runtime: {elapsed // (60 * 60)} hours, {elapsed // 60} minutes, {elapsed % 60} seconds \n"
          f"Number of timesteps: {args.total_timesteps}", flush=True)

    envs.close()
    writer.close()
