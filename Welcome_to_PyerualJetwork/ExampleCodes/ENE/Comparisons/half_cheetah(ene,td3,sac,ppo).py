import gym
import numpy as np
import matplotlib.pyplot as plt
from pyerualjetwork import ene, model_ops
from pyerualjetwork.cpu import data_ops
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter
import os
from dataclasses import dataclass
from multiprocessing import Pool, cpu_count

# --- ENE: ---


def evaluate_individual(args):
    individual_idx, W, A, seeds = args
    env = gym.make('HalfCheetah-v4')
    env.model.opt.timestep = 0.003
    total_reward = 0
    model = model_ops.build_model(W=W, activations=A, model_type='MLP')

    for seed in seeds:
        state = env.reset()[0]
        done = truncated = False
        reward_sum = 0

        while not done and not truncated:
            action = model_ops.predict_from_memory(state, model)
            action = data_ops.normalization(action, dtype=action.dtype)
            state, reward, done, truncated, _ = env.step(action)

            reward_sum += reward
        total_reward += reward_sum

    env.close()
    return total_reward / len(seeds)


def main():

    input_shape = 17
    output_shape = 6
    population_size = 50

    genome_weights, genome_activations = ene.define_genomes(input_shape, output_shape, population_size, neurons=[256, 256], activation_functions=['tanh', 'tanh'])

    rewards = [0] * population_size
    max_rewards = []
    reward_sum = 0

    generation = 0
    max_generation = 100

    time_stamps_1 = []
    max_rewards_1 = []

    start_time_1 = time.time()

    seeds = [random.randint(0, 2**32 - 1) for _ in range(1)]

    # Training Loop Starts
    for generation in range(max_generation):

        args_list = [
                (i, genome_weights[i], genome_activations[i], seeds)
                for i in range(population_size)
            ]

        with Pool(processes=cpu_count()) as pool:
            rewards = pool.map(evaluate_individual, args_list)


        # Evrim mekanizmasını çalıştır
        genome_weights, genome_activations = ene.evolver(
            genome_weights, genome_activations, generation, 
            np.array(rewards, dtype=np.float32), show_info=True, is_mlp=True, activation_mutate_prob=0
        )
        max_rewards_1.append(max(rewards))
        

    
        # Geçen süreyi kaydet (saniye cinsinden)
        elapsed_time = time.time() - start_time_1
        time_stamps_1.append(elapsed_time)



    # --- TD3: ---



    gamma = 0.99
    learning_rate = 1e-3
    buffer_size = 100000
    batch_size = 64
    tau = 0.005  
    policy_noise = 0.2
    noise_clip = 0.5
    policy_delay = 2  
    episodes = 200

    class Actor(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, output_dim)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return torch.tanh(self.fc3(x))

    class Critic(nn.Module):
        def __init__(self, input_dim, action_dim):
            super(Critic, self).__init__()
            self.fc1 = nn.Linear(input_dim + action_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, 1)

            self.fc4 = nn.Linear(input_dim + action_dim, 256)
            self.fc5 = nn.Linear(256, 256)
            self.fc6 = nn.Linear(256, 1)
        
        def forward(self, state, action):
            sa = torch.cat([state, action], dim=1)

            q1 = torch.tanh(self.fc1(sa))
            q1 = torch.tanh(self.fc2(q1))
            q1 = self.fc3(q1)

            q2 = torch.tanh(self.fc4(sa))
            q2 = torch.tanh(self.fc5(q2))
            q2 = self.fc6(q2)

            return q1, q2

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)
        
        def __len__(self):
            return len(self.buffer)

    def select_action(state, actor, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = actor(state).cpu().numpy().flatten()
        action += np.random.normal(0, noise, size=action.shape)
        return action

    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    env = gym.make('HalfCheetah-v4')
    env.model.opt.timestep = 0.003
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    actor = Actor(input_dim, output_dim)
    critic = Critic(input_dim, output_dim)
    actor_target = Actor(input_dim, output_dim)
    critic_target = Critic(input_dim, output_dim)

    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    time_stamps_2 = []
    max_rewards_2 = []

    start_time_2 = time.time()

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        step = 0

        while True:
            action = select_action(state, actor, noise=0.1)
            action = data_ops.normalization(action, dtype=action.dtype)
            next_state, reward, done, truncated, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done or truncated:
                break

            if len(replay_buffer) > batch_size:
                batch = replay_buffer.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)

                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                next_actions = actor_target(next_states)
                noise = torch.clamp(torch.randn_like(next_actions) * policy_noise, -noise_clip, noise_clip)
                next_actions = next_actions + noise

                q1_target, q2_target = critic_target(next_states, next_actions)
                q_target = rewards + gamma * torch.min(q1_target, q2_target) * (1 - dones)

                q1, q2 = critic(states, actions)
                critic_loss = nn.MSELoss()(q1, q_target.detach()) + nn.MSELoss()(q2, q_target.detach())

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                if step % policy_delay == 0:
                    actor_loss = -critic(states, actor(states))[0].mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    soft_update(actor_target, actor, tau)
                    soft_update(critic_target, critic, tau)

            step += 1

        elapsed_time = time.time() - start_time_2
        time_stamps_2.append(elapsed_time)
        max_rewards_2.append(total_reward)

        print(f"Episode {episode + 1}, Max Reward: {total_reward:.2f}")

    env.close()



    # --- SAC: ---



    gamma = 0.99
    learning_rate = 1e-3
    buffer_size = 100000
    batch_size = 64
    tau = 0.005
    alpha = 0.2
    episodes = 200

    class Actor(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(Actor, self).__init__()
            self.fc1 = nn.Linear(input_dim, 256)
            self.fc2 = nn.Linear(256, 256)
            self.fc3 = nn.Linear(256, output_dim)
            self.log_std_layer = nn.Linear(256, output_dim)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            mean = self.fc3(x)
            log_std = torch.clamp(self.log_std_layer(x), -20, 2)
            std = torch.exp(log_std)
            return mean, std

        def sample(self, state):
            mean, std = self.forward(state)
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()
            action = torch.tanh(z)
            log_prob = normal.log_prob(z)
            log_prob = log_prob.sum(dim=-1, keepdim=True)
            epsilon = 1e-6
            log_prob -= torch.sum(torch.log(1 - action.pow(2) + epsilon), dim=-1, keepdim=True)
            return action, log_prob

        def get_action(self, state, deterministic=True):
            state = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                if deterministic:
                    mean, _ = self.forward(state)
                    action = torch.tanh(mean)
                else:
                    action, _ = self.sample(state)
            return action.cpu().numpy().flatten()

    class Critic(nn.Module):
        def __init__(self, input_dim, action_dim):
            super(Critic, self).__init__()
            self.q1 = self._build_network(input_dim, action_dim)
            self.q2 = self._build_network(input_dim, action_dim)
        
        def _build_network(self, input_dim, action_dim):
            return nn.Sequential(
                nn.Linear(input_dim + action_dim, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
        
        def forward(self, state, action):
            sa = torch.cat([state, action], dim=1)
            return self.q1(sa), self.q2(sa)

    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        
        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, reward, next_state, done))
        
        def sample(self, batch_size):
            batch = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
            return states, actions, rewards, next_states, dones
        
        def __len__(self):
            return len(self.buffer)

    def soft_update(target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    env = gym.make('HalfCheetah-v4')
    env.model.opt.timestep = 0.003
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    actor = Actor(input_dim, output_dim)
    critic = Critic(input_dim, output_dim)
    critic_target = Critic(input_dim, output_dim)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
    replay_buffer = ReplayBuffer(buffer_size)

    time_stamps_4 = []
    max_rewards_4 = []
    start_time = time.time()

    for episode in range(episodes):
        state = env.reset()[0]
        total_reward = 0
        step = 0
        while True:
            action = actor.get_action(state, deterministic=False)
            action = data_ops.normalization(action, dtype=action.dtype)
            next_state, reward, done, truncated, _ = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            if done or truncated:
                break

            if len(replay_buffer) > batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
                
                states = torch.FloatTensor(states)
                actions = torch.FloatTensor(actions)
                rewards = torch.FloatTensor(rewards).unsqueeze(1)
                next_states = torch.FloatTensor(next_states)
                dones = torch.FloatTensor(dones).unsqueeze(1)

                with torch.no_grad():
                    next_actions, next_log_probs = actor.sample(next_states)
                    q1_target, q2_target = critic_target(next_states, next_actions)
                    q_target_min = torch.min(q1_target, q2_target)
                    q_target = rewards + gamma * (1 - dones) * (q_target_min - alpha * next_log_probs)
                
                q1, q2 = critic(states, actions)
                critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actions_new, log_probs = actor.sample(states)
                q1_new, q2_new = critic(states, actions_new)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_probs - q_new).mean()
                
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                soft_update(critic_target, critic, tau)
            step += 1
        
        elapsed_time = time.time() - start_time
        time_stamps_4.append(elapsed_time)
        max_rewards_4.append(total_reward)
        
        print(f"Episode {episode + 1}, Max Reward: {total_reward:.2f}")




    # --- PPO: ---



    # docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy

    @dataclass
    class Args:
        exp_name: str = os.path.basename(__file__)[: -len(".py")]
        """the name of this experiment"""
        seed: int = 1
        """seed of the experiment"""
        torch_deterministic: bool = True
        """if toggled, `torch.backends.cudnn.deterministic=False`"""
        cuda: bool = False
        """if toggled, cuda will be enabled by default"""
        track: bool = False
        """if toggled, this experiment will be tracked with Weights and Biases"""
        wandb_project_name: str = "cleanRL"
        """the wandb's project name"""
        wandb_entity: str = None
        """the entity (team) of wandb's project"""
        capture_video: bool = False
        """whether to capture videos of the agent performances (check out `videos` folder)"""
        save_model: bool = False
        """whether to save model into the `runs/{run_name}` folder"""
        upload_model: bool = False
        """whether to upload the saved model to huggingface"""
        hf_entity: str = ""
        """the user or org name of the model repository from the Hugging Face Hub"""

        # Algorithm specific arguments
        env_id: str = "HalfCheetah-v4"
        """the id of the environment"""
        total_timesteps: int = 1000000
        """total timesteps of the experiments"""
        learning_rate: float = 3e-4
        """the learning rate of the optimizer"""
        num_envs: int = 1
        """the number of parallel game environments"""
        num_steps: int = 2048
        """the number of steps to run in each environment per policy rollout"""
        anneal_lr: bool = True
        """Toggle learning rate annealing for policy and value networks"""
        gamma: float = 0.99
        """the discount factor gamma"""
        gae_lambda: float = 0.95
        """the lambda for the general advantage estimation"""
        num_minibatches: int = 32
        """the number of mini-batches"""
        update_epochs: int = 10
        """the K epochs to update the policy"""
        norm_adv: bool = True
        """Toggles advantages normalization"""
        clip_coef: float = 0.2
        """the surrogate clipping coefficient"""
        clip_vloss: bool = True
        """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
        ent_coef: float = 0.0
        """coefficient of the entropy"""
        vf_coef: float = 0.5
        """coefficient of the value function"""
        max_grad_norm: float = 0.5
        """the maximum norm for the gradient clipping"""
        target_kl: float = None
        """the target KL divergence threshold"""

        # to be filled in runtime
        batch_size: int = 0
        """the batch size (computed in runtime)"""
        minibatch_size: int = 0
        """the mini-batch size (computed in runtime)"""
        num_iterations: int = 0
        """the number of iterations (computed in runtime)"""


    def make_env(env_id, idx, capture_video, run_name, gamma):
        def thunk():
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ClipAction(env)
            env = gym.wrappers.NormalizeObservation(env)
            env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
            env = gym.wrappers.NormalizeReward(env, gamma=gamma)
            env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
            return env

        return thunk


    def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    class Agent(nn.Module):
        def __init__(self, envs):
            super().__init__()
            self.critic = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 1), std=1.0),
            )
            self.actor_mean = nn.Sequential(
                layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, 64)),
                nn.Tanh(),
                layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
            )
            self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

        def get_value(self, x):
            return self.critic(x)

        def get_action_and_value(self, x, action=None):
            action_mean = self.actor_mean(x)
            action_logstd = self.actor_logstd.expand_as(action_mean)
            action_std = torch.exp(action_logstd)
            probs = Normal(action_mean, action_std)
            if action is None:
                action = probs.sample()
            return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


    if __name__ == "__main__":
        args = tyro.cli(Args)
        args.batch_size = int(args.num_envs * args.num_steps)
        args.minibatch_size = int(args.batch_size // args.num_minibatches)
        args.num_iterations = args.total_timesteps // args.batch_size
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
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
        writer = SummaryWriter(f"runs/{run_name}")
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cpu")

        # env setup
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
        )
        time_stamps_5 = []
        max_rewards_5 = []
        start_time = time.time()
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

        agent = Agent(envs).to(device)
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

        for iteration in range(1, args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / args.num_iterations
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, args.num_steps):
                global_step += args.num_envs
                obs[step] = next_obs
                dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.flatten()
                actions[step] = action
                logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                rewards[step] = torch.tensor(reward).to(device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

                if "final_info" in infos:
                    for info in infos["final_info"]:
                        if info and "episode" in info:
                            print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                            elapsed_time = time.time() - start_time
                            time_stamps_5.append(elapsed_time)
                            max_rewards_5.append(info['episode']['r'])
                            writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                            writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + values

            # flatten the batch
            b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
            b_logprobs = logprobs.reshape(-1)
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(args.batch_size)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(b_inds)
                for start in range(0, args.batch_size, args.minibatch_size):
                    end = start + args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if args.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -args.clip_coef,
                            args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            print("SPS:", int(global_step / (time.time() - start_time)))
            writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        envs.close()
        writer.close()

    plt.figure(figsize=(10, 5))
    plt.plot(time_stamps_1, max_rewards_1, marker='o', linestyle='-', color='g', label="ENE (2 Hyperparameters Tuned)")
    plt.plot(time_stamps_2, max_rewards_2, marker='o', linestyle='-', color='r', label="TD3 (13 Hyperparameters Tuned)")
    plt.plot(time_stamps_4, max_rewards_4, marker='o', linestyle='-', color='y', label="SAC (11 Hyperparameters Tuned)")
    plt.plot(time_stamps_5, max_rewards_5, marker='o', linestyle='-', color='c', label="PPO (17 Hyperparameters Tuned)")
    plt.xlabel("Time(seconds) [Lower Better]")
    plt.ylabel("Max Reward [Higher Better]")
    plt.title("HalfCheetah-v4")
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn')
    main()