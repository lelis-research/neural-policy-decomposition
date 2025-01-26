import random
import time
import pickle

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


from combo import Game
from combo_gym import ComboGym
from args_test import ArgsTest
from model_recurrent import LstmAgent, GruAgent

    
def make_env(problem, episode_length=None, width=3, visitation_bonus=True, options=[]):
    def thunk():
        if len(options) > 0:
            env = ComboGym(rows=width, columns=width, problem=problem, random_initial=False, episode_length=episode_length, options=options, visitation_bonus=visitation_bonus)
        else:
            env = ComboGym(rows=width, columns=width, problem=problem, random_initial=True, episode_length=episode_length, visitation_bonus=visitation_bonus)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def _l1_norm(model, lambda_l1):
    l1_loss = 0
    for name, param in model.named_parameters():
        # Only apply L1 regularization to input weights of GRU (weight_ih_l0)
        if 'weight_ih_l0' in name and "bias" not in name:
            l1_loss += torch.sum(torch.abs(param))
    return lambda_l1 * l1_loss

def train_model(problem="test", option_dir=None):
    args = tyro.cli(ArgsTest)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    use_options = 0
    if option_dir is not None:
        use_options = 1
    run_name = f"{args.rnn_type}-{args.hidden_size}-{args.episode_length}-{args.num_steps}-{args.problem}-{args.seed}-{use_options}-quantized"
    print(run_name)
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=False,
            save_code=False,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    options = []
    if use_options == 1:
        with open("options/selected_options.pkl", "rb") as file:
            options = pickle.load(file)

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(problem, args.episode_length, args.game_width, args.visitation_bonus, options) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    if args.rnn_type == 'lstm':
        agent = LstmAgent(envs, args.hidden_size).to(device)
    elif args.rnn_type == 'gru':
        agent = GruAgent(envs, args.hidden_size, feature_extractor=True, option_len=len(options), quantized=args.quantized).to(device)
    else:
        print("Unknown type of model. Please choose between either LSTM or GRU.")
        exit()
    if args.fine_tune:
        agent = GruAgent(envs, args.hidden_size).to(device)     
        agent.load_state_dict(torch.load("models/gru-32-l1_1e-03-l2_0e+00-BL-TR.pt"))
        agent.train()
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay)
    optimizer = optim.Adam([
    {'params': agent.critic.parameters(), 'lr': args.value_learning_rate, 'name':'value'},
    {'params': [p for name, p in agent.named_parameters() if "critic" not in name], 'lr': args.learning_rate, 'eps':1e-5, 'weight_decay':args.weight_decay, 'name':'other'}
])
    # ALGO Logic: Storage setup
    # obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    # actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    # logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    # values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    global_step = 0
    positive_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)

    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    if args.ppo_type == 'lstm':
        next_rnn_state = (
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    elif args.ppo_type == 'gru':
        next_rnn_state = torch.zeros(agent.gru.num_layers, args.num_envs, agent.gru.hidden_size).to(device)


    # for iteration in range(1, args.num_iterations + 1):
    while global_step < args.total_timesteps:
        iteration = args.num_iterations

        obs = []
        actions = []
        logprobs = []
        rewards = []
        dones = []
        values = []

        if args.ppo_type == 'gru':
            initial_rnn_state = next_rnn_state.clone()
        elif args.ppo_type == 'lstm':
            initial_rnn_state = (next_rnn_state[0].clone(), next_rnn_state[1].clone())

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            # frac = 1.0 - (iteration - 1.0) / args.num_iterations
            frac = 1.0 - (global_step - 1.0) / args.total_timesteps
            if args.ppo_type == "original":
                lrnow = frac * args.learning_rate
                optimizer.param_groups[0]["lr"] = lrnow
            else:   # LSTM or GRU
                lr_value = frac * args.value_learning_rate
                lr_other = frac * args.learning_rate
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'value':
                        param_group['lr'] = lr_value
                    elif param_group.get('name') == 'other':
                        param_group['lr'] = lr_other

        positive_example = False
        number_samples = 0

        for step in range(0, args.num_steps):
            # print('############################ Step:', step)
            # obs[step] = next_obs
            # dones[step] = next_done
            obs.append(next_obs)
            dones.append(next_done)

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.ppo_type == 'gru':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                elif args.ppo_type == 'lstm':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                else:   # original
                    action, logprob, _, value, _ = agent.get_action_and_value(next_obs)
                # values[step] = value.flatten()
                values.append(value.flatten())

            actions.append(action)
            logprobs.append(logprob)
            # actions[step] = action
            # logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # print("next obs:", next_obs, "reward:", reward, "terminations:", terminations, "truncations:", truncations)
            # next_done = np.logical_or(terminations, truncations)
            next_done = terminations
            # print(infos)
            if 'g' in infos:
                goal_reached = infos['g'][0]
                episodic_goal_avg = infos["g"]
            elif 'final_info' in infos:
                goal_reached = infos['final_info'][0]['g']
                episodic_goal_avg = infos['final_info'][0]['g']
            if goal_reached > 0:
                positive_example = True
            # rewards[step] = torch.tensor(reward).to(device).view(-1)
            rewards.append(torch.tensor(reward).to(device).view(-1))
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)
            if next_done[0] == 1 or 1 in truncations:
                break

        if positive_example == True:
            # print('######################################################## Positive example ########################################################')
            positive_example = True
            number_samples = len(obs)
            # print('Number of samples:', number_samples)
        
        if positive_example:
        # Only count these steps towards total training steps
            # global_step += number_samples
            positive_step += number_samples


        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    episodic_l_avg = info["l"]
                    episodic_r_avg = info["episode"]["r"][0]
                    episodic_goal_avg = info["g"]
                    global_step += info["l"]

        if not positive_example:
            # continue with only 20% chance
            # if np.random.rand() > 0.2:
            # print("** negative example **")
            continue

        # print('Training positive')
        print("global step: ", global_step)
        rewards = torch.cat(rewards)
        values = torch.cat(values)
        # bootstrap value if not done
        with torch.no_grad():
            if args.ppo_type == 'gru':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            elif args.ppo_type == 'lstm':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            else:   # original
                next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(number_samples)):
                if t == number_samples - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        # b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        # b_logprobs = logprobs.reshape(-1)
        # b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        # b_dones = dones.reshape(-1) # done is used for GAE
        # b_advantages = advantages.reshape(-1)
        # b_returns = returns.reshape(-1)
        # b_values = values.reshape(-1)

        b_obs = torch.cat(obs).reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = torch.cat(logprobs).reshape(-1)
        b_actions = torch.cat(actions).reshape((-1,) + envs.single_action_space.shape)
        b_dones = torch.cat(dones).reshape(-1) # done is used for GAE
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        if args.ppo_type == "original":
            # b_inds = np.arange(args.batch_size)
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(number_samples).reshape(number_samples, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                # np.random.shuffle(b_inds)
                # for start in range(0, args.batch_size, args.minibatch_size):
                #     end = start + args.minibatch_size
                #     mb_inds = b_inds[start:end]
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        if len(mb_advantages) > 1:
                            std = mb_advantages.std()
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                        # else:
                        #     logger.info("Skipping normalization for single-element mb_advantages.")

                    # L1 loss
                    # l1_loss = _l1_norm(model=agent.actor, lambda_l1=args.l1_lambda)
                    l1_loss = agent.get_l1_norm()

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    # pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                    # pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_loss
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_loss * args.l1_lambda



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

                    # TODO: check
                    l1_reg = torch.tensor(0.).to(device)
                    for param in agent.actor.parameters():
                        l1_reg += torch.norm(param, 1)

                    # loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef + l1_lambda * l1_reg
                    loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef


                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

                if args.target_kl is not None and approx_kl > args.target_kl:
                    break
        
        else:   # LSTM or GRU
            assert args.num_envs % args.num_minibatches == 0
            envsperbatch = args.num_envs // args.num_minibatches
            envinds = np.arange(args.num_envs)
            flatinds = np.arange(number_samples).reshape(number_samples, args.num_envs)
            clipfracs = []
            for epoch in range(args.update_epochs):
                np.random.shuffle(envinds)
                for start in range(0, args.num_envs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                    if args.ppo_type == 'gru':
                        # print('mb_inds:', mb_inds, 'b_obs:', b_obs[mb_inds], 'initial_rnn_state:', initial_rnn_state[:, mbenvinds], 'b_dones:', b_dones[mb_inds], 'b_actions:', b_actions.long()[mb_inds])
                        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                            b_obs[mb_inds],
                            initial_rnn_state[:, mbenvinds],
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds],
                        )
                    elif args.ppo_type == 'lstm':
                        _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                            b_obs[mb_inds],
                            (initial_rnn_state[0][:, mbenvinds], initial_rnn_state[1][:, mbenvinds]),
                            b_dones[mb_inds],
                            b_actions.long()[mb_inds],
                        )
                    
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        if len(mb_advantages) > 1:
                            std = mb_advantages.std()
                            mb_advantages = (mb_advantages - mb_advantages.mean()) / (std + 1e-8)
                        # else:
                        #     logger.info("Skipping normalization for single-element mb_advantages.")

                    #L1 loss
                    if args.ppo_type == 'gru':
                        l1_loss = _l1_norm(agent, 1)
                    elif args.ppo_type == 'lstm':
                        l1_loss = _l1_norm(model=agent.lstm, lambda_l1=args.l1_lambda)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean() + args.l1_lambda * l1_loss

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
        # print("entropy ",entropy_loss.item())
        # print("goals reached ", episodic_goal_avg)
        print(episodic_r_avg, episodic_l_avg, episodic_goal_avg)
        if args.track:
                wandb.log({"value_loss": v_loss.item(), "policy_loss":pg_loss.item(),"entropy":entropy_loss.item(), "lr":lr_other, "valuelr":lr_value, "clipfac":np.mean(clipfracs), "old_approx_kl": old_approx_kl.item(), "approx_kl": approx_kl.item(), "explained_variance": explained_var, "episodic_return":episodic_r_avg ,"episodic_goals_reached":episodic_goal_avg, "episodic_length": episodic_l_avg})
    envs.close()
    writer.close()

    torch.save(agent.state_dict(), f'models/{args.seed}/{problem}-{use_options}-50-positive-not-quantized.pt')


if __name__ == "__main__":
    train_model(option_dir="options/selected_options.pkl")
    # train_model(option_dir=None)