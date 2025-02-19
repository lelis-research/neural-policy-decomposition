import os
os.environ["OMP_NUM_THREADS"] = "1"
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


from envs.combogrid import ComboGridEnv
from envs.combogrid_gym import ComboGridGym
from args import Args
from args_test import ArgsTest
from models.model_recurrent import LstmAgent, GruAgent
print("Importing libs completed")

    
def make_env(problem, episode_length=None, width=3, visitation_bonus=1, options=[]):
    def thunk():
        if len(options) > 0:
            env = ComboGridGym(rows=width, columns=width, problem=problem, random_initial=False, episode_length=episode_length, options=options, visitation_bonus=visitation_bonus)
        else:
            env = ComboGridGym(rows=width, columns=width, problem=problem, random_initial=True, episode_length=episode_length, visitation_bonus=visitation_bonus)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk

def _l1_norm(model, lambda_l1):
    l1_loss = 0
    for name, param in model.named_parameters():
        # Only apply L1 regularization to input weights of GRU (weight_ih_l0)
        if 'network' in name and "bias" not in name:
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
    run_name = f"{seed}/{problem}-lr{args.learning_rate}-num_step{args.num_steps}-clip_coef{args.clip_coef}-ent_coef{args.ent_coef}-epoch{args.update_epochs}-vloss{args.clip_vloss}-visit{args.visitation_bonus}-actor{args.actor_layer_size}-critic{args.critic_layer_size}"
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
    writer = SummaryWriter(f"logs/all-samples/{args.seed}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    options = []
    if use_options == 1:
        # with open(option_dir, "rb") as file:
        #     options = pickle.load(file)
        options = [ 3, 4, 5, 6]

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
        agent = GruAgent(envs, args.hidden_size, feature_extractor=False, option_len=len(options), quantized=args.quantized).to(device)
    else:
        print("Unknown type of model. Please choose between either LSTM or GRU.")
        exit()
    if args.fine_tune:
        agent = GruAgent(envs, args.hidden_size).to(device)     
        agent.load_state_dict(torch.load("training_data/models/gru-32-l1_1e-03-l2_0e+00-BL-TR.pt"))
        agent.train()
    # optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5, weight_decay=args.weight_decay)
    optimizer = optim.Adam([
    {'params': agent.critic.parameters(), 'lr': args.value_learning_rate, 'name':'value'},
    {'params': [p for name, p in agent.named_parameters() if "critic" not in name], 'lr': args.learning_rate, 'eps':1e-5, 'weight_decay':args.weight_decay, 'name':'other'}
])
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
    if args.rnn_type == 'lstm':
        next_rnn_state = (
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
            torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        )  # hidden and cell states (see https://youtu.be/8HyCNIVRbSU)
    else:
        next_rnn_state = torch.zeros(agent.gru.num_layers, args.num_envs, agent.gru.hidden_size).to(device)

    for iteration in range(1, args.num_iterations + 1):
        if args.rnn_type == 'gru':
            initial_rnn_state = next_rnn_state.clone()
        elif args.rnn_type == 'lstm':
            initial_rnn_state = (next_rnn_state[0].clone(), next_rnn_state[1].clone())
        
        if args.anneal_lr == 1:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lr_value = frac * args.value_learning_rate
            lr_other = frac * args.learning_rate
            for param_group in optimizer.param_groups:
                if param_group.get('name') == 'value':
                    param_group['lr'] = lr_value
                elif param_group.get('name') == 'other':
                    param_group['lr'] = lr_other
        episodic_r_avg = 0
        episodic_l_avg = 0
        episodic_goal_avg = 0
        counter = 0
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                if args.rnn_type == 'gru':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                elif args.rnn_type == 'lstm':
                    action, logprob, _, value, next_rnn_state = agent.get_action_and_value(next_obs, next_rnn_state, next_done)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            # print(infos)
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        # print(f"global_step={global_step}, episodic_return={info["episode"]}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["l"], global_step)
                        episodic_l_avg += info["l"]
                        episodic_r_avg += info["episode"]["r"][0]
                        episodic_goal_avg += info["g"]
                        counter += 1
        episodic_l_avg /= float(counter)
        episodic_r_avg /= float(counter)
        episodic_goal_avg /= float(counter)
        # print(episodic_l_avg)
        # if args.track:
        #     wandb.log({"episodic_return":episodic_r_avg , "episodic_length": episodic_l_avg})

        # bootstrap value if not done
        with torch.no_grad():
            if args.rnn_type == 'gru':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            elif args.rnn_type == 'lstm':
                next_value = agent.get_value(next_obs, next_rnn_state, next_done).reshape(1, -1)
            
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
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        gradients = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                if args.rnn_type == 'gru':
                    _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                        b_obs[mb_inds],
                        initial_rnn_state[:, mbenvinds],
                        b_dones[mb_inds],
                        b_actions.long()[mb_inds],
                    )
                elif args.rnn_type == 'lstm':
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
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                #L1 loss
                l1_loss = _l1_norm(model=agent.gru, lambda_l1=args.l1_lambda)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean() + l1_loss

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss == 1:
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
                for p in agent.parameters():
                    gradients += [p.grad.norm()]

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
        print("***********************************\n STEP:", global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
        if args.track:
            wandb.log({"value_loss": v_loss.item(), "policy_loss":pg_loss.item(),"entropy":entropy_loss.item(), "lr": optimizer.param_groups[0]["lr"],  "clipfac":np.mean(clipfracs), "old_approx_kl": old_approx_kl.item(), "approx_kl": approx_kl.item(), "explained_variance": explained_var, "episodic_return":episodic_r_avg ,"episodic_goals_reached":episodic_goal_avg, "episodic_length": episodic_l_avg})
    envs.close()
    writer.close()
    if not os.path.exists(f'training_data/models-all-samples/{args.seed}'):
        os.mkdir(f'training_data/models-all-samples/{args.seed}')
    torch.save(agent.state_dict(), f'training_data/models-all-samples/{args.seed}/{run_name}.pt')

if __name__ == "__main__":
    train_model(option_dir="training_data/optionsselected_options_width_5.pkl")
    # train_model(option_dir=None)
