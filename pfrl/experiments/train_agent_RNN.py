import logging
import os
from pfrl.experiments.evaluator import Evaluator, save_agent
from pfrl.utils.ask_yes_no import ask_yes_no
import csv
import time 


def save_agent_replay_buffer(agent, t, outdir, suffix="", logger=None):
    logger = logger or logging.getLogger(__name__)
    filename = os.path.join(outdir, "{}{}.replay.pkl".format(t, suffix))
    agent.replay_buffer.save(filename)
    logger.info("Saved the current replay buffer to %s", filename)


def ask_and_save_agent_replay_buffer(agent, t, outdir, suffix=""):
    if hasattr(agent, "replay_buffer") and ask_yes_no(
        "Replay buffer has {} transitions. Do you save them to a file?".format(
            len(agent.replay_buffer)
        )
    ):  # NOQA
        save_agent_replay_buffer(agent, t, outdir, suffix=suffix)

def train_agent_continuing_RNN(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
    wandb_logging=False, 
    env_checkpointable=False,
    buffer_checkpointable=False,
    load_env_state=False,
    total_reward_so_far = 0,
):

    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = 0
    total_reward = total_reward_so_far  # To calculate average reward

    # o_0, r_0
    obs , info = env.reset()
    if load_env_state:
        name = os.path.join(outdir, "checkpoint_{}.json".format(step_offset))
        env.load_env_state(name)
        logger.info("Loaded the environment state from %s", name)

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    episode_len = 0
    try:
        start = time.time()
        while t < steps:
            # a_t            
            action = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, terminated, truncated, info = env.step(action)
            
            t += 1
            total_reward += info['untransformed_rewards']  # Accumulate total reward
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False) or truncated
            agent.observe(obs, r, terminated, reset)

            for hook in step_hooks:
                hook(env, agent, t)
                
            episode_idx += 1


            episode_end = terminated or reset or t == steps

            if t == steps or episode_end:
                break

            if t % 100 == 0:  # Save values every 100 steps
                logger.info(
                "outdir:%s step:%s episode:%s R:%s",
                outdir,
                t,
                episode_idx,
                total_reward,
                )
                stats = agent.get_statistics()
                logger.info("statistics:%s", stats)
                print("SPS: ", episode_len / (time.time() - start))
                start = time.time()
                # Save episodic reward in a CSV file
                csv_filename = os.path.join(outdir, "episodic_rewards.csv")
                file_exists = os.path.isfile(csv_filename)

                with open(csv_filename, mode='a', newline='') as csv_file:
                    fieldnames = ['step', 'episode', 'reward', 'average_reward']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    average_reward = total_reward / (episode_idx if episode_idx > 0 else 1)
                    writer.writerow({'step': t, 'reward': total_reward, 'average_reward': average_reward})
                    if wandb_logging:
                        import wandb
                        wandb.log({'step': t, 'episode': episode_idx, 'reward': total_reward, 'average_reward': average_reward})
                
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")
                if env_checkpointable:
                    dirname = os.path.join(outdir, "{}{}".format(t, '_checkpoint'))
                    # Save the environment state
                    name = os.path.join(dirname, "checkpoint_{}.json".format(t))
                    env.save_env_state(name)
                if buffer_checkpointable:
                    save_agent_replay_buffer(agent, t, dirname, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        if env_checkpointable:
            dirname = os.path.join(outdir, "{}{}".format(t, '_except'))
            # Save the environment state
            name = os.path.join(dirname, "except_{}.json".format(t))
            env.save_env_state(name)

        if buffer_checkpointable:
            save_agent_replay_buffer(agent, t, dirname, suffix="_except")

        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")
    if env_checkpointable:
        dirname = os.path.join(outdir, "{}{}".format(t, '_finish'))
        # Save the environment state
        name = os.path.join(dirname, "finish_{}.json".format(t))
        env.save_env_state(name)
    if buffer_checkpointable:
            save_agent_replay_buffer(agent, t, dirname, suffix="_finish")
    return eval_stats_history




def train_agent_RNN(
    agent,
    env,
    steps,
    outdir,
    checkpoint_freq=None,
    max_episode_len=None,
    step_offset=0,
    evaluator=None,
    successful_score=None,
    step_hooks=(),
    eval_during_episode=False,
    logger=None,
    wandb_logging=False, 
    env_checkpointable=False,
    buffer_checkpointable=False,
    episode_idx = 0,
):
    logger = logger or logging.getLogger(__name__)

    episode_r = 0
    episode_idx = episode_idx
    other_bot_reward = 0
    # o_0, r_0
    obs , info = env.reset()

    t = step_offset
    if hasattr(agent, "t"):
        agent.t = step_offset

    eval_stats_history = []  # List of evaluation episode stats dict
    episode_len = 0
    try:
        start = time.time()
        while t < steps:
            # a_t            
            action = agent.act(obs)
            # o_{t+1}, r_{t+1}
            obs, r, terminated, truncated, info = env.step(action)
            
            t += 1
            episode_r += info['untransformed_rewards']
            if "other_bot_reward" in info:
                other_bot_reward += info['other_bot_reward']
            episode_len += 1
            reset = episode_len == max_episode_len or info.get("needs_reset", False) or truncated
            agent.observe(obs, r, terminated, reset)

            for hook in step_hooks:
                hook(env, agent, t)

            episode_end = terminated or reset or t == steps

            if episode_end:
                logger.info(
                    "outdir:%s step:%s episode:%s R:%s",
                    outdir,
                    t,
                    episode_idx,
                    episode_r,
                )
                stats = agent.get_statistics()
                logger.info("statistics:%s", stats)
                episode_idx += 1

            if evaluator is not None and (episode_end or eval_during_episode):
                eval_score = evaluator.evaluate_if_necessary(t=t, episodes=episode_idx)
                if eval_score is not None:
                    eval_stats = dict(agent.get_statistics())
                    eval_stats["eval_score"] = eval_score
                    eval_stats_history.append(eval_stats)
                if (
                    successful_score is not None
                    and evaluator.max_score >= successful_score
                ):
                    break

            if episode_end:
                if t == steps:
                    break
                print("SPS: " , episode_len / (time.time() - start))
                start = time.time()
                # Start a new episode
                # Save episodic reward in a CSV file
                csv_filename = os.path.join(outdir, "episodic_rewards.csv")
                file_exists = os.path.isfile(csv_filename)

                with open(csv_filename, mode='a', newline='') as csv_file:
                    if 'other_bot_reward' in info:
                        fieldnames = ['episode', 'steps', 'reward', 'other_bot_reward']
                    else: 
                        fieldnames = ['episode', 'steps', 'reward']
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                    if not file_exists:
                        writer.writeheader()
                    if 'other_bot_reward' in info:
                        writer.writerow({'episode': episode_idx,'steps': t , 'reward': episode_r, 'other_bot_reward': other_bot_reward})
                    else:
                        writer.writerow({'episode': episode_idx,'steps': t , 'reward': episode_r})
                    if wandb_logging:
                        import wandb
                        if 'other_bot_reward' in info:
                            writer.writerow({'episode': episode_idx,'steps': t , 'reward': episode_r, 'other_bot_reward': other_bot_reward})
                        else:
                            writer.writerow({'episode': episode_idx,'steps': t , 'reward': episode_r})
                other_bot_reward = 0
                episode_r = 0
                episode_len = 0
                obs, info = env.reset()
            if checkpoint_freq and t % checkpoint_freq == 0:
                save_agent(agent, t, outdir, logger, suffix="_checkpoint")
                if env_checkpointable:
                    dirname = os.path.join(outdir, "{}{}".format(t, '_checkpoint'))
                    # Save the environment state
                    name = os.path.join(dirname, "checkpoint_{}.json".format(t))
                    env.save_env_state(name)
                if buffer_checkpointable:
                    save_agent_replay_buffer(agent, t, dirname, suffix="_checkpoint")

    except (Exception, KeyboardInterrupt):
        # Save the current model before being killed
        save_agent(agent, t, outdir, logger, suffix="_except")
        if env_checkpointable:
            dirname = os.path.join(outdir, "{}{}".format(t, '_except'))
            # Save the environment state
            name = os.path.join(dirname, "except_{}.json".format(t))
            env.save_env_state(name)
        if buffer_checkpointable:
            save_agent_replay_buffer(agent, t, dirname, suffix="_except")
        raise

    # Save the final model
    save_agent(agent, t, outdir, logger, suffix="_finish")
    if env_checkpointable:
        dirname = os.path.join(outdir, "{}{}".format(t, '_finish'))
        # Save the environment state
        name = os.path.join(dirname, "finish_{}.json".format(t))
        env.save_env_state(name)
    if buffer_checkpointable:
        save_agent_replay_buffer(agent, t, dirname, suffix="_finish")

    return eval_stats_history


def train_agent_with_evaluation_RNN(
    agent,
    env,
    steps,
    eval_n_steps,
    eval_n_episodes,
    eval_interval,
    outdir,
    checkpoint_freq=None,
    train_max_episode_len=None,
    step_offset=0,
    eval_max_episode_len=None,
    eval_env=None,
    successful_score=None,
    step_hooks=(),
    evaluation_hooks=(),
    save_best_so_far_agent=True,
    use_tensorboard=False,
    eval_during_episode=False,
    logger=None,
    wandb_logging = False,
    case = "episodic", # episodic or continuing 
    env_checkpointable = False,
    buffer_checkpointable = False,
    load_env_state = False,
    total_reward_so_far = 0,
    episode_idx = 0,
):
    """Train an agent while periodically evaluating it.

    Args:
        agent: A pfrl.agent.Agent
        env: Environment train the agent against.
        steps (int): Total number of timesteps for training.
        eval_n_steps (int): Number of timesteps at each evaluation phase.
        eval_n_episodes (int): Number of episodes at each evaluation phase.
        eval_interval (int): Interval of evaluation.
        outdir (str): Path to the directory to output data.
        checkpoint_freq (int): frequency at which agents are stored.
        train_max_episode_len (int): Maximum episode length during training.
        step_offset (int): Time step from which training starts.
        eval_max_episode_len (int or None): Maximum episode length of
            evaluation runs. If None, train_max_episode_len is used instead.
        eval_env: Environment used for evaluation.
        successful_score (float): Finish training if the mean score is greater
            than or equal to this value if not None
        step_hooks (Sequence): Sequence of callable objects that accepts
            (env, agent, step) as arguments. They are called every step.
            See pfrl.experiments.hooks.
        evaluation_hooks (Sequence): Sequence of
            pfrl.experiments.evaluation_hooks.EvaluationHook objects. They are
            called after each evaluation.
        save_best_so_far_agent (bool): If set to True, after each evaluation
            phase, if the score (= mean return of evaluation episodes) exceeds
            the best-so-far score, the current agent is saved.
        use_tensorboard (bool): Additionally log eval stats to tensorboard
        eval_during_episode (bool): Allow running evaluation during training episodes.
            This should be enabled only when `env` and `eval_env` are independent.
        logger (logging.Logger): Logger used in this function.
    Returns:
        agent: Trained agent.
        eval_stats_history: List of evaluation episode stats dict.
    """

    logger = logger or logging.getLogger(__name__)

    for hook in evaluation_hooks:
        if not hook.support_train_agent:
            raise ValueError(
                "{} does not support train_agent_with_evaluation().".format(hook)
            )

    os.makedirs(outdir, exist_ok=True)

    if eval_env is None:
        assert not eval_during_episode, (
            "To run evaluation during training episodes, you need to specify `eval_env`"
            " that is independent from `env`."
        )
        eval_env = env

    if eval_max_episode_len is None:
        eval_max_episode_len = train_max_episode_len

    evaluator = Evaluator(
        agent=agent,
        n_steps=eval_n_steps,
        n_episodes=eval_n_episodes,
        eval_interval=eval_interval,
        outdir=outdir,
        max_episode_len=eval_max_episode_len,
        env=eval_env,
        step_offset=step_offset,
        evaluation_hooks=evaluation_hooks,
        save_best_so_far_agent=save_best_so_far_agent,
        use_tensorboard=use_tensorboard,
        logger=logger,
    )

    if case == "continuing":
        eval_stats_history = train_agent_continuing_RNN(
            agent,
            env,
            steps,
            outdir,
            checkpoint_freq=checkpoint_freq,
            max_episode_len=train_max_episode_len,
            step_offset=step_offset,
            evaluator=evaluator,
            successful_score=successful_score,
            step_hooks=step_hooks,
            eval_during_episode=eval_during_episode,
            logger=logger,
            wandb_logging=wandb_logging, 
            env_checkpointable=env_checkpointable, 
            buffer_checkpointable=buffer_checkpointable,
            load_env_state= load_env_state, 
            total_reward_so_far= total_reward_so_far,
        )
    else:
        eval_stats_history = train_agent_RNN(
            agent,
            env,
            steps,
            outdir,
            checkpoint_freq=checkpoint_freq,
            max_episode_len=train_max_episode_len,
            step_offset=step_offset,
            evaluator=evaluator,
            successful_score=successful_score,
            step_hooks=step_hooks,
            eval_during_episode=eval_during_episode,
            logger=logger,
            wandb_logging=wandb_logging,
            env_checkpointable=env_checkpointable,
            buffer_checkpointable=buffer_checkpointable,
            episode_idx=episode_idx,
        )

    return agent, eval_stats_history
