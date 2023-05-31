"""
Usage:
  experiments evaluate <environment> <agent> (--train|--test) [options]

Options:
  --train                Train the agent.
  --test                 Test the agent.

"""

from docopt import docopt

from RobustPlanner.trainer.evaluation import Evaluation
from RobustPlanner.common.factory import load_agent, load_environment

# from rl_agents.tr ainer.graphics import RewardViewer


def main():

    opts = docopt(__doc__)
    if opts['evaluate']:
        # for _ in range(int(opts['--repeat'])):
        evaluate(opts['<environment>'], opts['<agent>'], opts)

def evaluate(environment_config, agent_config, options):
    
    env = load_environment(environment_config)
    
    agent = load_agent(agent_config, env)
    evaluation = Evaluation(env, agent)

    if options['--test']:
        evaluation.test()

if __name__ == "__main__":
    main()
