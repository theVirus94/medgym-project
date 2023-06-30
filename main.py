import argparse
from experiment_enums import experimentsAll


def main():
    """
    The main function that parses arguments
    :return:
    """
    parser = argparse.ArgumentParser(description="run GAN experiment using the provided experiment enums.")
    args = parser.parse_args()
    experiment_setup(args)


def experiment_setup(args: argparse.Namespace) -> None:
    """
    This function sets up the experiment and runs it for both regular and logic GANs
    :param args: dictionary arguments from user
    :return: None
    """
    experiments = experimentsAll
    for experiment in experiments:
        experiment.run(logging_frequency=1)
        # experiment.visualize()


if __name__ == "__main__":
    main()