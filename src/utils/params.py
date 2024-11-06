import yaml, logging


def load_parameters(path):
    try:
        stream_file = open(path, "r")
        parameters = yaml.load(stream_file, Loader=yaml.FullLoader)

        logging.info(
            "[AROB-2024-KAPTIOS::main] Success: Loaded parameter file at: {}".format(
                path
            )
        )
    except:
        logging.error(
            "[AROB-2024-KAPTIOS::main] ERROR: Invalid parameter file at: {}, exiting...".format(
                path
            )
        )
        exit()

    return parameters
