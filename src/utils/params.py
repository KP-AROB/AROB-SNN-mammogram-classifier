import yaml, logging, importlib


def instanciate_encoder(module_name: str, class_name: str, params):
    try:
        module_ = importlib.import_module(module_name)
        if not hasattr(module_, class_name):
            raise AttributeError(f"Class '{class_name}' not found in module '{module_name}'")
        
        class_ = getattr(module_, class_name)
        mod = class_(**params)
        return mod
    except ModuleNotFoundError:
        raise ModuleNotFoundError(f"Module '{module_name}' not found")
    except AttributeError as e:
        raise e
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

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
