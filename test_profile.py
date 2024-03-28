import jsim, logging, os, re, sys
from piazza_bot import Answer, Followup, PiazzaBot
import parameters

JSIM_FILE = ".json"
JSIM_THRESHOLD = .25
JSIM_LIMIT = 50

def die(message):
    sys.stderr.write("{}\n".format(message))
    sys.exit(1)

def get_bot():
    """Prepare and return a PiazzaBot instance based on configuration parameters."""
    config_sources = {
        "email": parameters.email,
        "password": parameters.password,
        "class_code": parameters.class_code,
    }

    config = {}
    for var, source in config_sources.items():
        if not source:
            logging.error(f"Missing configuration for {var}. Please check your parameters.")
            sys.exit(1)
        config[var] = source

    return PiazzaBot.create_bot(**config)

def main():
    logging.basicConfig(level=logging.DEBUG)
    bot = get_bot()

if __name__ == "__main__":
    main()
