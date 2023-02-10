import argh, os, pickle
from simulator import main
from configs import *

parser = argh.ArghParser()


def run(exp_id="", run_id=0, runpath=""):
    _, runs = eval(f"config{exp_id}()")
    config = runs[run_id]
    if runpath != "":
        config["runpath"] = runpath
        os.chdir(runpath)
    with open("config.pickle", "wb") as pkl_file:
        pickle.dump(config, pkl_file)
    main(config)


parser = argh.ArghParser()
parser.add_commands([launch, run])

if __name__ == "__main__":
    parser.dispatch()
