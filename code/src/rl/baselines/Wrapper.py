from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, DummyVecEnv, VecNormalize
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
import stable_baselines3.common.policies
import os, yaml, subprocess, webbrowser, time, datetime
import gymnasium as gym
import rl.settings as settings
import rl.environments

"""
Potential improvements:
- set up of seeds: https://stable-baselines3.readthedocs.io/en/master/guide/examples.html
- add tensorboard integration and setup to config file (enable launching of tensorboard automatically)

Nice to haves:
- use of policies defined in algo and not in stable_baselines3.common.polices --> check differences
"""


def create_env(env_name, config=None, n_workers=1, **kwargs):
    """
    Parses the environment to correctly return the attributes based on the spec and type
    Creates a corresponding vectorized environment
    """

    def make_rl(**kwargs):
        """
        Decorator for generic envs
        """

        def _init():
            env_obj = getattr(rl.environments, env_name)
            env = env_obj(config)
            return env

        return _init

    def make_gym(rank, **kwargs):
        """
        Decorator for gym environments
        """

        def _init():
            env = gym.make(env_name)
            return env

        return _init

    if config is not None:
        n_workers = config["main"]["n_workers"]
    mapping = {"gym": make_gym, "rl": make_rl}
    env_type = get_env_type(env_name)
    env_decorator = mapping[env_type]
    envs = [env_decorator(rank=x) for x in range(n_workers)]

    # Parallelize
    if n_workers > 1:
        vectorized = SubprocVecEnv(envs, start_method="spawn")
        if "normalize" in config["main"].keys():
            vectorized = VecNormalize(vectorized, clip_obs=1, clip_reward=1)
    else:
        # Non multi-processing env
        vectorized = DummyVecEnv(envs)
    # Frame-stacking for CNN based environments
    if "frame_stack" in config["main"].keys():
        if config["main"]["frame_stack"] != 0:
            vectorized = VecFrameStack(
                vectorized, n_stack=config["main"]["frame_stack"]
            )

    return vectorized


def get_parameters(env_name, model_path=None, config_name=None, config_location=None):
    """
    Method for getting the YAML config file of the RL model, policy and environment
    Get config by prioritizing:
        1. Specific config file: /config/[name].yml
        2. From model's directory (in case of loading) /trained_models/_/_/_/parameters.yml
        3. /config/[env_name].yml
        4. /config/[env_type].yml
        5. /config/defaults.yml
    """
    env_type = get_env_type(env_name)
    env_params = os.path.join(settings.CONFIG, env_name + ".yml")
    if config_location is not None:
        path = config_location
    else:
        if config_name is not None:
            path = os.path.join(settings.CONFIG, config_name + ".yml")
        elif model_path is not None:
            path = os.path.join(model_path, "config.yml")
        elif os.path.isfile(env_params):
            path = env_params
        else:
            path = os.path.join(settings.CONFIG, env_type + ".yml")

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print("\nLoaded config file from: {}\n".format(path))

    return config


def get_env_type(env_name):
    """
    Get the type of environment from the env_name string
    - RL env is a custom environment (see naming requirements to initialize one in settings.py)
    - gym env is a standard 'gym' environment, which is imported with gym.make()
    """
    if any(substring in env_name for substring in settings.ENVIRONMENT_NAMES):
        return "rl"
    else:
        try:
            env = gym.make(env_name)
            del env
            return "gym"
        except:
            print("{} is not a viable environment.".format(env_name))
            return None


class Trainer(object):
    """
    Wrapper for stable_baselines library
    """

    def __init__(self, env, subdir="", model_from_file=None):

        self.config = None
        self.env = None
        self.model = None
        self.name = None

        self.env_name = env
        self._env_type = get_env_type(self.env_name)
        self.date = datetime.datetime.now().strftime("%m-%d_%H-%M")

        self._env_path = os.path.join(
            settings.TRAINED_MODELS, self._env_type, env, subdir
        )
        self._model_path = None
        self.reloaded = False
        self.done = True
        self.test_state = None
        print("Loading path {}".format(self._env_path))

    def load_model(self, num=None, config_file=None, latest=False, path=None):
        """
        Load a saved model
        """
        import glob

        assert os.path.isdir(self._env_path), "Path {} does not exist.".format(
            self._env_path
        )

        folder_list = glob.glob(self._env_path + "/*")
        if latest:
            model_path = max(folder_list, key=os.path.getctime)
        else:
            for folder in folder_list:
                if int(folder.split("/")[-1].split("_")[0]) == num:
                    model_path = folder
                    if not os.path.isfile(os.path.join(model_path, "model.pkl")):
                        model_path = model_path[:-1] + "1"
                    print("Model path:", model_path)
                    break

        self._model_path = model_path
        self.config = get_parameters(
            self.env_name, self._model_path, config_name=config_file
        )
        self.n_steps = self.config["main"]["n_steps"]
        model_file = os.path.join(model_path, "model.pkl")
        model_object = getattr(stable_baselines3, self.config["main"]["model"])
        self._unique_model_identifier = model_path.split("\\")[-1]
        print("Unique path: {}".format(self._unique_model_identifier))

        self.create_env()
        self.model = model_object.load(
            model_file[:-4], env=self.env, tensorboard_log=self._env_path
        )
        self.reloaded = True
        print("Loading model file {}".format(model_file))

        return self

    def create_model(
            self, config_file=None, dataset=None, config_location=None, name=None
    ):
        """
        Creates a new RL Model
        """

        if config_file is None:
            args = dict(env_name=self.env_name)
            args["config_location"] = config_location
            c = self.config = get_parameters(**args)
        else:
            c = self.config = config_file

        self.n_steps = self.config["main"]["n_steps"]
        self.create_env()

        policy_name = c["main"]["policy"]
        model_name = c["main"]["model"]
        # policy_params = c['policies'][policy_name] # for policy customization
        model_params = c["models"][model_name]
        print("\nCreating {} model...".format(model_name))

        self.policy = self._get_policy(policy_name)
        model_object = getattr(stable_baselines3, model_name)

        self.name = name

        model_args = dict(
            policy=self.policy,
            env=self.env,
            tensorboard_log=self._env_path,
            **model_params
        )

        self.model = model_object(**model_args)

        return self

    def _get_policy(self, policy_name):
        """
        Returns a corresponding policy object from stable_baselines
        --> potential add directions to custom policy defined in config file
        """
        return getattr(stable_baselines3.common.policies, policy_name)

    def create_env(self):
        """
        Parses the environment to correctly return the attributes based on the spec and type
        Creates a corresponding vectorized environment
        """
        print("Creating {} Environment...\n".format(self.env_name))
        self.env = create_env(self.env_name, self.config)
        if get_env_type(self.env_name) == "rl":
            self.env = VecMonitor(
                self.env
            )  # Only works if custom env is a vectorized env
        else:
            self.env = Monitor(self.env)

    def _create_model_dir(self):
        """
        Creates a unique subfolder in the environment directory for the current trained model
        """

        # Create the environment specific directory if it does not exist
        if not os.path.isdir(self._env_path):
            os.makedirs(self._env_path)

        # Get the unique id [N] of the directory ../trained_models/env_type/env/[N]_MODEL/...
        try:
            num = (
                    max([int(x.split("_")[0]) for x in os.listdir(self._env_path)]) + 1
            )  # Find the highest id number of current trained models
        except:
            num = 0

        c = self.config["main"]
        if self.name is not None:
            dir_name = self.name
        else:
            # Modify this based on what's relevant for identifying the trained models
            dir_name = "{}_{}_{}_{}_{}".format(
                c["model"], c["policy"], c["n_steps"], c["n_workers"], self.date
            )  # Unique stamp

        self._unique_model_identifier = (
                str(num) + "_" + dir_name
        )
        self._model_path = os.path.join(
            self._env_path, self._unique_model_identifier
        )  # trained_models/env_type/env/trainID_uniquestamp

    def _delete_incomplete_models(self):
        """
        Deletes directories that do not have the model file saved in them
        """
        import shutil

        count = 0
        for model_folder in os.listdir(self._env_path):
            path = os.path.join(self._env_path, model_folder)
            files = os.listdir(path)
            if "model.pkl" not in files:
                shutil.rmtree(path)
                count += 1
        print(
            "Cleaned directory {} and removed {} folders.".format(self._env_path, count)
        )

    def _save(self):
        self.model.save(os.path.join(self._model_path + '_1', "model"))
        # Save config
        with open(os.path.join(self._model_path + '_1', "config.yml"), "w") as f:
            yaml.dump(self.config, f, indent=4, sort_keys=False, line_break=" ")

        import shutil
        source = self.env_name + '.py'
        destination = self._model_path + '_1/' + self.env_name + '.py'
        shutil.copyfile(source, destination)

    # def _tensorboard(self, env_name=None):
    #     # Kill current session
    #     self._tensorboard_kill()
    #
    #     # Open the dir of the current env
    #     cmd = "tensorboard --logdir " + self._env_path #used to be tensorboard.exe --logdir " + ...
    #     print("Launching tensorboard at {}".format(self._env_path))
    #     DEVNULL = open(os.devnull, "wb")
    #     subprocess.Popen(cmd, shell=True, stdout=DEVNULL, stderr=DEVNULL)
    #     time.sleep(2)
    #     webbrowser.open_new_tab(
    #         url="http://localhost:6006/#scalars&_smoothingWeight=0.995"
    #     )

    # def _tensorboard_kill(self):
    #     print("Closing current session of tensorboard.")
    #     os.system("taskkill /f /im  tensorboard.exe")

    def _check_env_status(self):
        """
        In case one of the vectorized environments breaks - recreate it.
        """
        try:
            self.env.reset
        except BrokenPipeError as e:
            self.create_env()
            self.model.set_env(self.env)
            print(e, "\n BPE: Recreating environment...")
        except EOFError as e:
            self.create_env()
            self.model.set_env(self.env)
            print(e, "\n EOF: Recreating environment...")

    def _save_env_attribute(self, attribute):
        """
        Obtains and saves environment specific attributes in a text file
        (Only one of the environments in case they're running in parallel)
        """
        try:
            data = self.env.get_attr(attribute)
            with open(os.path.join(self._model_path, attribute + ".log"), "a") as f:
                for item in data:
                    f.write("%f\n" % item[0])
        except:
            print("Attribute does not exist.")

    def train(self, steps=None):
        """
        Train method
        """
        if not self.reloaded:
            self._create_model_dir()
        self._check_env_status()
        try:
            save_every = self.config["main"]["save_every"]
            # eval_every = self.config["main"]["eval_every"]
            save_every = save_every / self.config["main"]["n_workers"]  # save_every is provided to callback per environment
            # eval_every = eval_every / self.config["main"]["n_workers"]
            n_steps = steps if steps is not None else self.n_steps
            self.model.is_tb_set = True
            config = dict(
                total_timesteps=n_steps,
                tb_log_name=self._unique_model_identifier,
                reset_num_timesteps=True,
            )

            # Train the model and save a checkpoint every n steps
            print("CTRL + C to stop the training and save.\n")
            checkpoint_callback = CheckpointCallback(save_freq=save_every,
                                                     save_path=self._model_path + "_1/intermediate_models/",
                                                     name_prefix='intermediate_model')

            # if self.config["main"]["eval_callback"] == True:
            #     if "evaluation_instance" in self.config["environment"]:
            #
            #         eval_config = create_evaluation_config(self.config)
            #         eval_env = create_env(self.env_name, config=eval_config)
            #
            #     eval_callback = EvalCallback(eval_env, best_model_save_path=self._model_path + "_1/best_model/",
            #                                  log_path="./logs/results", eval_freq=eval_every)
            #
            #     callback = CallbackList([checkpoint_callback, eval_callback])
            #
            # else:
            callback = checkpoint_callback

            self.model = self.model.learn(callback=callback, **config)
            self._save()

        except KeyboardInterrupt:
            self._save()
            print("Done.")
