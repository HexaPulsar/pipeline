import yaml
import os
import glob

class DirectoryHandler:
    def __init__(
        self,
        global_config
    ):
        self.global_config = global_config
        self.path = self.handler_dirs()
        with open(os.path.join(self.path, "args.yaml"), "w") as file:
            yaml.dump(args, file, sort_keys=False)
        
    # create folder if not exist
    def handler_dirs(self):
        my_new_path = [
            self.global_config.SAVE_DIR_PATH
        ]
        s = ""
        if 'lc' in self.global_config.EXPERIMENT_TYPE:
            s += "LC_"
        if 'md' in self.global_config.EXPERIMENT_TYPE:
            s += "MD_"
        if 'feat' in self.global_config.EXPERIMENT_TYPE:
            s += "FEAT_"
        s = "/".join(my_new_path) +'/' + s[:-1]+"/" + self.global_config.EXPERIMENT_TYPE + "/"
        exp_path = os.path.join(s)
        if not os.path.exists(exp_path):
            try:
                os.makedirs(exp_path, exist_ok=False)
            except Exception as e:
                print(e)
        return exp_path

    def handler_ckpt_path(self, path):
        out_path = glob.glob(path + "*.ckpt")[0]
        return out_path

