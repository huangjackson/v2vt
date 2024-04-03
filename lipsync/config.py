import os


class LipSyncModel:

    def __init__(self):
        self.script_dir = os.path.dirname(os.path.realpath(__file__))
        self.project_dir = os.path.join(self.script_dir, '..')
        self.models_dir = os.path.join(self.script_dir, 'checkpoints')

        self.tmp_dir = os.path.join(self.project_dir, 'TEMP')
        self.out_dir = os.path.join(self.project_dir, 'logs')
