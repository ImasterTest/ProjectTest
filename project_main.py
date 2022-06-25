import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.getcwd(), r'resources\models'))
sys.path.append(os.path.join(os.getcwd(), r'resources\models\research'))
sys.path.append(os.path.join(os.getcwd(), r'resources\models\research\slim'))
if os.environ.get('PYTHONPATH') is None:
    os.environ['PYTHONPATH'] = ''
os.environ['PYTHONPATH'] += ';' + os.path.join(os.getcwd(), r'resources\models\research;')
os.environ['PYTHONPATH'] += os.path.join(os.getcwd(), r'resources\models;')
os.environ['PYTHONPATH'] += os.path.join(os.getcwd(), r'resources\models\research\slim;')

import PrintUtils
from colorama import Fore as Color
from project_gui import ProjectGui

def main():
    """
    The main project's function

    :return: None
    """
    if 'CONDA_DEFAULT_ENV' in os.environ:
        env = os.environ['CONDA_DEFAULT_ENV']  # The current python environment name
    else:
        env = 'Default Python'
    python_version = ".".join(map(str, sys.version_info[0:3]))  # The current python version

    PrintUtils.info('Current conda environment: {}'.format(Color.LIGHTGREEN_EX + env))
    PrintUtils.info('Current python version: {}'.format(Color.LIGHTGREEN_EX + python_version))

    projectgui = ProjectGui()
    projectgui.main_window()


if __name__ == '__main__':
    main()
