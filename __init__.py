import os
import sys 

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '1'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib-config'

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from classes.environments   import envs_register

from utils.training_fun     import convert_parameters_for_training
from utils.parse_yaml       import load_config_section, load_config, print_config, yaml
