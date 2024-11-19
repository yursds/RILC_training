import os
import sys 
import __init__

script_dir = os.path.dirname(os.path.abspath(__file__)) 
project_root = os.path.abspath(os.path.join(script_dir, '../..')) 
sys.path.append(project_root) 
