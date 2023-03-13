import sys
sys.path.append('../')
sys.path.append('../../')
from GA_class import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--generation', required=True, help='number of generation')
    args = parser.parse_args()
    GA_custom = GA_custom(feats_MG = 'all_res_996', init=True, initialization_path='../for_initialization',eval_on_bench=True)
    findBestFeats = GA_custom(generation=args.generation)
