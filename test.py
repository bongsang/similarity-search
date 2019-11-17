import argparse

parser = argparse.ArgumentParser(
    description="Amazon's Geographic Mass Classification (Author: Bongsang Kim)")
parser.add_argument('--labels', type=list, nargs='+',
                    default=['andesite', 'gneiss', 'marble', 'quartzite', 'rhyolite', 'schist'])
parser.add_argument('--test_image', action='store_true')
parser.add_argument('--test_path', default='./tests')
parser.add_argument('--result_path', default='./results')
parser.add_argument('--model_path', default='./models')
parser.add_argument('--model_file', action='store')
args = parser.parse_args()

msg = args.model_file
print(msg)

