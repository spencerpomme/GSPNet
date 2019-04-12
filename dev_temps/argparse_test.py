import argparse


parser = argparse.ArgumentParser(description='A sample test program for argparse.')
parser.add_argument('-a', action='store_true', default=False)
parser.add_argument('-b', action='store', dest='b')
parser.add_argument('-c', action='store', dest='c')

args = parser.parse_args()

print(args.a, args.b, args.c)
