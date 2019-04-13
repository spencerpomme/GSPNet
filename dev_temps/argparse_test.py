import argparse
import sys

parser = argparse.ArgumentParser(description='A sample test program for argparse.')
parser.add_argument('-a', action='store_true', default=False)
parser.add_argument('-b', action='store', dest='b')
parser.add_argument('-c', action='store', dest='c')


def main(*args):
    for arg in args:
        print(arg)

if __name__ == '__main__':

    main(*sys.argv)
