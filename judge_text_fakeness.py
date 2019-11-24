import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Use a Fake News Model')
    parser.add_argument('-model', dest='model_name',
                        help='The model you want to test against.')
    parser.add_argument('-data', dest='data_fname',
                        help='The file name of the data that you are testing against')
    parser.add_argument('-out', dest='fake',
                        help='The file where you want to save the generated adversarial example')
    args = parser.parse_args()

    if args.model_name == 'lstm':
