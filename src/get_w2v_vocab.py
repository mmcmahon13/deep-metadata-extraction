import argparse


def main():
    arg_parser = argparse.ArgumentParser(description='Get the vocab from an embeddings file')
    arg_parser.add_argument('embeddings_file_path')
    arg_parser.add_argument('target_path')
    args = arg_parser.parse_args()

    with open(args.embeddings_file_path, 'r') as ef:
        for line in ef:
            comps = line.split(" ")
            print(comps[0])

if __name__ == "main":
    main()