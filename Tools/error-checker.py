import os
import codecs

def print_file_contents(root_dir):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                with codecs.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    print(f"Contents of {file_path}:")
                    print(f.read())
            except Exception as e:
                print(f"Could not read file {file_path} due to {str(e)}")

print_file_contents('../partial-joint')