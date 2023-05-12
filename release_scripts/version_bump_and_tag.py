import subprocess
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.dirname(script_directory)
version_file_path = os.path.join(script_directory, "../version.txt")
# print(root_directory)
# print(version_file_path)
version_number_levels = 3


def get_github_status():

    git_process = subprocess.run(["git", "status", "-s"],
                                 capture_output=True,
                                 cwd=root_directory)

    ret = git_process.stdout.decode('UTF-8')
    if not ret:

        print("Working tree is clean, ready to proceed.")
        return True

    print("There are uncommitted differences, cannot proceed.")
    print(ret)

    return False


def get_version_number():

    try:
        version_file = open(version_file_path, "r")
        version_number_str = version_file.read()
        version_file.close()
    except FileNotFoundError:
        print('Version.txt not found!')
        return None

    version_number = parse_version_number(version_number_str)

    return version_number


def parse_version_number(version_number_str: str):

    version_number_list = version_number_str.split('.')
    version_number = []

    if len(version_number_list) > version_number_levels:
        print('Too many separators in version number!')
        return None

    if len(version_number_list) < version_number_levels:
        print('Too few separators in version number!')
        return None

    for element in version_number_list:
        try:
            version_number.append(int(element))
        except ValueError:
            print('There is something unexpected in the version number file!')
            return None

    return version_number


def update_version_number(current_version_number: list[int]):

    new_version_number = current_version_number
    input_not_accepted = True
    while input_not_accepted:

        print('Enter major, minor or patch to bump the respective version level'
              ', or enter your own version number. Enter c to cancel')
        text_input = input()
        if text_input == 'c':
            input_not_accepted = False

    return new_version_number


def release_process():

    current_version_number = get_version_number()
    if current_version_number is None:
        print('Aborting process.')
        return

    print(f'Current version number is ' +
          '.'.join(str(x) for x in current_version_number))

    working_tree_is_clean = get_github_status()
    if not working_tree_is_clean:
        return

    new_version_number = update_version_number(current_version_number)

    return


if __name__ == "__main__":

    release_process()
