import subprocess
import os

script_directory = os.path.dirname(os.path.realpath(__file__))
root_directory = os.path.dirname(script_directory)
version_file_path = os.path.join(script_directory, "../version.txt")
semantic_levels = ['major', 'minor', 'patch']
version_number_levels = len(semantic_levels)


def get_github_status():
    """
    Get status of the repo in which the script is run in. Return true if there
    are no uncommitted changes and false if there are changes. Additionally,
    print the uncommitted changes.
    Returns: bool
    """
    git_status_process = subprocess.run(["git", "status", "-s"],
                                        capture_output=True,
                                        cwd=root_directory)

    ret = git_status_process.stdout.decode('UTF-8')
    if not ret:
        print("Working tree is clean, ready to proceed.")
        return True

    print("There are uncommitted differences, cannot proceed.")
    print(ret)

    return False


def git_bump_and_tag(version_string: str):
    """
    Commit changed version.txt file and create a lightweight tag based on the
    new version number. Additionally, push to origin:master if user wants to.
    Args:
        version_string: string representation of the new version number.
    """
    subprocess.run(["git", "add", version_file_path])

    git_commit_process = subprocess.run(["git", "commit", "-m",
                                         f'Bump version to {version_string}'],
                                        capture_output=True,
                                        cwd=root_directory)
    ret_commit = git_commit_process.stdout.decode('UTF-8')
    print(ret_commit)

    git_tag_process = subprocess.run(["git", "tag", version_string],
                                     capture_output=True,
                                     cwd=root_directory)
    ret_tag = git_tag_process.stdout.decode('UTF-8')
    print(ret_tag)

    print('Ready to push. Do you want me to push to origin:master now y/n?')
    push_response = input().casefold()
    if push_response == 'y':

        subprocess.run(["git", "push", "origin", "master", version_string],
                       cwd=root_directory)
        print('Done pushing.')
    else:
        print('Remember to push the tag.')

    return


def get_version_number():
    """
    Get version number from the version.txt file.
    Returns: version number as a string or None if problems are encountered.
    """
    try:
        version_file = open(version_file_path, "r")
        version_number_str = version_file.read()
        version_file.close()
    except FileNotFoundError:
        print('Version.txt not found!')
        return None

    return version_number_str


def save_version_number(version_string: str):
    """
    Save version number to the version.txt file.
    Args:
        version_string: the version number to be saved as a string.
    """
    version_file = open(version_file_path, "w")
    version_file.write(version_string)
    version_file.close()

    return


def parse_version_number(version_number_str: str):
    """
    Parse version number from a string.
    Args:
        version_number_str: string representing the version number. Should be in
    the format of semantic version numbers with desired number of levels.

    Returns: the parsed version number as a list of ints, or None if parsing is
        not successful.
    """
    parsing_successful = True
    version_number_split = version_number_str.split('.')
    version_number = []

    if len(version_number_split) > version_number_levels:
        print('Too many separators in version number!')
        parsing_successful = False

    if len(version_number_split) < version_number_levels:
        print('Too few separators in version number!')
        parsing_successful = False

    for element in version_number_split:
        try:
            version_number.append(int(element))
        except ValueError:
            print('There is something unexpected in the version number!')
            parsing_successful = False

    for element in version_number:
        if element < 0:
            print('No negative numbers in version number!')
            parsing_successful = False

    if not parsing_successful:
        return None

    return version_number


def update_version_number(current_version_number: list[int]):
    """
    Updates the version number based on user input. User can either input a
    hardcoded semantic level to bump the respective level, or their own version
    number. User can also cancel by entering 'c'.
    Args:
        current_version_number: current version number as a list of ints.

    Returns: an updated version number as a list of ints.
    """
    new_version_number = current_version_number
    input_accepted = False
    used_semantic_levels = []
    for i in range(version_number_levels):
        used_semantic_levels.append(semantic_levels[i])

    while not input_accepted:

        print('Enter either ' + ', '.join(used_semantic_levels) +
              ' to bump the respective semantic level, or enter your own '
              'version number. Enter c to cancel.')
        text_input = input().casefold()
        if text_input == 'c':
            input_accepted = True

        bump_level_found = False
        for i, semantic_level in enumerate(used_semantic_levels):
            if text_input == semantic_level:
                new_version_number[i] += 1
                bump_level_found = True
                input_accepted = True
                continue
            if bump_level_found:
                new_version_number[i] = 0

        if not bump_level_found and not input_accepted:
            parsed_version_number = parse_version_number(text_input)
            if parsed_version_number is not None:
                new_version_number = parsed_version_number
                input_accepted = True

    return new_version_number


def release_process():
    """
    Function that manages the whole process of updating the version number.
    Returns:
    """
    current_version_string = get_version_number()
    current_version_number = parse_version_number(current_version_string)
    if current_version_number is None:
        print('Aborting process.')
        return

    print(f'Current version number is {current_version_string}')

    working_tree_is_clean = get_github_status()
    if not working_tree_is_clean:
        return

    new_version_number = update_version_number(current_version_number)
    new_version_string = '.'.join(str(x) for x in new_version_number)

    if new_version_number == current_version_number:
        print('Version number bump cancelled.')
        return

    print(f'New version number would be {new_version_string}')
    print('Continue with this number y/n?')
    continue_response = input().casefold()
    if continue_response == 'y':
        save_version_number(new_version_string)
        git_bump_and_tag(current_version_string)
    else:
        print('Version number bump cancelled.')

    return


if __name__ == "__main__":
    release_process()
