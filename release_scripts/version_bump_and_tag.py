import subprocess

root_path = "../"
version_file_path = "../version.txt"


def get_github_status():

    git_process = subprocess.run(["git", "status", "-s"], capture_output=True)
    ret = git_process.stdout.decode('UTF-8')
    if not ret:

        print("Working tree is clean, ready to proceed.")
        return True

    print("There are uncommitted differences, cannot proceed.")
    print(ret)

    return False


def release():

    get_github_status()

    return


if __name__ == "__main__":

    release()
