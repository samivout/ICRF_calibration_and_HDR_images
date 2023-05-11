import subprocess

root_path = "../"
version_file_path = "../version.txt"


def get_github_status():

    git_process = subprocess.run(["git", "status", "-s"], capture_output=True)
    print(git_process.stdout)

    return


def release():

    get_github_status()

    return


if __name__ == "__main__":

    release()
