from os import path, makedirs


def create_dir_if_not_exists(directory: str) -> None:
    if not path.exists(directory):
        makedirs(directory)


def get_file_name(full_file_path: str) -> str:
    _, file_extension = path.splitext(full_file_path)
    return full_file_path.split("/")[-1].replace(file_extension, "")


def get_correct_sub_directory(sub_dirs: list[str], path: str) -> str:
    for sub_dir in sub_dirs:
        if sub_dir in path:
            return sub_dir
    return ""
