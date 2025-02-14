import os

import constants


class Train_List_Generator:
    """
    this class is responsible for generating a list
    of files that will be used for training
    (data set has such lists only for testing and validation, each
    contains around 10% of data)
    """

    @staticmethod
    def generate():
        if os.path.isfile(constants.TRAIN_LIST_PATH):
            print("file containing train samples already exists")
            return

        sounds_not_for_training = set()

        # read testing_list.txt and validation_list.txt to a map
        with open(constants.TESTING_LIST_PATH, "r") as file:
            for line in file:
                sounds_not_for_training.add(line.strip())

        with open(constants.VALIDATE_LIST_PATH, "r") as file:
            for line in file:
                sounds_not_for_training.add(line.strip())

        # write to file all files that do not exist in sound_not_for_testing
        with open(constants.TRAIN_LIST_PATH, "w") as target_file:
            base_path = constants.AUDIO_BASE_PATH
            classes = constants.CLASS_MAPPINGS
            for class_name in classes:
                class_folder_path = base_path + class_name
                all_files_of_this_class = os.listdir(class_folder_path)

                # add folder prefix
                files_with_prefixes = [
                    class_name + "/" + x for x in all_files_of_this_class
                ]

                filtered_file_names = [
                    x for x in files_with_prefixes if x not in sounds_not_for_training
                ]

                # save all filtered file names to file
                for file_name in filtered_file_names:
                    target_file.write(file_name + "\n")


if __name__ == "__main__":
    Train_List_Generator.generate()
