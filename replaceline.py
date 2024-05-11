import os

def replace_line_in_files(folder_path, old_line, new_line):
    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        # Check if it's a file
        if os.path.isfile(filepath):
            # Open the file for reading and writing
            with open(filepath, 'r') as file:
                lines = file.readlines()
            # Replace the old line with the new one if it exists
            with open(filepath, 'w') as file:
                for line in lines:
                    if line.strip() == old_line:
                        file.write(new_line + '\n')
                    else:
                        file.write(line)

# Example usage
folder_path = '/DATA/raghavendra_2211cs14/mtp/one_for_all/code'
old_line = 'sdsfds'
new_line = 'T    = no_of_opt_iteration   # maximum number of generations'
replace_line_in_files(folder_path, old_line, new_line)