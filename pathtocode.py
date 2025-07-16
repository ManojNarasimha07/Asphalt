import os

def read_files_and_concatenate(file_paths):
    """
    Given a list of file paths, read each file and concatenate the contents with the file path as the header.
    
    Args:
    - file_paths (list): A list of paths to the text files in the same folder.
    
    Returns:
    - concatenated_content (str): The concatenated content of all files with headers.
    """
    concatenated_content = ""
    
    # Loop through each file path
    for path in file_paths:
        if os.path.exists(path):
            # Add the file path as a header
            concatenated_content += f"=== {path} ===\n"
            
            # Open and read the file's content
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Append the content of the file to the result
                concatenated_content += content + "\n\n"  # Adding a newline after content for separation
        else:
            print(f"File {path} does not exist.")
    
    return concatenated_content


#full_content = read_files_and_concatenate(file_paths)


