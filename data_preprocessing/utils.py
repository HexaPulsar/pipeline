import os
import shutil

def clear_export_directory(self,directory):
    """Clear the export directory after user confirmation."""
    if os.path.exists(directory):
        confirm = input(f"The directory '{directory}' already exists. Do you want to delete it? (y/n): ")
        if confirm.lower() == 'y':
            shutil.rmtree(directory)
            print(f"Directory '{directory}' deleted.")
            os.makedirs(directory)
            print(f"New directory '{directory}' created.")
        else:
            print("Directory deletion canceled. Continuing without deletion.")
    else:
        os.makedirs(directory)
        print(f"Directory '{directory}' created.")