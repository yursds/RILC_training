import subprocess
import os

def replace_code_in_file_with_sudo(file_path: str, old_code: str, new_code: str):
    try:
        # Path for the backup file
        backup_file_path = file_path + '.bak'

        # Check if the backup file already exists
        if not os.path.exists(backup_file_path):
            # Create a backup of the original file with sudo permissions
            subprocess.run(['sudo', 'cp', file_path, backup_file_path])
            print(f"Backup file created: {backup_file_path}")
        else:
            print(f"Backup file already exists: {backup_file_path}")

        # Read the content of the original file with sudo permissions
        result = subprocess.run(['sudo', 'cat', file_path], stdout=subprocess.PIPE, text=True)
        content = result.stdout

        # Modify the content
        content = content.replace(old_code, new_code)

        # Write the modified content to a new temporary file
        temp_file_path = '/tmp/mujoco_rendering_modified.py'
        with open(temp_file_path, 'w') as temp_file:
            temp_file.write(content)

        # Replace the original file with the modified file using sudo
        subprocess.run(['sudo', 'mv', temp_file_path, file_path])

        print(f"The file {file_path} has been successfully modified.")

        # Delete the temporary file (the source temp file doesn't need deletion after mv)
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            print(f"Temporary file deleted: {temp_file_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

# Example usage
file_path = '/usr/local/lib/python3.10/dist-packages/gymnasium/envs/mujoco/mujoco_rendering.py'
old_code = '''        self.add_overlay(
            bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
        )'''
new_code = '''        if mujoco.__version__ >= "3.0.0":
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_niter[0] + 1)
            )
        elif mujoco.__version__ < "3.0.0":
            self.add_overlay(
                bottomleft, "Solver iterations", str(self.data.solver_iter + 1)
            )'''

replace_code_in_file_with_sudo(file_path, old_code, new_code)
