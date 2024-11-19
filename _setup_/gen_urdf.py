import os

def insert_string_in_urdf(file_path:str, search_string:str, insert_string:str):
    # Verifica se il file esiste
    
    file_path_abs = os.path.join(current_path, file_path)
    
    if not os.path.isfile(file_path_abs):
        print(f"Il file {file_path_abs} non esiste.")
        return

    # Leggi il contenuto del file
    with open(file_path_abs, 'r') as file:
        lines = file.readlines()

    # Processa le linee e inserisce la stringa desiderata
    modified_lines = []
    for line in lines:
        if search_string in line:
            line = line.replace(search_string, f"{insert_string}")
        modified_lines.append(line)

    # Scrivi le modifiche in un nuovo file
    new_file_path = file_path_abs.replace('leg_constrained_template.urdf', 'leg_constrained.urdf')
    with open(new_file_path, 'w') as file:
        file.writelines(modified_lines)

    print(f"Il file modificato è stato salvato come {new_file_path}")

# Esempio di utilizzo
current_path = os.path.dirname(os.path.abspath(__file__)) 
file_path = '../classes/robots/robot_models/softleg_urdf/urdf/leg_constrained_template.urdf'
search_string = '${path_to_stl}'
insert_string = os.path.join(current_path,'../classes/robots/robot_models/softleg_urdf/meshes/')
insert_string_in_urdf(file_path, search_string, insert_string)
