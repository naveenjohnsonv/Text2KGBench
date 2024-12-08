import json
import re
import sys
from pathlib import Path

def update_onto_list(json_data):
    try:
        # Extract path pattern for ontologies
        onto_pattern = json_data["path_patterns"]["onto"]
        
        # Extract base directory from the pattern (up to the last '/')
        base_dir = Path(onto_pattern.split("$$onto$$")[0]).resolve()
        
        # Extract filenames from the directory
        filenames = []
        for f in base_dir.glob('*.json'):
            if f.is_file() and re.match(r'^\d+_\w+_ontology\.json$', f.name):
                # Extract number and category name
                match = re.search(r'^(\d+_\w+)_ontology', f.stem)
                if match:
                    filenames.append(match.group(1))
        
        filenames.sort()
        
        if filenames:
            json_data['onto_list'] = filenames
            return json_data
        else:
            print("No matching files found in directory")
            return None
            
    except FileNotFoundError:
        print(f"Directory not found: {base_dir}")
        return None
    except KeyError as e:
        print(f"Missing key in configuration: {str(e)}")
        return None
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return None

def main():
    config_path = next((arg.split('=')[1] for arg in sys.argv 
                       if arg.startswith('--prompt_gen_config_path=')), None)
    
    if not config_path:
        print("Error: Please provide --prompt_gen_config_path argument")
        return
        
    try:
        with open(config_path, 'r') as f:
            data = json.load(f)
        
        updated_data = update_onto_list(data)
        
        if updated_data:
            with open(config_path, 'w') as f:
                json.dump(updated_data, f, indent=2)
            print("Successfully updated config file")
            
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
    except json.JSONDecodeError:
        print("Invalid JSON format in config file")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
