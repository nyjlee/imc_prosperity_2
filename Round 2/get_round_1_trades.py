

import os

script_dir = os.path.dirname(__file__)
log_file_path = os.path.join(script_dir, 'data', 'round1_final.log')
output_file_path = os.path.join(script_dir, 'data', 'trades_round1')



def extract_trades_section(input_file, output_file):
    with open(input_file, 'r') as file:
        data = file.read()
    
    # Find the start of the 'trades' section
    start_index = data.find('Trade History:')
    
    # Extract the data from 'trades' to the end of the file
    if start_index != -1:
        trades_data = data[start_index:].strip()  # Includes everything from 'trades' to the end
    else:
        trades_data = "No 'trades' section found."
    
    # Write the extracted data to a new file
    with open(output_file, 'w') as file:
        file.write(trades_data)

# Usage
extract_trades_section(log_file_path, output_file_path)
