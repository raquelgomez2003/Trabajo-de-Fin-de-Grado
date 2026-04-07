import csv

input_file = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto5\Biopac data\Subject11F_Triggers_block.csv"
output_file = r"C:\Users\raque\Desktop\TFG\base de datos TFG\Base1_Sujeto5\Biopac data\Subject11F_Triggers_block_separado.csv"

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:

    for line in infile:
        line = line.strip()
        
        # Cambiar comas por punto y coma
        new_line = line.replace(',', ';')
        
        outfile.write(new_line + '\n')

print("Archivo listo para Excel")