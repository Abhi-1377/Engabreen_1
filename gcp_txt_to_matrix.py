#the gcp files will mainly be in text format but for further use in various functions including projection 
#function it needs to be converted to a 2-d array format 
#This function do that job


def save_txt_to_matrix(file_path): #argument is the file path name
    import numpy as np
    import re
    # Open the text file
    with open(file_path, 'r') as file:
        # Read the file contents
        file_content = file.readlines()

        # Initialize the matrix
        matx = []

        
        for line in file_content:
            #a variable which will help in not reading the header
            data_start = False
            # in the gcp file hearder starts with % so if the first character is % then continue reading the next line
            #if not that then data_start is set to True
            if line.startswith('%'):
                continue
            else:
                data_start = True
                
            #id data_start is true

            if data_start:
                # Split the line into values using multiple delimiters
                values = re.split(r'\s+|,\s*|\t', line.strip())
                row = []

                for value in values:
                       if "+" in value:
                              base, power = value.split("+")
                              base = float(base)
                              power = int(power)

                              num = base * pow(10, power)
                       else:
                              num = float(value)
                       row.append(num)                              



                # Append the row to the matrix
                matx.append(row)
            
            matrix = np.array(matx) #converting the list to array

    return matrix