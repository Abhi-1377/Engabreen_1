
#A function to convert the .mat file to txt file and return a list containing the text file path names
#the argument to the function is a string contatining the file name

def matTotxt(doc_name):
    import scipy.io
    # Load data from .mat file
    data = scipy.io.loadmat(doc_name) 
    #a list to store the .txt file paths
    text_list = []

    # Iterate over the variables and convert them to text
    for items in data.items():
        text = str(items[1])
        # Save the text to a file with the variable name
        with open(f'dem_txt\{items[0]}.txt', 'w') as f:
            f.write(text)
        text_list.append('dem_txt\{}.txt'.format(items[0])) #storing the file name in a list

    #printing the contents of each variale in the mat file
    for text in text_list:
        File = open(text,'r')
        File_content = File.read()
        print('\n\nFile {}\n\n'.format(text))
        print(File_content)
        File.close

    return text_list


