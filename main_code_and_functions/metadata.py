#A function which take a list of img ids, empty list of metedatas and Serial_Time
# as the imput arguments and returns filled metadats and Serail_Time lists

def metadata_and_time(paths,metadatas,Serial_Time):

    from PIL import Image
    from PIL.ExifTags import TAGS
    from date_to_num import datenum
    

    for path in paths:
        img_path = path #storing the image path

        #opening the image in PIL(Python Imaging Library Image object) image format to get metadata from exifdata
        pil_img = Image.open(img_path)
        exifdata = pil_img.getexif()

        meta_dict = {} #declaring an empty dictionary to store metadata obtained from EXIF data

        print('\n\n---------Metadata of Image {}--------\n\n'.format(id))
  
        #get each tag and the corresponding data from the EXIF metada 
        for tid in exifdata:
            tag = TAGS.get(tid) #tag correspond to the parameter of the image or cam used and data correspomd to the data stored in the parameter 
            data = exifdata.get(tid)

            if isinstance(data, bytes):
                data = "..."
    
            if tag != None: # eliminating out the parameter which has None value
                meta_dict[tag] = data #Storing the tag and corresponding data in a dictionary
                print(f"{tag:20}: {data}") #printing the metadata

        metadatas.append(meta_dict) # appending the formed metadata to the list

        date_time = meta_dict.get('DateTime') #taking the datetime string from the metadata
        
        #converting it to serial time using afunction datenum which take the datetime string as input
        srl_time = datenum(date_time)

        #printing it and appending it to the list Serial_Time
        
        print('\n\nSerial Time of img {} = '.format(path),srl_time)
        Serial_Time.append(srl_time)
    
    return metadatas,Serial_Time
  