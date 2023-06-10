#A function to convert the date and time at which the image is taken to serial time format

def datenum(date_time):
    from datetime import date
    from datetime import datetime as dt

    d = dt.strptime(date_time,'%Y:%m:%d %H:%M:%S')
    return 366 + d.toordinal() + (d - dt.fromordinal(d.toordinal())).total_seconds()/(24*60*60)