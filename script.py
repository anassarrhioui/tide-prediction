import csv
from datetime import datetime

def get_month_data(station_name, year):
    with open("Tide Prediction.csv",'r') as full_data_file:
        reader = csv.reader(full_data_file)
        next(reader);next(reader)

        month, month_data_file, writer = None, None, None

        for line in reader:
            date = datetime.fromisoformat(line[0][:-1])
            if line[3] != station_name and month == None:
                continue
            elif line[3] != station_name or (date.year != year and month!=None):
                break
            if date.year!=year or date.minute != 0:
                continue
            if date.month != month:
                if month_data_file: month_data_file.close()
                month = date.month
                month_data_file = open(f"{station_name}-{month}-{year}.csv", 'w',newline='')
                writer = csv.writer(month_data_file)

            # writer.writerow( [date.timestamp(), date.year, line[3], line[4]] )
            writer.writerow( [date.timestamp(), line[4]] )


# Stations : Achill_Island_MODELLED, Aranmore
# 2017 - 2019
if __name__ == "__main__":
    get_month_data("Achill_Island_MODELLED", 2017)