import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import timedelta
from GraphFunctions import *

#initidal data
obs_File = 'ExampleObserved.csv' #enter observed flow data. Should have three columns - datetime, ML/d and quality rating
rain_File = 'ExampleRainfall.csv' #enter rainfall data. Should have columns for datetime, and Mm/d, it can also accumilate a qaulity rating column, but the graphs do not currently account for one
mod_File = 'ExampleModelled.csv'  #enter modelled data. Should have columns for datetime, ML/d, it can also accumilate a qaulity rating column, but the graphs do not currently account for one
dateFormat = "%d/%m/%Y %H:%M" #the datetime format used in your data files.
imageFormat = '.png' #the resulting file format for generated graphs

#data standards
largestFloodNo = 2 #the number of top largest floods you want to identify. Only needed if you are creating largest flood graphs
endOfWaterYear = 6 #the final month of a water year - for example, its currently set to July-June water years
acceptableMissing = 0 #how many low quality data points can be missing per water year before its considered missing data

#a dictionary for customising graph text sizes
sizes = {'title':20, 'subtitle':14,'ticks':14, 'xLabel':16, 'yLabel':16, 'legend':12} 

#a prewitten dictionary for Queensland graph colors
coloursQLD = {'base':"#3B6E8F", 'obs':"#00008B",'mod':"#D02090", 
                         'rain':"#7F7F7F", 'footnote':"#15467A", 'stringsAsFactors':False}

#a prewitten dictionary for New South Wales graph colors
coloursNSW = {'base':"#3B6E8F", 'obs':"#FF0000",'mod':"#0000FF", 
                         'rain':"#7F7F7F", 'footnote':"#15467A", 'stringsAsFactors':False}


#get data from file, then stores it into a single dataframe - needs to be run before constucting any graph
dataframe = parseFiles(obs_File, rain_File, mod_File, dateFormat)                  

#Largest flood graphs - this can make as many flood graphs as disired from the same data set.
BuildLargestFloods(dataframe,largestFloodNo,coloursQLD,sizes,imageFormat)

#annual time series graph. Years with missing data are represented with dotted lines
BuildAnnualGraph(dataframe,acceptableMissing,endOfWaterYear,coloursQLD,sizes,imageFormat)

#-------------Exceedence graphs----------------#

#cuts off data less then 0.5ML/d
obsx, obsy, modx, mody = FormatExceedenceData(dataframe,0.5)
makeLowExceedance(obsx, obsy, modx, mody, coloursQLD,sizes,imageFormat)

#cuts off data less then 0.0ML/d, or in other words, removes no data execpt NaNs
obsx, obsy, modx, mody = FormatExceedenceData(dataframe,0.0)
makeHighExceedance(obsx, obsy, modx, mody, coloursQLD,sizes,imageFormat)

#-------------Residual mass series graph----------------#

dataframeRM = FormatData(dataframe)
makeResidualGraph(dataframeRM,coloursQLD,sizes,imageFormat)  