import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from datetime import timedelta



def parseFiles(obs_File, rain_File, mod_File,dateFormat):

    dateparser = lambda x: pd.datetime.strptime(x, str(dateFormat))
    # if you get an error message saying: TypeError: strptime() argument 1 must be str, not float, that means that your file has
    #empty commas in in (if you edit it in exel, this problem happens). To fix, remove the empty comma seperators from your input file
    
    df_obs = pd.read_csv(obs_File, index_col=0, sep=',', parse_dates=[0], date_parser=dateparser)
    df_obs.columns.values[0] = 'ObservedFlow'
    df_obs.columns.values[1] = 'ObservedQuality'
    
    df_mod = pd.read_csv(mod_File, index_col=0, sep=',', parse_dates=[0], date_parser=dateparser)
    df_mod.columns.values[0] = 'ModelledFlow'
    if (len(df_mod.columns)>1):
        df_mod.columns.values[1] = 'ModelledQuality'
    
    df_rain = pd.read_csv(rain_File, index_col=0, sep=',', parse_dates=[0], date_parser=dateparser)
    df_rain.columns.values[0] = 'Rainfall'
    if (len(df_rain.columns)>1):
        df_rain.columns.values[1] = 'RainfallQuality'
        
    frames = [df_obs, df_mod, df_rain]
    df_obsModRain = pd.concat(frames, axis=1, join="outer", ignore_index=False, keys=None, levels=None, copy=True)
    df_obsModRain.reset_index(inplace=True) 
    df_obsModRain.columns.values[0] = 'Date'
    return(df_obsModRain)
                        
#------------------------------------Largest flood graphs----------------------------------#
def FloodDataManipulation(dataframe):
    dataframe['MaxFlow']= dataframe.ObservedFlow #Creates a column used to find second largest flood
    return(dataframe)

def findFlood(dataframe):
    maxObs = dataframe['MaxFlow'].max() #used to find the observed flow
    maxObsDate = dataframe.loc[dataframe.MaxFlow == maxObs]['Date'].iloc[0] #Finds the date corrisponding to the maxObs
    return(maxObsDate)

def MakeFloodGraph(floodNo,maxObsDate,dataframe,colours,sizes,imageFormat):
    date = dataframe.Date
    obs = dataframe.ObservedFlow
    mod = dataframe.ModelledFlow
    rain = dataframe.Rainfall
    fig = plt.figure(figsize=(9.50,4.30))                      
    ax = fig.add_subplot(111)
                  
    ln1 = ax.plot(date, obs/1000, color=colours['obs'], label = 'obs', linewidth=2)
    ln2 = ax.plot(date, mod/1000, color=colours['mod'], label = 'mod', linewidth=2)
    ax2 = ax.twinx() #adds second axes for rainfall
    ln3 = ax2.plot(date, rain, color=colours['rain'], label = 'rainfall', linewidth=2)

    #Title formatting
    fig.suptitle('Largest Flood #'+str(floodNo), fontsize=sizes['title'], color=colours['base'], fontweight='bold', fontname = 'Calibri')
    ax2.set_ylabel('Rainfall (mm/day)', color=colours['rain'],fontsize=sizes['yLabel'], fontname = 'Calibri')
        
        
    #Sets x limits to 20 days before lagerst flood to 30 days after
    minDate = (maxObsDate-timedelta(days=20))
    maxDate = (maxObsDate+timedelta(days=30)) 
    
    #set graph limits and place dates
    ax.set_xlim(minDate,maxDate+(timedelta(days=4))) #extended 4 days after so that right ticks and bottom ticks don't overlap
    ax.set_xticks([minDate, minDate+timedelta(days=10),minDate+timedelta(days=20),
                  minDate+timedelta(days=30),minDate+timedelta(days=40),minDate+timedelta(days=50)])
    #Formats dates
    dates_fmt = mdates.DateFormatter('%d/%m/%Y')
    ax.xaxis.set_major_formatter(dates_fmt) 
    
    #legend
    lns = ln1+ln2+ln3
    labs = [l.get_label() for l in lns] #uses l in lines as a counter
    legend = ax.legend(lns, labs,loc='best', frameon=False, fontsize=sizes['legend'])
    legend_texts = legend.get_texts() # list of matplotlib Text instances.
    legend_texts[0].set_color(colours['base'])
    legend_texts[1].set_color(colours['base'])
    legend_texts[2].set_color(colours['base'])

    #formatting for axis 1
    ax.grid( linestyle=':', linewidth=2, color='lightgrey') #adds grid and defines style
    ax.set_ylabel('Flow(GL/d)', fontsize=sizes['yLabel'], color=colours['base'],fontname = 'Calibri')
    
    ax.tick_params(axis='both', which='major',labelsize=sizes['ticks'], labelcolor=colours['base'] )
    ax.yaxis.set_tick_params(rotation=90)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
        
    ax.spines['right'].set_color('none') #removes the right border
    ax.spines['top'].set_color('none') #removes the top border
    

    #formatting for axis 2

    ax2.set_ylim(1000,-15) #Lower limit is set to negative to avoid cutting off rainfall line
    ax2.grid(  color='none')
    ax2.spines['left'].set_color(colours['base']) #because ax2 overlaps ax, ax2 needs to be coloured for the colours to show up.
    ax2.spines['bottom'].set_color(colours['base'])
    ax2.spines['right'].set_color(colours['rain'])
    ax2.spines['top'].set_color('none')
    
    ax2.tick_params(axis='y',labelsize=sizes['ticks'], labelcolor=colours['rain'])
    ax2.yaxis.set_tick_params(rotation=90,labelsize=sizes['ticks'])
    ax2.plot(clip_on=False)
    
    #export file
    plt.savefig('FloodGraph'+str(floodNo)+imageFormat,dpi=300, bbox_extra_artists=(ln3))

#This is used to mask max observation values, to allow you to create graphs for the 2nd, 3rd, 4th ect. largest flood
def MaskFlood(dataframe, maxObsDate): #May not cope with a largest flood near the start or end of time series
 
    minDate = (maxObsDate-timedelta(days=21))
    maxDate = (maxObsDate+timedelta(days=30))
    
    min_index = dataframe[dataframe['Date'] ==minDate].index.tolist()[0] #finds index of min date
    max_index = dataframe[dataframe['Date'] ==maxDate].index.tolist()[0] #finds index of max date
      
    dataframe.loc[min_index:max_index, 'MaxFlow']=0 #sets MaxFlow to zero between selected dates

def BuildLargestFloods(dataframe,floodNo,colours,sizes,imageFormat):
    
    dataframe = FloodDataManipulation(dataframe)
    
    counter = 1
    while (floodNo>=counter):
        maxObsDate = findFlood(dataframe) #find largest flood
        MakeFloodGraph(counter,maxObsDate,dataframe, colours,sizes,imageFormat) #make largest flood graph
        MaskFlood(dataframe, maxObsDate) #hide largest flood
        counter = counter+1


#------------------------------------Annual time series graph----------------------------------#

#Reads Data from three files and extracts it into datasets. It then manipultes the data and returns two dictionaries and a set.
def getFlow(dataframe,acceptableMissing, endOfWaterYear):
 
    setUndotted = set()
        
    #build month/year/water year columns
    dataframe['Month'] = dataframe['Date'].dt.strftime('%m').astype(int) 
    dataframe['Year'] = dataframe['Date'].dt.strftime('%Y').astype(int)
    dataframe['WaterYear'] = dataframe['Year'] + dataframe['Month'].apply(lambda x: -1 if x<=endOfWaterYear else 0)

    temp_df = dataframe[dataframe['ObservedQuality']<= 139]
    temp_df = temp_df.groupby(['WaterYear'])['ObservedQuality'].count()
    data = temp_df.to_dict()

    for key,val in data.items():
        year=key+1 #check if next year is a leap year (cos years is already in water years, so a leap year would affect the wateryear before it)
        days=365
        if (year % 4) == 0:
            if (year % 100) == 0:
                if (year % 400) == 0:
                    days = 366;
               
        if (val>=(days-acceptableMissing)):
            setUndotted.add(key)

    #Gets total modeled and observed by water year, and convert to GL
    dataframe['obsTotal'] = dataframe.groupby(['WaterYear'])['ObservedFlow'].transform(sum)/1000
    dataframe['modTotal'] = dataframe.groupby(['WaterYear'])['ModelledFlow'].transform(sum)/1000
   
    obsTotals_dict = dict(zip(dataframe.WaterYear, dataframe.obsTotal))
    modTotals_dict = dict(zip(dataframe.WaterYear, dataframe.modTotal))
    
    return(setUndotted, obsTotals_dict, modTotals_dict)
    

#creates two sets of datapoints, one for missing data and one for non-missing. It ensures the data is visually linked by
#overlapping where the two lines connect.
def createDatapoints(setUndotted, dictionary):
    dates = []
    valuesOk = []
    valuesDotted = []
    
    years = [y for y in dictionary.keys()]
    years.sort() #Gets years the dictonary covers

    for y in years:
        dates.append(np.datetime64(str(y)+'-07-01')) #creates three datapoints, the beginning,
        dates.append(np.datetime64(str(y+1)+'-01-01')) #Middle
        dates.append(np.datetime64(str(y+1)+'-06-30')) #And end of a water year. Used to make the graph square
        v = dictionary[y]
        if (y not in setUndotted): #the current year is dotted
            
            if (y-1) in setUndotted: #the year before is undotted
                valuesOk.append(v) #add this year's beginning point to undotted
                
            else:#the year before is dotted
                valuesOk.append(None) #Don't add this year point to undotted
                
            valuesOk.append(None) #This year is confirmed to be dotted, so it's midpoint isn't added to undotted
            
            if (y+1) in setUndotted: #if the year after isn't dotted
                valuesOk.append(v) #add this year's end point to undotted
            else:
                valuesOk.append(None)   
                
            valuesDotted.append(v) #the current year is dotted so it's beginning
            valuesDotted.append(v) #Middle
            valuesDotted.append(v) #and end point are added to dotted
        
        else:  #if the current year is undotted
            valuesDotted.append(None) #don't add beginning
            valuesDotted.append(None) #Middle
            valuesDotted.append(None) #or end to dotted
            
            valuesOk.append(v) #add this years beginning
            valuesOk.append(v) #Middle
            valuesOk.append(v) #and end to undotted
    
    
    data = pd.DataFrame(data={'dates':dates,'vals':valuesOk})   
    dotted = pd.DataFrame(data={'dates':dates,'vals':valuesDotted})
    
    return data, dotted

def MakeAnnualGraph(obsDotted, modDotted, obsTotals, modTotals, colours,sizes,imageFormat):
    
    fig = plt.figure(figsize=(12.50,4.30))
    ax = fig.add_subplot(111)
    
    #plots data, one line for solid colour and one line for dotted
    obsDotted = ax.plot(obsDotted['dates'].dt.strftime('%Y').astype(int), obsDotted['vals'],':', color=colours['obs'])
    obsLine = ax.plot(obsTotals['dates'].dt.strftime('%Y').astype(int), obsTotals['vals'], color=colours['obs'], label = 'obs')
    
    modDotted = ax.plot(modDotted['dates'].dt.strftime('%Y').astype(int), modDotted['vals'],':', color=colours['mod'])  
    modLine = ax.plot(modTotals['dates'].dt.strftime('%Y').astype(int), modTotals['vals'], color=colours['mod'], label = 'mod')

    #Titles
    fig.suptitle('Annual time series (July to June)', fontsize=sizes['title'], color=colours['base'], fontweight='bold', fontname = 'Calibri')
    ax.set_title('Years with missing data represented with dotted lines',y=1.0, pad=-14,fontsize=sizes['subtitle'], color=colours['base'], fontname = 'Calibri')
    ax.set_ylabel('Flow(GL/y)', fontsize=sizes['yLabel'], color=colours['base'],fontname = 'Calibri')

    #legend
    legend = ax.legend(loc=0,frameon=False, fontsize=sizes['legend'])
    legend_texts = legend.get_texts() # list of matplotlib Text instances.
    legend_texts[0].set_color(colours['base'])
    legend_texts[1].set_color(colours['base'])

    
    #formatting ticks 
    ax.tick_params(axis='both', which='major', labelsize=sizes['ticks'],labelcolor=colours['base'] )
    plt.yticks(rotation=90) #rotates y-axis
    
    #formatting borders/lines
    ax.grid(linestyle=':', linewidth=2, color='lightgrey') #adds grid and defines style
    ax.spines['right'].set_color('none') #removes the right border
    ax.spines['top'].set_color('none') #removes the top border
    ax.spines['left'].set_color(colours['base']) 
    ax.spines['bottom'].set_color(colours['base'])

    
    #export file
    plt.savefig('AnnualGraph'+imageFormat,dpi=300, bbox_inches='tight')
    plt.show()
    
def BuildAnnualGraph(dataframe,acceptableMissing,endOfWaterYear,colours,sizes,imageFormat):
    setUndotted, modTotals, obsTotals = getFlow(dataframe, acceptableMissing,endOfWaterYear)
    obsTotals, obsDotted = createDatapoints(setUndotted, obsTotals)
    modTotals, modDotted = createDatapoints(setUndotted, modTotals)
    MakeAnnualGraph(obsDotted, modDotted, obsTotals, modTotals, colours,sizes,imageFormat)



#------------------------------------Exceedence graphs----------------------------------#

def FormatExceedenceData(dataframe, cutoff):
    dataframe['ObservedFlow'] = pd.to_numeric(dataframe['ObservedFlow'], errors='coerce')
    dataframe = dataframe.dropna(subset=['ObservedFlow']) #drops NaNs in observed flow
    dataframe['ModelledFlow'] = pd.to_numeric(dataframe['ModelledFlow'], errors='coerce')
    dataframe = dataframe.dropna(subset=['ModelledFlow'])  #drops NaNs in Modelled flow
    obs = dataframe['ObservedFlow'].tolist()
    mod = dataframe['ModelledFlow'].tolist()
    
    obs.sort(reverse=True)
    mod.sort(reverse=True)
    obsx = []
    modx = []
    obsy = []
    mody = []
    size = len(obs)
    for idx, val in enumerate(obs):
        if (val>=cutoff):
            obsy.append(val)
            obsx.append((idx+1)/size)

    for idx, val in enumerate(mod):
        if (val>=cutoff):
            mody.append(val)
            modx.append((idx+1)/size) 
   
    return(obsx,obsy,modx,mody)

def makeHighExceedance(obsx, obsy, modx, mody, colours,sizes,imageFormat):
    fig = plt.figure(figsize=(9,4.30))
    ax = fig.add_subplot(111)
    
    ax.plot(obsx, obsy, label='obs', color=colours['obs'],linewidth=2)
    ax.plot(modx, mody, label='mod', color=colours['mod'],linewidth=2)

    
    #Titles
    fig.suptitle('Exceedence curve showing high flow', fontsize=sizes['title'], color=colours['base'], fontweight='bold', fontname = 'Calibri')   
    ax.set_ylabel('Flow (ML/d)', fontsize=sizes['yLabel'], color=colours['base'],fontname = 'Calibri')
    ax.set_xlabel('Fraction of time flow is equalled or exceeded - LOG scale', fontsize=sizes['xLabel'], color=colours['base'],fontname = 'Calibri')
   
    
    #legend
    legend = ax.legend(loc=0,frameon=False, fontsize=sizes['legend'])
    legend_texts = legend.get_texts() # list of matplotlib Text instances.
    legend_texts[0].set_color(colours['base'])
    legend_texts[1].set_color(colours['base'])
    
    #ticks
    ax.tick_params(axis='both', which='major', labelsize=sizes['ticks'], labelcolor=colours['base'] )
    plt.yticks(rotation=90) #rotates y-axis
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    #borders/lines
    ax.grid(linestyle=':', linewidth=2, color='lightgrey') #adds grid and defines style
    ax.spines['right'].set_color('none') #removes the right border
    ax.spines['top'].set_color('none') #removes the top border
    ax.spines['left'].set_color(colours['base']) 
    ax.spines['bottom'].set_color(colours['base'])
    plt.axvline(x=0.02,linestyle='--', color=colours['base'])
    plt.axvline(x=0.1,linestyle='--', color=colours['base'])
    
    #export file
    plt.savefig('HighExceedanceGraph'+imageFormat,dpi=300, bbox_inches='tight')
    plt.show()
    
def makeLowExceedance(obsx, obsy, modx, mody, colours,sizes,imageFormat):
    fig = plt.figure(figsize=(9,4.30))
    ax = fig.add_subplot(111)
    
    ax.plot(obsx, obsy, label='obs', color=colours['obs'],linewidth=2)
    ax.plot(modx, mody, label='mod', color=colours['mod'],linewidth=2)

    
    #Titles
    fig.suptitle('Exceedence curve showing low flow', fontsize=sizes['title'], color=colours['base'], fontweight='bold', fontname = 'Calibri')   
    ax.set_ylabel('Flow (ML/d) - LOG scale', fontsize=sizes['yLabel'], color=colours['base'],fontname = 'Calibri')
    ax.set_xlabel('Fraction of time flow is equalled or exceeded', fontsize=sizes['xLabel'], color=colours['base'],fontname = 'Calibri')
   
    
    #legend
    legend = ax.legend(loc=0,frameon=False, fontsize=sizes['legend'])
    legend_texts = legend.get_texts() # list of matplotlib Text instances.
    legend_texts[0].set_color(colours['base'])
    legend_texts[1].set_color(colours['base'])
    
    #ticks
    ax.tick_params(axis='both', which='major', labelsize=sizes['ticks'], labelcolor=colours['base'] )
    plt.yticks(rotation=90) #rotates y-axis
    ax.set_yscale('log')
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)

    
    #borders/lines
    ax.grid(linestyle=':', linewidth=2, color='lightgrey') #adds grid and defines style
    ax.spines['right'].set_color('none') #removes the right border
    ax.spines['top'].set_color('none') #removes the top border
    ax.spines['left'].set_color(colours['base']) 
    ax.spines['bottom'].set_color(colours['base'])
    plt.axvline(x=0.02,linestyle='--', color=colours['base'])
    plt.axvline(x=0.1,linestyle='--', color=colours['base'])
    
    #export file    
    plt.savefig('LowExceedanceGraph'+imageFormat,dpi=300, bbox_inches='tight')
    plt.show()
  

#------------------------------------Residual mass series graph----------------------------------#

def FormatData(dataframe):    
     
    #observed caculations
    dataframe['obsMean'] = dataframe['ObservedFlow'].mean()
    dataframe.loc[dataframe['ObservedFlow'].isnull(), 'obsMean'] = float('NAN')
    
    dataframe['obsMeanCumlative'] = dataframe['obsMean'].cumsum()
    dataframe['obsCumulative'] = dataframe['ObservedFlow'].cumsum()
    dataframe['obsDiff'] = dataframe['obsCumulative'] -  dataframe['obsMeanCumlative']
    
    #modelled caculations
    dataframe['modMean'] = dataframe['ModelledFlow'].mean()
    dataframe.loc[dataframe['ModelledFlow'].isnull(), 'modMean'] = float('NAN')
    
    dataframe['modMeanCumlative'] = dataframe['modMean'].cumsum()
    dataframe['modCumulative'] = dataframe['ModelledFlow'].cumsum()
    dataframe['modDiff'] = dataframe['modCumulative'] -  dataframe['modMeanCumlative']
    
    #convert to GL
    dataframe['obsDiff'] = dataframe['obsDiff']
    dataframe['modDiff'] = dataframe['modDiff']
   
    return dataframe;
    
def makeResidualGraph(dataframe, colours,sizes,imageFormat):

    fig = plt.figure(figsize=(12.50,4.30))
    ax = fig.add_subplot(111)
    
    #draw line over 0
    plt.axhline(y=0, xmin=0, xmax=1,color="#000000")
                 
    #plots data, one line for solid colour and one line for dotted
    ax.plot(dataframe['Date'], dataframe['obsDiff']/1000, label='obs', color=colours['obs'])
    ax.plot(dataframe['Date'], dataframe['modDiff']/1000, label='mod', color=colours['mod'])

    #Titles
    fig.suptitle('Residual mass series', fontsize=sizes['title'], color=colours['base'], fontweight='bold', fontname = 'Calibri')
    ax.set_ylabel('Residual Mass (GL)', fontsize=sizes['yLabel'], color=colours['base'],fontname = 'Calibri')

    #legend
    legend = ax.legend(loc=0,frameon=False, fontsize=sizes['legend'])
    legend_texts = legend.get_texts() # list of matplotlib Text instances.
    legend_texts[0].set_color(colours['base'])
    legend_texts[1].set_color(colours['base'])
    
    #formatting ticks 
    ax.tick_params(axis='both', which='major', labelsize=sizes['ticks'],labelcolor=colours['base'] )
    plt.yticks(rotation=90) #rotates y-axis

    
    #formatting borders/lines
    ax.grid(linestyle=':', linewidth=2, color='lightgrey') #adds grid and defines style
    ax.spines['right'].set_color('none') #removes the right border
    ax.spines['top'].set_color('none') #removes the top border
    ax.spines['left'].set_color(colours['base']) 
    ax.spines['bottom'].set_color(colours['base'])
    
    ax.set_xlim(dataframe['Date'].iloc[0]-timedelta(days=365),dataframe['Date'].iloc[-1]+timedelta(days=365))

   
    #export file
    plt.savefig('ResidualMassGraph'+imageFormat,dpi=300, bbox_inches='tight')
    plt.show()
